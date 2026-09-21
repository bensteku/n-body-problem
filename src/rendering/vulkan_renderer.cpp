#include "rendering/vulkan_renderer.hpp"

#include "rendering/render_camera.hpp"
#include "rendering/render_diagnostics.hpp"
#include "rendering/render_graph.hpp"
#include "rendering/render_graph_executor.hpp"
#include "rendering/render_picking.hpp"
#include "rendering/render_style.hpp"
#include "rendering/vulkan_pipeline_manager.hpp"
#include "rendering/vulkan_upload_arena.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <span>
#include <vector>

namespace nbody::rendering {

namespace {

std::atomic_uint64_t validation_messages{};
std::atomic_uint64_t validation_warnings{};
std::atomic_uint64_t validation_errors{};

void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(operation) + " failed");
}

struct BodyVertex {
    float position[2];
    float color[4];
};

struct BodyInstance {
    float center[2];
    float radius[2];
    float color[4];
};

VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT && data && data->pMessage) {
        std::fputs(data->pMessage, stderr);
        std::fputc('\n', stderr);
    }
    validation_messages.fetch_add(1, std::memory_order_relaxed);
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        validation_errors.fetch_add(1, std::memory_order_relaxed);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        validation_warnings.fetch_add(1, std::memory_order_relaxed);
    }
    return VK_FALSE;
}

}

struct VulkanRenderer::Impl {
    VkInstance instance{};
    VkDebugUtilsMessengerEXT debug_messenger{};
    bool validation_enabled_{};
    VkSurfaceKHR surface{};
    VkPhysicalDevice physical_device{};
    VkDevice device{};
    VkQueue queue{};
    std::uint32_t queue_family{};
    VkSwapchainKHR swapchain{};
    VkFormat format{};
    VkFormat depth_format{VK_FORMAT_D32_SFLOAT};
    VkExtent2D extent{};
    std::vector<VkImageView> views;
    VkRenderPass render_pass{};
    std::vector<VkFramebuffer> framebuffers;
    VkImage depth_image{};
    VkDeviceMemory depth_memory{};
    VkImageView depth_view{};
    VkCommandPool command_pool{};
    std::vector<VkCommandBuffer> commands;
    VkSemaphore image_available{};
    std::vector<VkSemaphore> render_finished;
    VkFence fence{};
    VkDescriptorPool descriptor_pool{};
    VkPipeline body_pipeline{};
    VkPipelineLayout body_pipeline_layout{};
    VkPipeline transparent_body_pipeline{};
    VkPipelineLayout transparent_body_pipeline_layout{};
    VkDeviceSize body_upload_offset{};
    VkDeviceSize body_upload_size{};
    VkDeviceSize grid_upload_offset{};
    VkDeviceSize grid_upload_size{};
    VkDeviceSize trajectory_upload_offset{};
    VkDeviceSize trajectory_upload_size{};
    VkPipeline grid_pipeline{};
    VkPipeline trajectory_pipeline{};
    VulkanUploadArena upload_arena_;
    VulkanPipelineManager pipeline_manager_;
    std::vector<BodyInstance> body_instances_;
    std::size_t opaque_body_count_{};
    std::vector<BodyVertex> grid_vertices_;
    std::vector<BodyVertex> trajectory_vertices_;
    RenderDiagnostics diagnostics_;
    RenderGraphExecutor graph_executor_;
    std::uint64_t graph_key_{std::numeric_limits<std::uint64_t>::max()};
    RenderGraph logical_graph_;

    const RenderDiagnostics& diagnostics() const { return diagnostics_; }

    void initialize(GLFWwindow* window) {
        createInstance();
        check(glfwCreateWindowSurface(instance, window, nullptr, &surface), "window surface");
        selectDevice();
        createDevice();
        createSwapchain(window);
        createRenderPass();
        createDepthResources();
        createBodyRenderer();
        createFramebuffers();
        createCommands();
        createSync();
        initializeImGui(window);
    }

    void render(GLFWwindow* window, ImDrawData* draw_data,
                const ::nbody::rendering::RenderScene& scene) {
        const auto frame_start = std::chrono::steady_clock::now();
        const std::uint64_t graph_key =
            (scene.hasCpuBodies() || scene.hasExternalGpuResource() ? 1ull : 0ull)
            | (scene.settings.show_trajectories ? 2ull : 0ull)
            | (scene.settings.show_debug_overlays ? 4ull : 0ull)
            | (scene.settings.show_selection_outline ? 8ull : 0ull);
        if (graph_key != graph_key_) {
            logical_graph_ = ::nbody::rendering::RenderGraph::make2D(
                scene.hasCpuBodies() || scene.hasExternalGpuResource(),
                scene.settings.show_trajectories,
                scene.settings.show_debug_overlays,
                true,
                scene.settings.show_selection_outline);
            if (!graph_executor_.compile(logical_graph_)) {
                throw std::runtime_error("invalid logical render graph");
            }
            graph_key_ = graph_key;
        }
        diagnostics_.compiled_passes = graph_executor_.passes().size();
        diagnostics_.resource_barriers = graph_executor_.barriers().size();
        diagnostics_.validation_enabled = validation_enabled_;
        diagnostics_.validation_messages = validation_messages.load(std::memory_order_relaxed);
        diagnostics_.validation_warnings = validation_warnings.load(std::memory_order_relaxed);
        diagnostics_.validation_errors = validation_errors.load(std::memory_order_relaxed);
        check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "wait fence");
        int framebuffer_width = 0;
        int framebuffer_height = 0;
        glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
        if (framebuffer_width <= 0 || framebuffer_height <= 0) return;
        if (static_cast<std::uint32_t>(framebuffer_width) != extent.width
            || static_cast<std::uint32_t>(framebuffer_height) != extent.height) {
            recreateSwapchain(window);
        }
        upload_arena_.beginFrame(fence);
        buildBodyInstances(scene);
        buildGridVertices(scene);
        buildTrajectoryVertices(scene);
        prepareUploadLayout();
        diagnostics_.frame_count += 1;
        diagnostics_.submitted_bodies = body_instances_.size();
        diagnostics_.used_external_gpu_frame = scene.hasExternalGpuResource();
        uploadBodyInstances();
        uploadGridVertices();
        uploadTrajectoryVertices();
        diagnostics_.cpu_prepare_milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - frame_start).count();
        std::uint32_t image = 0;
        const VkResult acquired = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                                         image_available, VK_NULL_HANDLE, &image);
        if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain(window);
            return;
        }
        check(acquired, "acquire swapchain image");
        check(vkResetFences(device, 1, &fence), "reset fence");
        check(vkResetCommandBuffer(commands[image], 0), "reset command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(commands[image], &begin), "begin command buffer");
        VkClearValue clear[2]{};
        clear[0].color = {{0.02f, 0.03f, 0.06f, 1.0f}};
        clear[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        pass.renderPass = render_pass;
        pass.framebuffer = framebuffers[image];
        pass.renderArea.extent = extent;
        pass.clearValueCount = 2;
        pass.pClearValues = clear;
        vkCmdBeginRenderPass(commands[image], &pass, VK_SUBPASS_CONTENTS_INLINE);
        diagnostics_.pass_cpu_milliseconds.fill(0.0);
        const auto pass_start = [&]() { return std::chrono::steady_clock::now(); };
        const auto finish_pass = [&](std::size_t index, auto started) {
            diagnostics_.pass_cpu_milliseconds[index] = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - started).count() / 1000.0;
        };
        const auto setViewport = [&]() {
            VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent.width),
                                static_cast<float>(extent.height), 0.0f, 1.0f};
            VkRect2D scissor{{0, 0}, extent};
            vkCmdSetViewport(commands[image], 0, 1, &viewport);
            vkCmdSetScissor(commands[image], 0, 1, &scissor);
        };
        // The native backend fuses the logical passes into one subpass. Draw
        // the transparent grid before bodies so it cannot overpaint them.
        auto started = pass_start();
        if (graph_executor_.hasPass(RenderPassKind::GridAndDebug) && !grid_vertices_.empty()) {
            setViewport();
            vkCmdBindPipeline(commands[image], VK_PIPELINE_BIND_POINT_GRAPHICS, grid_pipeline);
            const VkDeviceSize offset = grid_upload_offset;
            const VkBuffer upload_buffer = upload_arena_.buffer();
            vkCmdBindVertexBuffers(commands[image], 0, 1, &upload_buffer, &offset);
            vkCmdDraw(commands[image], static_cast<std::uint32_t>(grid_vertices_.size()), 1, 0, 0);
        }
        finish_pass(0, started);
        // Subpass 0: depth prepass. It is currently reserved for future
        // depth-only geometry; keeping it explicit makes the 2D order stable.
        started = pass_start();
        if (graph_executor_.hasPass(RenderPassKind::OpaqueBodies) && opaque_body_count_ != 0) {
            setViewport();
            vkCmdBindPipeline(commands[image], VK_PIPELINE_BIND_POINT_GRAPHICS, body_pipeline);
            const VkDeviceSize offset = body_upload_offset;
            const VkBuffer upload_buffer = upload_arena_.buffer();
            vkCmdBindVertexBuffers(commands[image], 0, 1, &upload_buffer, &offset);
            vkCmdDraw(commands[image], 6, static_cast<std::uint32_t>(opaque_body_count_), 0, 0);
        }
        finish_pass(1, started);
        // Subpass 2: alpha-blended bodies, after opaque depth writes.
        started = pass_start();
        if (graph_executor_.hasPass(RenderPassKind::TransparentBodies)
            && body_instances_.size() > opaque_body_count_) {
            setViewport();
            vkCmdBindPipeline(commands[image], VK_PIPELINE_BIND_POINT_GRAPHICS,
                              transparent_body_pipeline);
            const VkDeviceSize offset = body_upload_offset
                + static_cast<VkDeviceSize>(opaque_body_count_ * sizeof(BodyInstance));
            const VkBuffer upload_buffer = upload_arena_.buffer();
            vkCmdBindVertexBuffers(commands[image], 0, 1, &upload_buffer, &offset);
            vkCmdDraw(commands[image], 6,
                      static_cast<std::uint32_t>(body_instances_.size() - opaque_body_count_), 0, 0);
        }
        finish_pass(2, started);
        // Logical grid/debug work was fused into the initial native pass.
        // Subpass 4: trajectories/orbits.
        started = pass_start();
        if (graph_executor_.hasPass(RenderPassKind::Trajectories) && !trajectory_vertices_.empty()) {
            setViewport();
            vkCmdBindPipeline(commands[image], VK_PIPELINE_BIND_POINT_GRAPHICS,
                              trajectory_pipeline);
            const VkDeviceSize offset = trajectory_upload_offset;
                const VkBuffer upload_buffer = upload_arena_.buffer();
                vkCmdBindVertexBuffers(commands[image], 0, 1, &upload_buffer, &offset);
            vkCmdDraw(commands[image], static_cast<std::uint32_t>(trajectory_vertices_.size()), 1, 0, 0);
        }
        finish_pass(4, started);
        // Subpass 5: selection outlines. This is a separate composition stage
        // even while the outline geometry is still pending.
        started = pass_start();
        if (graph_executor_.hasPass(RenderPassKind::SelectionOutlines)
            && scene.settings.show_selection_outline && scene.selected_body) {
            // Reserved for the selection-outline renderer.
        }
        finish_pass(5, started);
        // Subpass 6: UI composition.
        started = pass_start();
        if (graph_executor_.hasPass(RenderPassKind::UserInterface)) {
            ImGui_ImplVulkan_RenderDrawData(draw_data, commands[image]);
        }
        finish_pass(6, started);
        vkCmdEndRenderPass(commands[image]);
        check(vkEndCommandBuffer(commands[image]), "end command buffer");
        diagnostics_.command_record_milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - frame_start).count();

        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &image_available;
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commands[image];
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &render_finished[image];
        check(vkQueueSubmit(queue, 1, &submit, fence), "submit command buffer");
        upload_arena_.endFrame(fence);

        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &render_finished[image];
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain;
        present.pImageIndices = &image;
        const VkResult presented = vkQueuePresentKHR(queue, &present);
        if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) {
            recreateSwapchain(window);
        } else {
            check(presented, "present");
        }
        diagnostics_.frame_milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - frame_start).count();
        const double instantaneous_fps = diagnostics_.frame_milliseconds > 0.0
            ? 1000.0 / diagnostics_.frame_milliseconds : 0.0;
        diagnostics_.frames_per_second = diagnostics_.frames_per_second == 0.0
            ? instantaneous_fps
            : diagnostics_.frames_per_second * 0.9 + instantaneous_fps * 0.1;
    }

    ~Impl() {
        if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device);
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        destroyBodyRenderer();
        if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (image_available) vkDestroySemaphore(device, image_available, nullptr);
        for (VkSemaphore semaphore : render_finished) {
            if (semaphore) vkDestroySemaphore(device, semaphore, nullptr);
        }
        render_finished.clear();
        if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
        for (VkFramebuffer framebuffer : framebuffers) vkDestroyFramebuffer(device, framebuffer, nullptr);
        if (render_pass) vkDestroyRenderPass(device, render_pass, nullptr);
        if (depth_view) vkDestroyImageView(device, depth_view, nullptr);
        if (depth_image) vkDestroyImage(device, depth_image, nullptr);
        if (depth_memory) vkFreeMemory(device, depth_memory, nullptr);
        for (VkImageView view : views) vkDestroyImageView(device, view, nullptr);
        if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (debug_messenger) {
            auto destroy_debug = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy_debug) destroy_debug(instance, debug_messenger, nullptr);
        }
        if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }

private:
    void createInstance() {
        std::uint32_t count = 0;
        const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&count);
        if (!glfw_extensions) throw std::runtime_error("GLFW Vulkan extensions unavailable");
        std::vector<const char*> extensions(glfw_extensions, glfw_extensions + count);
        std::uint32_t layer_count = 0;
        vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
        std::vector<VkLayerProperties> layers(layer_count);
        vkEnumerateInstanceLayerProperties(&layer_count, layers.data());
        for (const VkLayerProperties& layer : layers) {
            if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                validation_enabled_ = true;
                break;
            }
        }
        if (validation_enabled_) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        std::vector<const char*> enabled_layers;
        if (validation_enabled_) enabled_layers.push_back("VK_LAYER_KHRONOS_validation");
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.pApplicationName = "N-Body Simulator";
        application.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        application.pEngineName = "N-Body Greenfield";
        application.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        application.apiVersion = VK_API_VERSION_1_0;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        info.pApplicationInfo = &application;
        info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
        info.ppEnabledExtensionNames = extensions.data();
        info.enabledLayerCount = static_cast<std::uint32_t>(enabled_layers.size());
        info.ppEnabledLayerNames = enabled_layers.data();
        VkDebugUtilsMessengerCreateInfoEXT debug_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        debug_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug_info.pfnUserCallback = vulkanDebugCallback;
        if (validation_enabled_) info.pNext = &debug_info;
        check(vkCreateInstance(&info, nullptr, &instance), "create Vulkan instance");
        if (validation_enabled_) {
            auto create_debug = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
            if (create_debug) check(create_debug(instance, &debug_info, nullptr, &debug_messenger),
                                    "create Vulkan debug messenger");
        }
    }

    bool suitable(VkPhysicalDevice candidate, std::uint32_t& selected_family) const {
        std::uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, families.data());
        for (std::uint32_t index = 0; index < count; ++index) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(candidate, index, surface, &present);
            if ((families[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                std::uint32_t formats = 0;
                std::uint32_t modes = 0;
                vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface, &formats, nullptr);
                vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface, &modes, nullptr);
                if (formats && modes) {
                    selected_family = index;
                    return true;
                }
            }
        }
        return false;
    }

    void selectDevice() {
        std::uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        if (!count) throw std::runtime_error("no Vulkan device available");
        std::vector<VkPhysicalDevice> candidates(count);
        vkEnumeratePhysicalDevices(instance, &count, candidates.data());
        for (VkPhysicalDevice candidate : candidates) {
            if (suitable(candidate, queue_family)) {
                physical_device = candidate;
                return;
            }
        }
        throw std::runtime_error("no Vulkan graphics/present queue available");
    }

    void createDevice() {
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = queue_family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        const char* extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        info.queueCreateInfoCount = 1;
        info.pQueueCreateInfos = &queue_info;
        info.enabledExtensionCount = 1;
        info.ppEnabledExtensionNames = extensions;
        check(vkCreateDevice(physical_device, &info, nullptr, &device), "create Vulkan device");
        vkGetDeviceQueue(device, queue_family, 0, &queue);
    }

    void createSwapchain(GLFWwindow* window) {
        VkSurfaceCapabilitiesKHR capabilities{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities);
        std::uint32_t format_count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(format_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, formats.data());
        format = formats.front().format;
        for (const VkSurfaceFormatKHR candidate : formats) {
            if (candidate.format == VK_FORMAT_B8G8R8A8_SRGB) format = candidate.format;
        }
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        extent = capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()
            ? capabilities.currentExtent
            : VkExtent2D{static_cast<std::uint32_t>(std::max(width, 1)), static_cast<std::uint32_t>(std::max(height, 1))};
        std::uint32_t image_count = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount) image_count = std::min(image_count, capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        info.surface = surface;
        info.minImageCount = image_count;
        info.imageFormat = format;
        info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        info.imageExtent = extent;
        info.imageArrayLayers = 1;
        info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.preTransform = capabilities.currentTransform;
        info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        info.clipped = VK_TRUE;
        check(vkCreateSwapchainKHR(device, &info, nullptr, &swapchain), "create swapchain");
        std::uint32_t actual_count = 0;
        vkGetSwapchainImagesKHR(device, swapchain, &actual_count, nullptr);
        std::vector<VkImage> images(actual_count);
        vkGetSwapchainImagesKHR(device, swapchain, &actual_count, images.data());
        views.resize(images.size());
        for (std::size_t index = 0; index < images.size(); ++index) {
            VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view.image = images[index];
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = format;
            view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            check(vkCreateImageView(device, &view, nullptr, &views[index]), "create image view");
        }
    }

    void createRenderPass() {
        VkAttachmentDescription color{};
        color.format = format;
        color.samples = VK_SAMPLE_COUNT_1_BIT;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentDescription depth{};
        depth.format = depth_format;
        depth.samples = VK_SAMPLE_COUNT_1_BIT;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depth_reference{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &reference;
        subpass.pDepthStencilAttachment = &depth_reference;
        VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        const VkAttachmentDescription attachments[] = {color, depth};
        info.attachmentCount = 2;
        info.pAttachments = attachments;
        info.subpassCount = 1;
        info.pSubpasses = &subpass;
        check(vkCreateRenderPass(device, &info, nullptr, &render_pass), "create render pass");
    }

    void destroyUploadBuffer() {
        upload_arena_.shutdown();
        body_upload_offset = 0;
        body_upload_size = 0;
        grid_upload_offset = 0;
        grid_upload_size = 0;
        trajectory_upload_offset = 0;
        trajectory_upload_size = 0;
    }

    void destroyBodyRenderer() {
        destroyUploadBuffer();
        pipeline_manager_.destroyAll();
        body_pipeline = VK_NULL_HANDLE;
        grid_pipeline = VK_NULL_HANDLE;
        trajectory_pipeline = VK_NULL_HANDLE;
        body_pipeline_layout = VK_NULL_HANDLE;
        transparent_body_pipeline = VK_NULL_HANDLE;
        transparent_body_pipeline_layout = VK_NULL_HANDLE;
    }

    void destroySwapchain() {
        for (VkFramebuffer framebuffer : framebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        framebuffers.clear();
        if (depth_view) {
            vkDestroyImageView(device, depth_view, nullptr);
            depth_view = VK_NULL_HANDLE;
        }
        if (depth_image) {
            vkDestroyImage(device, depth_image, nullptr);
            depth_image = VK_NULL_HANDLE;
        }
        if (depth_memory) {
            vkFreeMemory(device, depth_memory, nullptr);
            depth_memory = VK_NULL_HANDLE;
        }
        for (VkImageView view : views) vkDestroyImageView(device, view, nullptr);
        views.clear();
        if (render_pass) {
            vkDestroyRenderPass(device, render_pass, nullptr);
            render_pass = VK_NULL_HANDLE;
        }
        if (swapchain) {
            vkDestroySwapchainKHR(device, swapchain, nullptr);
            swapchain = VK_NULL_HANDLE;
        }
    }

    void createDepthResources() {
        VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = depth_format;
        image.extent = {extent.width, extent.height, 1};
        image.mipLevels = 1;
        image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        check(vkCreateImage(device, &image, nullptr, &depth_image), "create depth image");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device, depth_image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(device, &allocation, nullptr, &depth_memory), "allocate depth memory");
        check(vkBindImageMemory(device, depth_image, depth_memory, 0), "bind depth memory");
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = depth_image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = depth_format;
        view.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(device, &view, nullptr, &depth_view), "create depth view");
    }

    void recreateSwapchain(GLFWwindow* window) {
        vkDeviceWaitIdle(device);
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        destroyRenderFinishedSemaphores();
        destroyBodyRenderer();
        destroySwapchain();
        destroyCommands();
        createSwapchain(window);
        createRenderFinishedSemaphores();
        createRenderPass();
        createDepthResources();
        createBodyRenderer();
        createFramebuffers();
        createCommands();
        initializeImGuiBackend(window);
    }

    std::uint32_t findMemoryType(std::uint32_t filter, VkMemoryPropertyFlags properties) const {
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memory);
        for (std::uint32_t index = 0; index < memory.memoryTypeCount; ++index) {
            if ((filter & (1u << index))
                && (memory.memoryTypes[index].propertyFlags & properties) == properties) {
                return index;
            }
        }
        throw std::runtime_error("no compatible Vulkan memory type");
    }

    void prepareUploadLayout() {
        body_upload_size = static_cast<VkDeviceSize>(body_instances_.size() * sizeof(BodyInstance));
        grid_upload_size = static_cast<VkDeviceSize>(grid_vertices_.size() * sizeof(BodyVertex));
        trajectory_upload_size = static_cast<VkDeviceSize>(
            trajectory_vertices_.size() * sizeof(BodyVertex));
        const VulkanUploadAllocation body = upload_arena_.allocate(body_upload_size);
        const VulkanUploadAllocation grid = upload_arena_.allocate(grid_upload_size);
        const VulkanUploadAllocation trajectory = upload_arena_.allocate(trajectory_upload_size);
        body_upload_offset = body.offset;
        grid_upload_offset = grid.offset;
        trajectory_upload_offset = trajectory.offset;
        diagnostics_.uploaded_bytes = body_upload_size + grid_upload_size + trajectory_upload_size;
    }

    void uploadBodyInstances() {
        if (body_upload_size != 0) {
            upload_arena_.write({body_upload_offset, body_upload_size},
                std::as_bytes(std::span<const BodyInstance>(body_instances_.data(), body_instances_.size())));
        }
    }

    void uploadGridVertices() {
        if (grid_upload_size != 0) {
            upload_arena_.write({grid_upload_offset, grid_upload_size},
                std::as_bytes(std::span<const BodyVertex>(grid_vertices_.data(), grid_vertices_.size())));
        }
    }

    void uploadTrajectoryVertices() {
        if (trajectory_upload_size != 0) {
            upload_arena_.write({trajectory_upload_offset, trajectory_upload_size},
                std::as_bytes(std::span<const BodyVertex>(
                    trajectory_vertices_.data(), trajectory_vertices_.size())));
        }
    }

    void buildBodyInstances(const ::nbody::rendering::RenderScene& scene) {
        body_instances_.clear();
        opaque_body_count_ = 0;
        if (!scene.hasCpuBodies()) return;
        const float width = std::max(1.0f, scene.viewport_width);
        const float height = std::max(1.0f, scene.viewport_height);
        const ::nbody::rendering::RenderViewport viewport{width, height};
        body_instances_.reserve(scene.cpuBodies().size());
        for (const RenderBody& body : scene.cpuBodies()) {
            const auto center = ::nbody::rendering::worldToClip(
                scene.camera, viewport, body.position.x, body.position.y);
            const bool selected = scene.selected_body && *scene.selected_body == body.id;
            const bool focused = scene.focused_body && *scene.focused_body == body.id;
            const auto style = ::nbody::rendering::styleFor(body, selected, focused);
            const auto radius = ::nbody::rendering::worldRadiusToClip(
                scene.camera, viewport, body.radius);
            body_instances_.push_back({
                {center[0], center[1]},
                {radius[0], radius[1]},
                {static_cast<float>(style.color.x), static_cast<float>(style.color.y),
                 static_cast<float>(style.color.z), style.alpha}});
        }
        const auto first_transparent = std::stable_partition(
            body_instances_.begin(), body_instances_.end(),
            [](const BodyInstance& instance) { return instance.color[3] >= 0.999f; });
        opaque_body_count_ = static_cast<std::size_t>(
            std::distance(body_instances_.begin(), first_transparent));
    }

    void buildGridVertices(const ::nbody::rendering::RenderScene& scene) {
        grid_vertices_.clear();
        if (!scene.settings.grid.visible || scene.settings.grid.spacing <= 0.0
            || scene.viewport_width <= 0.0f || scene.viewport_height <= 0.0f) return;
        const double zoom = scene.camera.zoom;
        const double world_width = scene.viewport_width / zoom;
        const double world_height = scene.viewport_height / zoom;
        const double left = scene.camera.position.x - world_width * 0.5;
        const double right = scene.camera.position.x + world_width * 0.5;
        const double bottom = scene.camera.position.y - world_height * 0.5;
        const double top = scene.camera.position.y + world_height * 0.5;
        const double spacing = scene.settings.grid.spacing;
        const auto clip = [&](double x, double y) {
            return std::array<float, 2>{
                static_cast<float>(2.0 * (x - left) / world_width - 1.0),
                static_cast<float>(2.0 * (top - y) / world_height - 1.0)};
        };
        const int first_x = static_cast<int>(std::floor(left / spacing)) - 1;
        const int last_x = static_cast<int>(std::ceil(right / spacing)) + 1;
        const int first_y = static_cast<int>(std::floor(bottom / spacing)) - 1;
        const int last_y = static_cast<int>(std::ceil(top / spacing)) + 1;
        constexpr float color[4] = {0.18f, 0.28f, 0.40f, 0.35f};
        for (int line = first_x; line <= last_x; ++line) {
            const double x = static_cast<double>(line) * spacing;
            const auto a = clip(x, bottom);
            const auto b = clip(x, top);
            grid_vertices_.push_back({{a[0], a[1]}, {color[0], color[1], color[2], color[3]}});
            grid_vertices_.push_back({{b[0], b[1]}, {color[0], color[1], color[2], color[3]}});
        }
        for (int line = first_y; line <= last_y; ++line) {
            const double y = static_cast<double>(line) * spacing;
            const auto a = clip(left, y);
            const auto b = clip(right, y);
            grid_vertices_.push_back({{a[0], a[1]}, {color[0], color[1], color[2], color[3]}});
            grid_vertices_.push_back({{b[0], b[1]}, {color[0], color[1], color[2], color[3]}});
        }
    }

    void buildTrajectoryVertices(const ::nbody::rendering::RenderScene& scene) {
        trajectory_vertices_.clear();
        if (!scene.settings.show_trajectories || scene.trajectories.points.size() < 2
            || scene.viewport_width <= 0.0f || scene.viewport_height <= 0.0f) return;
        const ::nbody::rendering::RenderViewport viewport{
            scene.viewport_width, scene.viewport_height};
        trajectory_vertices_.reserve(scene.trajectories.points.size() * 2);
        const auto toClip = [&](const Vec3& position) {
            return ::nbody::rendering::worldToClip(
                scene.camera, viewport, position.x, position.y);
        };
        constexpr float regular_color[4] = {0.28f, 0.62f, 0.88f, 0.55f};
        constexpr float selected_color[4] = {1.0f, 0.78f, 0.20f, 0.85f};
        std::size_t begin = 0;
        while (begin < scene.trajectories.points.size()) {
            const BodyId body = scene.trajectories.points[begin].body;
            std::size_t end = begin + 1;
            while (end < scene.trajectories.points.size()
                && scene.trajectories.points[end].body == body) {
                ++end;
            }
            const float* color = scene.selected_body && *scene.selected_body == body
                ? selected_color : regular_color;
            for (std::size_t index = begin + 1; index < end; ++index) {
                const auto previous = toClip(scene.trajectories.points[index - 1].position);
                const auto current = toClip(scene.trajectories.points[index].position);
                trajectory_vertices_.push_back({
                    {previous[0], previous[1]}, {color[0], color[1], color[2], color[3]}});
                trajectory_vertices_.push_back({
                    {current[0], current[1]}, {color[0], color[1], color[2], color[3]}});
            }
            begin = end;
        }
    }

    void createBodyRenderer() {
        upload_arena_.initialize(device, physical_device, 4 * 1024 * 1024);
        pipeline_manager_.initialize(device, NBODY_SHADER_DIR);
        VkVertexInputBindingDescription binding{0, sizeof(BodyInstance), VK_VERTEX_INPUT_RATE_INSTANCE};
        VkVertexInputAttributeDescription attributes[2]{
            {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(BodyInstance, center)},
            {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(BodyInstance, radius)}};
        VkVertexInputAttributeDescription color_attribute{
            2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(BodyInstance, color)};
        VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertex_input.vertexBindingDescriptionCount = 1;
        vertex_input.pVertexBindingDescriptions = &binding;
        vertex_input.vertexAttributeDescriptionCount = 3;
        vertex_input.pVertexAttributeDescriptions = attributes;
        VkVertexInputAttributeDescription body_attributes[3] = {
            attributes[0], attributes[1], color_attribute};
        vertex_input.pVertexAttributeDescriptions = body_attributes;
        VkVertexInputBindingDescription grid_binding{0, sizeof(BodyVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription grid_attributes[2]{
            {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(BodyVertex, position)},
            {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(BodyVertex, color)}};
        VkPipelineVertexInputStateCreateInfo grid_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        grid_input.vertexBindingDescriptionCount = 1;
        grid_input.pVertexBindingDescriptions = &grid_binding;
        grid_input.vertexAttributeDescriptionCount = 2;
        grid_input.pVertexAttributeDescriptions = grid_attributes;
        VulkanPipelineDescription body_description{
            "body-2d", VulkanShaderId::BodyVertex, VulkanShaderId::BodyFragment,
            VulkanShaderInterface::Body2D, render_pass, vertex_input,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, true, true, false};
        const VulkanPipelineHandle body = pipeline_manager_.createGraphicsPipeline(body_description);
        body_pipeline = body.pipeline;
        body_pipeline_layout = body.layout;

        VulkanPipelineDescription transparent_body_description{
            "transparent-body-2d", VulkanShaderId::BodyVertex, VulkanShaderId::BodyFragment,
            VulkanShaderInterface::Body2D, render_pass, vertex_input,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, true, false, true};
        const VulkanPipelineHandle transparent_body =
            pipeline_manager_.createGraphicsPipeline(transparent_body_description);
        transparent_body_pipeline = transparent_body.pipeline;
        transparent_body_pipeline_layout = transparent_body.layout;

        VulkanPipelineDescription grid_description{
            "grid-2d", VulkanShaderId::GridVertex, VulkanShaderId::GridFragment,
            VulkanShaderInterface::Grid2D, render_pass, grid_input,
            VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 0, false, false, true};
        grid_pipeline = pipeline_manager_.createGraphicsPipeline(grid_description).pipeline;

        VulkanPipelineDescription trajectory_description{
            "trajectories-2d", VulkanShaderId::GridVertex, VulkanShaderId::GridFragment,
            VulkanShaderInterface::Grid2D, render_pass, grid_input,
            VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 0, false, false, true};
        trajectory_pipeline = pipeline_manager_.createGraphicsPipeline(trajectory_description).pipeline;
    }

    void createFramebuffers() {
        framebuffers.resize(views.size());
        for (std::size_t index = 0; index < views.size(); ++index) {
            const VkImageView attachments[] = {views[index], depth_view};
            VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            info.renderPass = render_pass;
            info.attachmentCount = 2;
            info.pAttachments = attachments;
            info.width = extent.width;
            info.height = extent.height;
            info.layers = 1;
            check(vkCreateFramebuffer(device, &info, nullptr, &framebuffers[index]), "create framebuffer");
        }
    }

    void createCommands() {
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = queue_family;
        check(vkCreateCommandPool(device, &pool, nullptr, &command_pool), "create command pool");
        commands.resize(framebuffers.size());
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = command_pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = static_cast<std::uint32_t>(commands.size());
        check(vkAllocateCommandBuffers(device, &allocation, commands.data()), "allocate command buffers");
    }

    void destroyCommands() {
        if (command_pool) {
            vkDestroyCommandPool(device, command_pool, nullptr);
            command_pool = VK_NULL_HANDLE;
        }
        commands.clear();
    }

    void createSync() {
        VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        check(vkCreateSemaphore(device, &semaphore, nullptr, &image_available), "create semaphore");
        createRenderFinishedSemaphores();
        check(vkCreateFence(device, &fence_info, nullptr, &fence), "create fence");
    }

    void createRenderFinishedSemaphores() {
        render_finished.resize(views.size());
        VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (VkSemaphore& finished : render_finished) {
            check(vkCreateSemaphore(device, &semaphore, nullptr, &finished),
                  "create render-finished semaphore");
        }
    }

    void destroyRenderFinishedSemaphores() {
        for (VkSemaphore semaphore : render_finished) {
            if (semaphore) vkDestroySemaphore(device, semaphore, nullptr);
        }
        render_finished.clear();
    }

    void initializeImGui(GLFWwindow* window) {
        const VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000}};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool.maxSets = 1000;
        pool.poolSizeCount = static_cast<std::uint32_t>(std::size(pool_sizes));
        pool.pPoolSizes = pool_sizes;
        check(vkCreateDescriptorPool(device, &pool, nullptr, &descriptor_pool), "create descriptor pool");
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowRounding = 8.0f;
        style.ChildRounding = 6.0f;
        style.FrameRounding = 5.0f;
        style.PopupRounding = 7.0f;
        style.GrabRounding = 5.0f;
        style.WindowBorderSize = 1.0f;
        style.FrameBorderSize = 1.0f;
        style.Colors[ImGuiCol_WindowBg] = ImVec4(0.035f, 0.045f, 0.080f, 0.96f);
        style.Colors[ImGuiCol_TitleBg] = ImVec4(0.025f, 0.035f, 0.065f, 1.0f);
        style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.080f, 0.150f, 0.240f, 1.0f);
        style.Colors[ImGuiCol_Button] = ImVec4(0.100f, 0.240f, 0.360f, 1.0f);
        style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.140f, 0.380f, 0.520f, 1.0f);
        style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.100f, 0.300f, 0.440f, 1.0f);
        initializeImGuiBackend(window);
    }

    void initializeImGuiBackend(GLFWwindow* window) {
        ImGui_ImplGlfw_InitForVulkan(window, true);
        ImGui_ImplVulkan_InitInfo info{};
        info.ApiVersion = VK_API_VERSION_1_0;
        info.Instance = instance;
        info.PhysicalDevice = physical_device;
        info.Device = device;
        info.QueueFamily = queue_family;
        info.Queue = queue;
        info.DescriptorPool = descriptor_pool;
        info.MinImageCount = static_cast<std::uint32_t>(views.size());
        info.ImageCount = static_cast<std::uint32_t>(views.size());
        info.PipelineInfoMain.RenderPass = render_pass;
        info.PipelineInfoMain.Subpass = 0;
        info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        if (!ImGui_ImplVulkan_Init(&info)) throw std::runtime_error("initialize ImGui Vulkan backend");
    }
};



VulkanRenderer::VulkanRenderer() : impl_(new Impl) {}

VulkanRenderer::~VulkanRenderer() {
    delete impl_;
}

void VulkanRenderer::initialize(GLFWwindow* window) {
    impl_->initialize(window);
}

void VulkanRenderer::render(GLFWwindow* window, ImDrawData* draw_data,
                            const RenderScene& scene) {
    impl_->render(window, draw_data, scene);
}

const RenderDiagnostics& VulkanRenderer::diagnostics() const {
    return impl_->diagnostics();
}

}
