#include "frontend/vulkan_frontend.hpp"

#include "app/app_state.hpp"
#include "app/commands.hpp"
#include "rendering/render_scene.hpp"
#include "rendering/vulkan_renderer.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <span>
#include <vector>

namespace nbody::frontend {

namespace {

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

} // namespace

struct VulkanFrontendContext {
    VkInstance instance{};
    VkSurfaceKHR surface{};
    VkPhysicalDevice physical_device{};
    VkDevice device{};
    VkQueue queue{};
    std::uint32_t queue_family{};
    VkSwapchainKHR swapchain{};
    VkFormat format{};
    VkExtent2D extent{};
    std::vector<VkImageView> views;
    VkRenderPass render_pass{};
    std::vector<VkFramebuffer> framebuffers;
    VkCommandPool command_pool{};
    std::vector<VkCommandBuffer> commands;
    VkSemaphore image_available{};
    VkSemaphore render_finished{};
    VkFence fence{};
    VkDescriptorPool descriptor_pool{};
    VkPipeline body_pipeline{};
    VkPipelineLayout body_pipeline_layout{};
    VkBuffer body_vertex_buffer{};
    VkDeviceMemory body_vertex_memory{};
    void* body_vertex_mapping{};
    VkDeviceSize body_vertex_capacity{};
    VkPipeline grid_pipeline{};
    VkBuffer grid_vertex_buffer{};
    VkDeviceMemory grid_vertex_memory{};
    void* grid_vertex_mapping{};
    VkDeviceSize grid_vertex_capacity{};
    std::vector<BodyInstance> body_instances_;
    std::vector<BodyVertex> grid_vertices_;

    void initialize(GLFWwindow* window) {
        createInstance();
        check(glfwCreateWindowSurface(instance, window, nullptr, &surface), "window surface");
        selectDevice();
        createDevice();
        createSwapchain(window);
        createRenderPass();
        createBodyRenderer();
        createFramebuffers();
        createCommands();
        createSync();
        initializeImGui(window);
    }

    void render(GLFWwindow* window, ImDrawData* draw_data,
                const ::nbody::rendering::RenderScene& scene) {
        check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "wait fence");
        int framebuffer_width = 0;
        int framebuffer_height = 0;
        glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
        if (framebuffer_width <= 0 || framebuffer_height <= 0) return;
        if (static_cast<std::uint32_t>(framebuffer_width) != extent.width
            || static_cast<std::uint32_t>(framebuffer_height) != extent.height) {
            recreateSwapchain(window);
        }
        buildBodyInstances(scene);
        buildGridVertices(scene);
        uploadBodyInstances();
        uploadGridVertices();
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
        VkClearValue clear{};
        clear.color = {{0.02f, 0.03f, 0.06f, 1.0f}};
        VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        pass.renderPass = render_pass;
        pass.framebuffer = framebuffers[image];
        pass.renderArea.extent = extent;
        pass.clearValueCount = 1;
        pass.pClearValues = &clear;
        vkCmdBeginRenderPass(commands[image], &pass, VK_SUBPASS_CONTENTS_INLINE);
        if (!grid_vertices_.empty()) {
            VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent.width),
                                static_cast<float>(extent.height), 0.0f, 1.0f};
            VkRect2D scissor{{0, 0}, extent};
            vkCmdSetViewport(commands[image], 0, 1, &viewport);
            vkCmdSetScissor(commands[image], 0, 1, &scissor);
            vkCmdBindPipeline(commands[image], VK_PIPELINE_BIND_POINT_GRAPHICS, grid_pipeline);
            const VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(commands[image], 0, 1, &grid_vertex_buffer, &offset);
            vkCmdDraw(commands[image], static_cast<std::uint32_t>(grid_vertices_.size()), 1, 0, 0);
        }
        if (!body_instances_.empty()) {
            VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent.width),
                                static_cast<float>(extent.height), 0.0f, 1.0f};
            VkRect2D scissor{{0, 0}, extent};
            vkCmdSetViewport(commands[image], 0, 1, &viewport);
            vkCmdSetScissor(commands[image], 0, 1, &scissor);
            vkCmdBindPipeline(commands[image], VK_PIPELINE_BIND_POINT_GRAPHICS, body_pipeline);
            const VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(commands[image], 0, 1, &body_vertex_buffer, &offset);
            vkCmdDraw(commands[image], 6, static_cast<std::uint32_t>(body_instances_.size()), 0, 0);
        }
        ImGui_ImplVulkan_RenderDrawData(draw_data, commands[image]);
        vkCmdEndRenderPass(commands[image]);
        check(vkEndCommandBuffer(commands[image]), "end command buffer");

        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &image_available;
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commands[image];
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &render_finished;
        check(vkQueueSubmit(queue, 1, &submit, fence), "submit command buffer");

        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &render_finished;
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain;
        present.pImageIndices = &image;
        const VkResult presented = vkQueuePresentKHR(queue, &present);
        if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) {
            recreateSwapchain(window);
        } else {
            check(presented, "present");
        }
    }

    ~VulkanFrontendContext() {
        if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device);
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        destroyBodyRenderer();
        if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (image_available) vkDestroySemaphore(device, image_available, nullptr);
        if (render_finished) vkDestroySemaphore(device, render_finished, nullptr);
        if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
        for (VkFramebuffer framebuffer : framebuffers) vkDestroyFramebuffer(device, framebuffer, nullptr);
        if (render_pass) vkDestroyRenderPass(device, render_pass, nullptr);
        for (VkImageView view : views) vkDestroyImageView(device, view, nullptr);
        if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }

private:
    void createInstance() {
        std::uint32_t count = 0;
        const char** extensions = glfwGetRequiredInstanceExtensions(&count);
        if (!extensions) throw std::runtime_error("GLFW Vulkan extensions unavailable");
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.pApplicationName = "N-Body Simulator";
        application.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        application.pEngineName = "N-Body Greenfield";
        application.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        application.apiVersion = VK_API_VERSION_1_0;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        info.pApplicationInfo = &application;
        info.enabledExtensionCount = count;
        info.ppEnabledExtensionNames = extensions;
        check(vkCreateInstance(&info, nullptr, &instance), "create Vulkan instance");
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
        VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &reference;
        VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        info.attachmentCount = 1;
        info.pAttachments = &color;
        info.subpassCount = 1;
        info.pSubpasses = &subpass;
        check(vkCreateRenderPass(device, &info, nullptr, &render_pass), "create render pass");
    }

    void destroyBodyRenderer() {
        if (body_vertex_mapping) vkUnmapMemory(device, body_vertex_memory);
        if (body_vertex_buffer) vkDestroyBuffer(device, body_vertex_buffer, nullptr);
        if (body_vertex_memory) vkFreeMemory(device, body_vertex_memory, nullptr);
        if (grid_vertex_mapping) vkUnmapMemory(device, grid_vertex_memory);
        if (grid_vertex_buffer) vkDestroyBuffer(device, grid_vertex_buffer, nullptr);
        if (grid_vertex_memory) vkFreeMemory(device, grid_vertex_memory, nullptr);
        if (body_pipeline) vkDestroyPipeline(device, body_pipeline, nullptr);
        if (grid_pipeline) vkDestroyPipeline(device, grid_pipeline, nullptr);
        if (body_pipeline_layout) vkDestroyPipelineLayout(device, body_pipeline_layout, nullptr);
        body_vertex_mapping = nullptr;
        grid_vertex_mapping = nullptr;
        body_vertex_buffer = VK_NULL_HANDLE;
        grid_vertex_buffer = VK_NULL_HANDLE;
        body_vertex_memory = VK_NULL_HANDLE;
        grid_vertex_memory = VK_NULL_HANDLE;
        body_pipeline = VK_NULL_HANDLE;
        grid_pipeline = VK_NULL_HANDLE;
        body_pipeline_layout = VK_NULL_HANDLE;
        body_vertex_capacity = 0;
        grid_vertex_capacity = 0;
    }

    void destroySwapchain() {
        for (VkFramebuffer framebuffer : framebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        framebuffers.clear();
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

    void recreateSwapchain(GLFWwindow* window) {
        vkDeviceWaitIdle(device);
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        destroyBodyRenderer();
        destroySwapchain();
        destroyCommands();
        createSwapchain(window);
        createRenderPass();
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

    VkShaderModule loadShader(const char* name) {
        std::ifstream input(std::string(NBODY_SHADER_DIR) + "/" + name,
                            std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error(std::string("could not open shader ") + name);
        const std::streamsize size = input.tellg();
        if (size <= 0 || size % 4 != 0) throw std::runtime_error("invalid SPIR-V shader size");
        std::vector<char> bytes(static_cast<std::size_t>(size));
        input.seekg(0);
        input.read(bytes.data(), size);
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = bytes.size();
        info.pCode = reinterpret_cast<const std::uint32_t*>(bytes.data());
        VkShaderModule module{};
        check(vkCreateShaderModule(device, &info, nullptr, &module), "create shader module");
        return module;
    }

    void createBodyVertexBuffer(VkDeviceSize capacity) {
        VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = capacity;
        buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device, &buffer_info, nullptr, &body_vertex_buffer), "create body vertex buffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, body_vertex_buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device, &allocation, nullptr, &body_vertex_memory), "allocate body vertex memory");
        check(vkBindBufferMemory(device, body_vertex_buffer, body_vertex_memory, 0), "bind body vertex memory");
        check(vkMapMemory(device, body_vertex_memory, 0, capacity, 0, &body_vertex_mapping),
              "map body vertex memory");
        body_vertex_capacity = capacity;
    }

    void createGridVertexBuffer(VkDeviceSize capacity) {
        VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = capacity;
        buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device, &buffer_info, nullptr, &grid_vertex_buffer), "create grid vertex buffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, grid_vertex_buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device, &allocation, nullptr, &grid_vertex_memory), "allocate grid vertex memory");
        check(vkBindBufferMemory(device, grid_vertex_buffer, grid_vertex_memory, 0), "bind grid vertex memory");
        check(vkMapMemory(device, grid_vertex_memory, 0, capacity, 0, &grid_vertex_mapping),
              "map grid vertex memory");
        grid_vertex_capacity = capacity;
    }

    void uploadBodyInstances() {
        const VkDeviceSize required = static_cast<VkDeviceSize>(body_instances_.size() * sizeof(BodyInstance));
        if (required > body_vertex_capacity) {
            vkUnmapMemory(device, body_vertex_memory);
            vkDestroyBuffer(device, body_vertex_buffer, nullptr);
            vkFreeMemory(device, body_vertex_memory, nullptr);
            body_vertex_mapping = nullptr;
            createBodyVertexBuffer(std::max<VkDeviceSize>(required, body_vertex_capacity * 2));
        }
        if (required != 0) std::memcpy(body_vertex_mapping, body_instances_.data(), required);
    }

    void uploadGridVertices() {
        const VkDeviceSize required = static_cast<VkDeviceSize>(grid_vertices_.size() * sizeof(BodyVertex));
        if (required > grid_vertex_capacity) {
            vkUnmapMemory(device, grid_vertex_memory);
            vkDestroyBuffer(device, grid_vertex_buffer, nullptr);
            vkFreeMemory(device, grid_vertex_memory, nullptr);
            grid_vertex_mapping = nullptr;
            createGridVertexBuffer(std::max<VkDeviceSize>(required, grid_vertex_capacity * 2));
        }
        if (required != 0) std::memcpy(grid_vertex_mapping, grid_vertices_.data(), required);
    }

    void buildBodyInstances(const ::nbody::rendering::RenderScene& scene) {
        body_instances_.clear();
        if (!scene.hasCpuBodies()) return;
        const float width = std::max(1.0f, scene.viewport_width);
        const float height = std::max(1.0f, scene.viewport_height);
        const float zoom = static_cast<float>(scene.camera.zoom);
        const float center_x = width * 0.5f;
        const float center_y = height * 0.5f;
        body_instances_.reserve(scene.cpuBodies().size());
        for (const RenderBody& body : scene.cpuBodies()) {
            const float screen_x = center_x
                + (static_cast<float>(body.position.x) - static_cast<float>(scene.camera.position.x)) * zoom;
            const float screen_y = center_y
                - (static_cast<float>(body.position.y) - static_cast<float>(scene.camera.position.y)) * zoom;
            const float red = body.kind == BodyKind::Star ? 1.0f : (body.is_static ? 0.75f : 0.35f);
            const float green = body.kind == BodyKind::Star ? 0.75f : (body.is_static ? 0.50f : 0.85f);
            const float blue = body.kind == BodyKind::Star ? 0.25f : 1.0f;
            body_instances_.push_back({
                {2.0f * screen_x / width - 1.0f, 2.0f * screen_y / height - 1.0f},
                {2.0f * std::max(0.75f, static_cast<float>(body.radius) * zoom) / width,
                 2.0f * std::max(0.75f, static_cast<float>(body.radius) * zoom) / height},
                {red, green, blue, body.kind == BodyKind::Star ? 1.0f : 0.85f}});
        }
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

    void createBodyRenderer() {
        createBodyVertexBuffer(4 * 1024 * 1024);
        createGridVertexBuffer(1024 * 1024);
        VkShaderModule vertex_shader = loadShader("body.vert.spv");
        VkShaderModule fragment_shader = loadShader("body.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertex_shader;
        stages[0].pName = "main";
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragment_shader;
        stages[1].pName = "main";
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
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterization.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState blend{};
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend.blendEnable = VK_TRUE;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.colorBlendOp = VK_BLEND_OP_ADD;
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.alphaBlendOp = VK_BLEND_OP_ADD;
        VkPipelineColorBlendStateCreateInfo color_blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        color_blend.attachmentCount = 1;
        color_blend.pAttachments = &blend;
        VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = 2;
        dynamic.pDynamicStates = dynamic_states;
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        check(vkCreatePipelineLayout(device, &layout, nullptr, &body_pipeline_layout),
              "create body pipeline layout");
        VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipeline.stageCount = 2;
        pipeline.pStages = stages;
        pipeline.pVertexInputState = &vertex_input;
        pipeline.pInputAssemblyState = &assembly;
        pipeline.pViewportState = &viewport;
        pipeline.pRasterizationState = &rasterization;
        pipeline.pMultisampleState = &multisample;
        pipeline.pColorBlendState = &color_blend;
        pipeline.pDynamicState = &dynamic;
        pipeline.layout = body_pipeline_layout;
        pipeline.renderPass = render_pass;
        pipeline.subpass = 0;
        check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &body_pipeline),
              "create body graphics pipeline");
        vkDestroyShaderModule(device, fragment_shader, nullptr);
        vkDestroyShaderModule(device, vertex_shader, nullptr);

        VkShaderModule grid_vertex_shader = loadShader("grid.vert.spv");
        VkShaderModule grid_fragment_shader = loadShader("grid.frag.spv");
        VkPipelineShaderStageCreateInfo grid_stages[2]{};
        grid_stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        grid_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        grid_stages[0].module = grid_vertex_shader;
        grid_stages[0].pName = "main";
        grid_stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        grid_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        grid_stages[1].module = grid_fragment_shader;
        grid_stages[1].pName = "main";
        VkVertexInputBindingDescription grid_binding{0, sizeof(BodyVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription grid_attributes[2]{
            {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(BodyVertex, position)},
            {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(BodyVertex, color)}};
        VkPipelineVertexInputStateCreateInfo grid_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        grid_input.vertexBindingDescriptionCount = 1;
        grid_input.pVertexBindingDescriptions = &grid_binding;
        grid_input.vertexAttributeDescriptionCount = 2;
        grid_input.pVertexAttributeDescriptions = grid_attributes;
        VkPipelineInputAssemblyStateCreateInfo grid_assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        grid_assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        VkGraphicsPipelineCreateInfo grid_pipeline_info = pipeline;
        grid_pipeline_info.stageCount = 2;
        grid_pipeline_info.pStages = grid_stages;
        grid_pipeline_info.pVertexInputState = &grid_input;
        grid_pipeline_info.pInputAssemblyState = &grid_assembly;
        check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &grid_pipeline_info,
                                         nullptr, &grid_pipeline), "create grid graphics pipeline");
        vkDestroyShaderModule(device, grid_fragment_shader, nullptr);
        vkDestroyShaderModule(device, grid_vertex_shader, nullptr);
    }

    void createFramebuffers() {
        framebuffers.resize(views.size());
        for (std::size_t index = 0; index < views.size(); ++index) {
            VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            info.renderPass = render_pass;
            info.attachmentCount = 1;
            info.pAttachments = &views[index];
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
        check(vkCreateSemaphore(device, &semaphore, nullptr, &render_finished), "create semaphore");
        check(vkCreateFence(device, &fence_info, nullptr, &fence), "create fence");
    }

    void initializeImGui(GLFWwindow* window) {
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool.maxSets = 1000;
        pool.poolSizeCount = 1;
        pool.pPoolSizes = &pool_size;
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

namespace {

class VulkanFrontendState {
public:
    VulkanFrontendState() : commands_(application_) {}

    void draw(GLFWwindow* window, rendering::VulkanRenderer& renderer) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - last_frame_).count();
        last_frame_ = now;
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        if (application_.mode == nbody::app::AppMode::MainMenu) {
            drawTitle();
        } else if (application_.mode == nbody::app::AppMode::Simulation) {
            drawSimulation(window, std::min(elapsed, 0.1));
        }
        ImGui::Render();
        renderer.render(window, ImGui::GetDrawData(), buildRenderScene());
    }

    void onScroll(double offset) { pending_scroll_ += offset; }

    static void scrollCallback(GLFWwindow* window, double x_offset, double y_offset) {
        ImGui_ImplGlfw_ScrollCallback(window, x_offset, y_offset);
        if (auto* state = static_cast<VulkanFrontendState*>(glfwGetWindowUserPointer(window))) {
            state->onScroll(y_offset);
        }
    }

private:
    nbody::app::ApplicationState application_;
    nbody::app::CommandDispatcher commands_;
    std::optional<FramePublication> publication_;
    std::chrono::steady_clock::time_point last_frame_{std::chrono::steady_clock::now()};
    double accumulator_{};
    double time_scale_value_{1.0};
    double simulation_timestep_{0.001};
    double simulation_steps_per_second_{};
    int time_scale_unit_{};
    int generator_{};
    int generated_count_{8};
    int spiral_arms_{2};
    int random_seed_{42};
    float body_x_{};
    float body_y_{};
    float body_velocity_x_{};
    float body_velocity_y_{};
    float body_mass_{1.0f};
    float body_radius_{0.25f};
    float minimum_mass_{0.5f};
    float maximum_mass_{2.0f};
    float minimum_radius_{0.1f};
    float maximum_radius_{0.5f};
    float distribution_size_{5.0f};
    float gaussian_sigma_{2.0f};
    float inner_radius_{2.0f};
    float outer_radius_{8.0f};
    float spiral_turns_{2.0f};
    float tangential_velocity_{1.0f};
    bool add_body_{};
    bool static_body_{};
    bool scaled_solar_system_{};
    float camera_x_{};
    float camera_y_{};
    float view_scale_{20.0f};
    double pending_scroll_{};
    double last_cursor_x_{};
    double last_cursor_y_{};
    bool have_cursor_position_{};
    ImVec2 viewport_min_{};
    ImVec2 viewport_max_{};
    bool reset_confirmation_{};
    std::optional<Dimension> dimension_confirmation_;

    WorldState& world() { return application_.session.state().world; }
    const WorldState& world() const { return application_.session.state().world; }
    SimulationParameters& parameters() { return application_.session.state().parameters; }
    const SimulationParameters& parameters() const { return application_.session.state().parameters; }

    bool simulationRunning() const {
        return application_.session.mode() == nbody::app::SimulationMode::Running;
    }

    ::nbody::rendering::RenderScene buildRenderScene() const {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        const float render_width = std::max(1.0f, viewport->WorkSize.x);
        const float render_height = std::max(1.0f, viewport->WorkSize.y);
        ::nbody::rendering::RenderCamera camera;
        camera.position = {camera_x_, camera_y_, 0.0};
        camera.zoom = view_scale_;
        ::nbody::rendering::RenderSceneSettings settings;
        settings.grid.visible = application_.presentation.grid.visible;
        settings.grid.spacing = static_cast<double>(gridStep(std::min(render_width, render_height)));
        return ::nbody::rendering::makeRenderScene(
            publication_.value_or(FramePublication{}), camera, settings, {},
            render_width, render_height);
    }

    void drawTitle() {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->GetCenter(), ImGuiCond_Always, {0.5f, 0.5f});
        ImGui::SetNextWindowSize({440.0f, 240.0f}, ImGuiCond_Always);
        ImGui::Begin("N-Body Simulator", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove);
        ImGui::Dummy({1.0f, 20.0f});
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - ImGui::CalcTextSize("N-BODY SIMULATOR").x) * 0.5f);
        ImGui::TextUnformatted("N-BODY SIMULATOR");
        ImGui::Dummy({1.0f, 60.0f});
        if (ImGui::Button("Enter simulation", {-1.0f, 36.0f}) || ImGui::IsKeyPressed(ImGuiKey_Enter)) {
            commands_.dispatch(nbody::app::StartSimulation{});
        }
        ImGui::TextDisabled("Press Enter to begin");
        ImGui::End();
    }

    void drawSimulation(GLFWwindow* window, double elapsed) {
        updatePanelAnimation(elapsed);
        updateViewportBounds();
        drawTopBar();
        drawTopBarIndicator();
        drawSidePanels();
        drawBottomBar();
        drawBottomBarIndicator();
        updateCameraInput(window);

        simulation_timestep_ = std::clamp(simulation_timestep_, 1.0e-5, 86400.0);
        parameters().timestep = simulation_timestep_;
        parameters().gravitational_constant = scaled_solar_system_ ? 9.33076e-11 : 0.1;
        // Solar-system distances use a physical kilometre scale. Keep the
        // regularization far below satellite orbital distances; it is not a
        // substitute for collision handling or timestep selection.
        constexpr double solar_system_softening_km = 100.0;
        parameters().softening_length = scaled_solar_system_
            ? solar_system_softening_km / worldKilometersPerUnit()
            : 0.05;
        parameters().collision.model = CollisionModel::Transparent;
        const double requested_rate = timeScaleSecondsPerRealSecond();
        if (simulationRunning()) {
            accumulator_ += elapsed * requested_rate;
        } else {
            accumulator_ = 0.0;
        }
        std::size_t steps_this_frame = 0;
        while (simulationRunning() && accumulator_ >= parameters().timestep && steps_this_frame < 256) {
            application_.session.step();
            accumulator_ -= parameters().timestep;
            ++steps_this_frame;
        }
        publication_ = application_.session.publishFrame();
        updateCameraLock();
        if (elapsed > 0.000001) {
            const double measured = static_cast<double>(steps_this_frame) / elapsed;
            simulation_steps_per_second_ = simulation_steps_per_second_ == 0.0
                ? measured
                : simulation_steps_per_second_ * 0.9 + measured * 0.1;
        }
        if (add_body_) drawAddBody();
        drawWorld(window);
    }

    void updatePanelAnimation(double elapsed) {
        for (auto& panel : application_.presentation.panels.panels) {
            const float target = panel.expanded ? 1.0f : 0.0f;
            const float step = static_cast<float>(std::min(elapsed, 0.5) / 0.5);
            if (panel.animation_progress < target) {
                panel.animation_progress = std::min(target, panel.animation_progress + step);
            } else {
                panel.animation_progress = std::max(target, panel.animation_progress - step);
            }
        }
    }

    nbody::app::FoldablePanelState& panel(nbody::app::PanelEdge edge) {
        for (auto& candidate : application_.presentation.panels.panels) {
            if (candidate.edge == edge) return candidate;
        }
        return application_.presentation.panels.panels[0];
    }

    void drawTopBar() {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        constexpr float height = 42.0f;
        const float progress = panel(nbody::app::PanelEdge::Top).animation_progress;
        ImGui::SetNextWindowPos({viewport->WorkPos.x, viewport->WorkPos.y - height * (1.0f - progress)},
                                ImGuiCond_Always);
        ImGui::SetNextWindowSize({viewport->WorkSize.x, height}, ImGuiCond_Always);
        ImGui::Begin("Top bar", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        const auto mode = application_.session.mode();
        if (mode == nbody::app::SimulationMode::StartupEdit) {
            if (ImGui::Button("Start simulation")) {
                commands_.dispatch(nbody::app::FinishStartupEdit{});
                commands_.dispatch(nbody::app::ResumeSimulation{});
            }
            ImGui::SameLine();
            const int current_dimension = parameters().dimension == Dimension::Three ? 1 : 0;
            int selected_dimension = current_dimension;
            const char* dimensions[] = {"2D", "3D"};
            ImGui::SetNextItemWidth(70.0f);
            if (ImGui::Combo("Dimension", &selected_dimension, dimensions, 2)
                && selected_dimension != current_dimension) {
                dimension_confirmation_ = selected_dimension == 1 ? Dimension::Three : Dimension::Two;
                reset_confirmation_ = true;
            }
        } else if (ImGui::Button(simulationRunning() ? "Pause" : "Start")) {
            if (simulationRunning()) commands_.dispatch(nbody::app::PauseSimulation{});
            else commands_.dispatch(nbody::app::ResumeSimulation{});
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100.0f);
        ImGui::InputDouble("##time-scale", &time_scale_value_, 0.1, 1.0, "%.3f");
        ImGui::SameLine();
        const char* rate_units[] = {"seconds / real second", "days / real second",
                                    "months / real second", "years / real second"};
        ImGui::SetNextItemWidth(155.0f);
        ImGui::Combo("##time-scale-unit", &time_scale_unit_, rate_units, 4);
        ImGui::SameLine();
        ImGui::Text("sim time %.3f", displayedSimulationTime());
        ImGui::SameLine();
        int display_unit = static_cast<int>(application_.presentation.units.time);
        const char* display_units[] = {"seconds", "days", "months", "years"};
        ImGui::SetNextItemWidth(85.0f);
        if (ImGui::Combo("##sim-time-unit", &display_unit, display_units, 4)) {
            commands_.dispatch(nbody::app::SetTimeDisplayUnit{
                static_cast<nbody::app::TimeDisplayUnit>(display_unit)});
        }
        ImGui::SameLine();
        ImGui::Text("| bodies %zu", world().bodyCount());
        ImGui::SameLine();
        if (ImGui::Button(application_.presentation.grid.visible ? "Hide grid" : "Show grid")) {
            commands_.dispatch(nbody::app::SetGridVisible{!application_.presentation.grid.visible});
        }
        const bool edit_mode = mode == nbody::app::SimulationMode::StartupEdit
            || mode == nbody::app::SimulationMode::Edit;
        if (edit_mode) {
            ImGui::SameLine();
            if (ImGui::Button("Add body")) add_body_ = true;
            ImGui::SameLine();
            if (ImGui::Button("Solar system")) createDebugSolarSystem();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            dimension_confirmation_.reset();
            reset_confirmation_ = true;
        }
        ImGui::SameLine(ImGui::GetWindowWidth() - 28.0f);
        if (ImGui::SmallButton("v")) panel(nbody::app::PanelEdge::Top).expanded = false;
        ImGui::End();

        if (reset_confirmation_) {
            ImGui::OpenPopup("Confirm reset");
            reset_confirmation_ = false;
        }
        if (ImGui::BeginPopupModal("Confirm reset", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextUnformatted(dimension_confirmation_
                ? "Switching dimension will discard the entire current setup."
                : "This will discard the current setup and return to Initial Edit Mode.");
            if (ImGui::Button("Reset", {100.0f, 0.0f})) {
                if (dimension_confirmation_) {
                    commands_.dispatch(nbody::app::SwitchDimension{*dimension_confirmation_});
                    dimension_confirmation_.reset();
                } else {
                    commands_.dispatch(nbody::app::ResetToInitialEdit{});
                }
                resetTransientFrontendState();
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", {100.0f, 0.0f})) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
    }

    void resetTransientFrontendState() {
        accumulator_ = 0.0;
        scaled_solar_system_ = false;
        simulation_timestep_ = 0.001;
        time_scale_value_ = 1.0;
        time_scale_unit_ = 0;
        camera_x_ = 0.0f;
        camera_y_ = 0.0f;
        view_scale_ = 20.0f;
        add_body_ = false;
        pending_scroll_ = 0.0;
        publication_.reset();
    }

    void drawSidePanels() {
        drawPanelIndicator(nbody::app::PanelEdge::Left);
        drawPanelIndicator(nbody::app::PanelEdge::Right);
        drawLeftPanel();
        drawRightPanel();
    }

    void drawBottomBar() {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        constexpr float height = 48.0f;
        const float progress = panel(nbody::app::PanelEdge::Bottom).animation_progress;
        const float render_span = std::min(viewport->WorkSize.x, viewport->WorkSize.y);
        ImGui::SetNextWindowPos({viewport->WorkPos.x,
                                 viewport->WorkPos.y + viewport->WorkSize.y - height * progress},
                                ImGuiCond_Always);
        ImGui::SetNextWindowSize({viewport->WorkSize.x, height}, ImGuiCond_Always);
        ImGui::Begin("Bottom bar", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        ImGui::SetNextItemWidth(115.0f);
        ImGui::InputDouble("##timestep", &simulation_timestep_, 0.001, 0.01, "%.6g");
        ImGui::SameLine();
        ImGui::Text("step s  |  sim %.1f frames/s  |  %s", simulation_steps_per_second_,
                    simulationRunning() ? "running" : "paused");
        ImGui::SameLine();
        ImGui::Text("|  zoom %.3g  |  grid %.3g %s", view_scale_,
                    gridDistance(render_span).value,
                    gridDistance(render_span).unit);
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset view")) {
            camera_x_ = 0.0f;
            camera_y_ = 0.0f;
            view_scale_ = 20.0f;
        }
        ImGui::SameLine(ImGui::GetWindowWidth() - 28.0f);
        if (ImGui::SmallButton("^")) panel(nbody::app::PanelEdge::Bottom).expanded = false;
        ImGui::End();
    }

    void drawTopBarIndicator() {
        auto& state = panel(nbody::app::PanelEdge::Top);
        if (state.expanded) return;
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos({viewport->WorkPos.x + viewport->WorkSize.x * 0.5f - 18.0f,
                                 viewport->WorkPos.y + 3.0f}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({36.0f, 24.0f}, ImGuiCond_Always);
        ImGui::Begin("Top bar indicator", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        if (ImGui::IsWindowHovered()) state.hover_time += ImGui::GetIO().DeltaTime;
        else state.hover_time = 0.0f;
        if (state.hover_time >= application_.presentation.panels.reveal_dwell_seconds
            || ImGui::Button("v")) {
            state.expanded = true;
            state.hover_time = 0.0f;
        }
        ImGui::End();
    }

    void drawBottomBarIndicator() {
        auto& state = panel(nbody::app::PanelEdge::Bottom);
        if (state.expanded) return;
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos({viewport->WorkPos.x + viewport->WorkSize.x * 0.5f - 18.0f,
                                 viewport->WorkPos.y + viewport->WorkSize.y - 27.0f}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({36.0f, 24.0f}, ImGuiCond_Always);
        ImGui::Begin("Bottom bar indicator", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        if (ImGui::IsWindowHovered()) state.hover_time += ImGui::GetIO().DeltaTime;
        else state.hover_time = 0.0f;
        if (state.hover_time >= application_.presentation.panels.reveal_dwell_seconds
            || ImGui::Button("^")) {
            state.expanded = true;
            state.hover_time = 0.0f;
        }
        ImGui::End();
    }

    void drawPanelIndicator(nbody::app::PanelEdge edge) {
        auto& state = panel(edge);
        if (state.expanded) return;
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        const bool left = edge == nbody::app::PanelEdge::Left;
        const ImVec2 position = left
            ? ImVec2{viewport->WorkPos.x + 4.0f, viewport->WorkPos.y + viewport->WorkSize.y * 0.5f - 20.0f}
            : ImVec2{viewport->WorkPos.x + viewport->WorkSize.x - 34.0f,
                     viewport->WorkPos.y + viewport->WorkSize.y * 0.5f - 20.0f};
        ImGui::SetNextWindowPos(position, ImGuiCond_Always);
        ImGui::SetNextWindowSize({30.0f, 40.0f}, ImGuiCond_Always);
        ImGui::Begin(left ? "Left panel indicator" : "Right panel indicator", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        if (ImGui::IsWindowHovered()) state.hover_time += ImGui::GetIO().DeltaTime;
        else state.hover_time = 0.0f;
        if (state.hover_time >= application_.presentation.panels.reveal_dwell_seconds
            || ImGui::Button(left ? ">" : "<")) {
            state.expanded = true;
            state.hover_time = 0.0f;
        }
        ImGui::End();
    }

    void drawLeftPanel() {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        constexpr float width = 280.0f;
        const float progress = panel(nbody::app::PanelEdge::Left).animation_progress;
        const float top = 42.0f * panel(nbody::app::PanelEdge::Top).animation_progress;
        const float bottom = 48.0f * panel(nbody::app::PanelEdge::Bottom).animation_progress;
        ImGui::SetNextWindowPos({viewport->WorkPos.x - width * (1.0f - progress),
                                 viewport->WorkPos.y + top}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({width, std::max(1.0f, viewport->WorkSize.y - top - bottom)}, ImGuiCond_Always);
        ImGui::Begin("Bodies", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
            | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings);
        if (ImGui::Button("Fold")) panel(nbody::app::PanelEdge::Left).expanded = false;
        ImGui::SameLine();
        if (application_.presentation.focus.body) {
            ImGui::Text("Locked to #%llu", static_cast<unsigned long long>(application_.presentation.focus.body->value));
        }
        else ImGui::TextDisabled("Select a body to inspect");
        ImGui::Separator();
        if (publication_ && publication_->published()) {
            for (const RenderBody& body : publication_->cpu_snapshot->bodies) {
                ImGui::PushID(static_cast<int>(body.id.value));
                const bool selected = application_.presentation.selection.primary == body.id;
                if (ImGui::Selectable("##body", selected, 0, {245.0f, 24.0f})) {
                    commands_.dispatch(nbody::app::SelectBody{body.id});
                }
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    commands_.dispatch(nbody::app::FocusBody{body.id});
                }
                ImGui::SameLine();
                ImGui::Text("#%llu  %.3g", static_cast<unsigned long long>(body.id.value), body.mass);
                ImGui::PopID();
            }
        }
        ImGui::End();
    }

    void drawRightPanel() {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        constexpr float width = 300.0f;
        const float progress = panel(nbody::app::PanelEdge::Right).animation_progress;
        const float top = 42.0f * panel(nbody::app::PanelEdge::Top).animation_progress;
        const float bottom = 48.0f * panel(nbody::app::PanelEdge::Bottom).animation_progress;
        ImGui::SetNextWindowPos({viewport->WorkPos.x + viewport->WorkSize.x - width * progress,
                                 viewport->WorkPos.y + top}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({width, std::max(1.0f, viewport->WorkSize.y - top - bottom)}, ImGuiCond_Always);
        ImGui::Begin("Inspector", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
            | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings);
        if (ImGui::Button("Fold")) panel(nbody::app::PanelEdge::Right).expanded = false;
        ImGui::Separator();
        const auto selected = application_.presentation.selection.primary;
        if (selected && world().bodyCount() > 0) {
            for (std::size_t index = 0; index < world().bodyCount(); ++index) {
                if (world().body(index).id == *selected) {
                    const ConstBodyView body = world().body(index);
                    ImGui::Text("Body #%llu", static_cast<unsigned long long>(body.id.value));
                    ImGui::Text("Mass %.6g", body.mass);
                    ImGui::Text("Radius %.6g", body.radius);
                    ImGui::Text("Position %.4g, %.4g, %.4g", body.position.x, body.position.y, body.position.z);
                    break;
                }
            }
        } else {
            ImGui::TextDisabled("No body selected");
        }
        ImGui::End();
    }

    void updateViewportBounds() {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        const float left = panel(nbody::app::PanelEdge::Left).animation_progress * 280.0f;
        const float right = panel(nbody::app::PanelEdge::Right).animation_progress * 300.0f;
        const float top = 42.0f * panel(nbody::app::PanelEdge::Top).animation_progress;
        const float bottom = 48.0f * panel(nbody::app::PanelEdge::Bottom).animation_progress;
        viewport_min_ = {viewport->WorkPos.x + left, viewport->WorkPos.y + top};
        viewport_max_ = {viewport->WorkPos.x + viewport->WorkSize.x - right,
                         viewport->WorkPos.y + viewport->WorkSize.y - bottom};
    }

    void updateCameraLock() {
        const auto focused_body = application_.presentation.focus.body;
        if (!focused_body || !publication_ || !publication_->published()) return;
        for (const RenderBody& body : publication_->cpu_snapshot->bodies) {
            if (body.id == *focused_body) {
                camera_x_ = static_cast<float>(body.position.x);
                camera_y_ = static_cast<float>(body.position.y);
                return;
            }
        }
        commands_.dispatch(nbody::app::FocusBody{std::nullopt});
    }

    double timeScaleSecondsPerRealSecond() const {
        const double value = std::max(0.0, time_scale_value_);
        return value * secondsPerDisplayTimeUnit(time_scale_unit_);
    }

    static double secondsPerDisplayTimeUnit(int unit) {
        switch (unit) {
        case 1: return 86400.0;
        case 2: return 30.0 * 86400.0;
        case 3: return 365.25 * 86400.0;
        default: return 1.0;
        }
    }

    double displayedSimulationTime() const {
        return world().time() / secondsPerDisplayTimeUnit(
            static_cast<int>(application_.presentation.units.time));
    }

    void drawAddBody() {
        ImGui::OpenPopup("Add body");
        if (!ImGui::BeginPopupModal("Add body", &add_body_, ImGuiWindowFlags_AlwaysAutoResize)) return;
        const char* generators[] = {"Single body", "Uniform", "Gaussian", "Spiral", "Circle"};
        ImGui::Combo("Pattern", &generator_, generators, 5);
        if (generator_ == 0) {
            ImGui::InputFloat("X", &body_x_);
            ImGui::InputFloat("Y", &body_y_);
            ImGui::InputFloat("Velocity X", &body_velocity_x_);
            ImGui::InputFloat("Velocity Y", &body_velocity_y_);
            ImGui::InputFloat("Mass", &body_mass_);
            ImGui::InputFloat("Radius", &body_radius_);
        } else {
            ImGui::InputInt("Count", &generated_count_);
            ImGui::InputFloat("Center X", &body_x_);
            ImGui::InputFloat("Center Y", &body_y_);
            ImGui::InputFloat("Min mass", &minimum_mass_);
            ImGui::InputFloat("Max mass", &maximum_mass_);
            ImGui::InputFloat("Min radius", &minimum_radius_);
            ImGui::InputFloat("Max radius", &maximum_radius_);
            ImGui::InputFloat("Velocity X", &body_velocity_x_);
            ImGui::InputFloat("Velocity Y", &body_velocity_y_);
            if (generator_ == 1) ImGui::InputFloat("Uniform half-size", &distribution_size_);
            if (generator_ == 2) ImGui::InputFloat("Gaussian sigma", &gaussian_sigma_);
            if (generator_ == 3) {
                ImGui::InputInt("Arms", &spiral_arms_);
                ImGui::InputFloat("Inner radius", &inner_radius_);
                ImGui::InputFloat("Outer radius", &outer_radius_);
                ImGui::InputFloat("Turns", &spiral_turns_);
                ImGui::InputFloat("Tangential velocity", &tangential_velocity_);
            }
            if (generator_ == 4) {
                ImGui::InputFloat("Inner radius", &inner_radius_);
                ImGui::InputFloat("Outer radius", &outer_radius_);
                ImGui::InputFloat("Tangential velocity", &tangential_velocity_);
            }
            ImGui::InputInt("Random seed", &random_seed_);
        }
        ImGui::Checkbox("Static body/bodies", &static_body_);
        if (ImGui::Button("Create")) {
            generateBodies();
            accumulator_ = 0.0;
            add_body_ = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            add_body_ = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    void generateBodies() {
        const int count = generator_ == 0 ? 1 : std::max(1, generated_count_);
        std::mt19937 generator(static_cast<std::mt19937::result_type>(random_seed_));
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        std::uniform_real_distribution<float> mass(std::min(minimum_mass_, maximum_mass_),
                                                   std::max(minimum_mass_, maximum_mass_));
        std::uniform_real_distribution<float> radius(std::min(minimum_radius_, maximum_radius_),
                                                     std::max(minimum_radius_, maximum_radius_));
        constexpr float pi = 3.14159265358979323846f;
        for (int index = 0; index < count; ++index) {
            float x = body_x_;
            float y = body_y_;
            float angle = 0.0f;
            if (generator_ == 1) {
                x += (unit(generator) * 2.0f - 1.0f) * distribution_size_;
                y += (unit(generator) * 2.0f - 1.0f) * distribution_size_;
            } else if (generator_ == 2) {
                std::normal_distribution<float> gaussian(0.0f, std::max(0.001f, gaussian_sigma_));
                x += gaussian(generator);
                y += gaussian(generator);
            } else if (generator_ == 3 || generator_ == 4) {
                const float fraction = count == 1 ? 0.0f : static_cast<float>(index) / (count - 1);
                angle = generator_ == 3
                    ? 2.0f * pi * static_cast<float>(index % std::max(1, spiral_arms_))
                        / std::max(1, spiral_arms_) + 2.0f * pi * spiral_turns_ * fraction
                    : 2.0f * pi * fraction;
                const float radial = inner_radius_ + (outer_radius_ - inner_radius_) * fraction;
                x += radial * std::cos(angle);
                y += radial * std::sin(angle);
            }
            BodyState body;
            body.position = {x, y, 0.0};
            body.velocity = {body_velocity_x_, body_velocity_y_, 0.0};
            if (generator_ == 3 || generator_ == 4) {
                // Build the tangential component from the actual radial
                // offset so it remains perpendicular to the spiral centre,
                // independent of how the point was generated.
                const float radial_x = x - body_x_;
                const float radial_y = y - body_y_;
                const float radial_length = std::hypot(radial_x, radial_y);
                if (radial_length > 0.0f) {
                    body.velocity.x += -radial_y / radial_length * tangential_velocity_;
                    body.velocity.y += radial_x / radial_length * tangential_velocity_;
                }
            }
            body.mass = generator_ == 0 ? std::max(0.001f, body_mass_) : std::max(0.001f, mass(generator));
            body.radius = generator_ == 0 ? std::max(0.001f, body_radius_) : std::max(0.001f, radius(generator));
            body.is_static = static_body_;
            commands_.dispatch(nbody::app::CreateBody{body});
        }
    }

    void updateCameraInput(GLFWwindow* window) {
        const ImVec2 work_min = viewport_min_;
        const ImVec2 work_max = viewport_max_;
        const ImGuiViewport* full_viewport = ImGui::GetMainViewport();
        const float viewport_width = std::max(1.0f, full_viewport->WorkSize.x);
        const float viewport_height = std::max(1.0f, full_viewport->WorkSize.y);
        const float viewport_span = std::min(viewport_width, viewport_height);
        const float maximum_view_scale = maximumViewScale(viewport_span);
        double cursor_x = 0.0;
        double cursor_y = 0.0;
        glfwGetCursorPos(window, &cursor_x, &cursor_y);
        const bool over_world = cursor_x >= work_min.x && cursor_x <= work_max.x
            && cursor_y >= work_min.y + 28.0 && cursor_y <= work_max.y;
        const bool ui_using_mouse = ImGui::IsAnyItemActive() || add_body_;
        if (over_world && !ui_using_mouse && pending_scroll_ != 0.0) {
            const float old_scale = view_scale_;
            view_scale_ = std::clamp(view_scale_ * std::pow(1.15f, static_cast<float>(pending_scroll_)),
                                     0.25f, maximum_view_scale);
            const ImVec2 mouse{static_cast<float>(cursor_x), static_cast<float>(cursor_y)};
            const float center_x = full_viewport->WorkPos.x + viewport_width * 0.5f;
            const float center_y = full_viewport->WorkPos.y + viewport_height * 0.5f;
            const float world_x = camera_x_ + (mouse.x - center_x) / old_scale;
            const float world_y = camera_y_ - (mouse.y - center_y) / old_scale;
            camera_x_ = world_x - (mouse.x - center_x) / view_scale_;
            camera_y_ = world_y + (mouse.y - center_y) / view_scale_;
        }
        pending_scroll_ = 0.0;
        if (over_world && !ui_using_mouse
            && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            if (!have_cursor_position_) {
                last_cursor_x_ = cursor_x;
                last_cursor_y_ = cursor_y;
                have_cursor_position_ = true;
            }
            const double delta_x = cursor_x - last_cursor_x_;
            const double delta_y = cursor_y - last_cursor_y_;
            if (delta_x != 0.0 || delta_y != 0.0) {
                commands_.dispatch(nbody::app::FocusBody{std::nullopt});
                camera_x_ -= static_cast<float>(delta_x) / view_scale_;
                camera_y_ += static_cast<float>(delta_y) / view_scale_;
            }
        }
        last_cursor_x_ = cursor_x;
        last_cursor_y_ = cursor_y;
        have_cursor_position_ = true;
    }

    double worldKilometersPerUnit() const {
        if (!scaled_solar_system_) return 1.0;
        constexpr double kilometersPerAu = 149597870.7;
        return 30.070 * kilometersPerAu / 400.0;
    }

    float maximumViewScale(float viewport_width) const {
        constexpr double moon_diameter_km = 2.0 * 1737.4;
        constexpr double margin_factor = 1.3;
        const double moon_span_world = moon_diameter_km * margin_factor / worldKilometersPerUnit();
        return std::max(0.25f, static_cast<float>(viewport_width / moon_span_world));
    }

    float gridStep(float viewport_span) const {
        // The Moon reference zoom is defined as four cells across the
        // reference span. Keeping this proportional also makes the indicator
        // describe the actual rendered grid rather than a separate scale.
        return viewport_span / (4.0f * view_scale_);
    }

    struct GridDistance {
        double value;
        const char* unit;
    };

    GridDistance gridDistance(float viewport_width) const {
        constexpr double kilometersPerAu = 149597870.7;
        constexpr double kilometersPerLightYear = 9.460730472e12;
        const double kilometers = static_cast<double>(gridStep(viewport_width)) * worldKilometersPerUnit();
        if (kilometers < 0.01 * kilometersPerAu) return {kilometers, "km"};
        if (kilometers < kilometersPerLightYear) return {kilometers / kilometersPerAu, "AU"};
        return {kilometers / kilometersPerLightYear, "Ly"};
    }

    void createDebugSolarSystem() {
        struct PlanetDefinition {
            const char* name;
            double orbit_au;
            double mass_ratio;
            double radius_km;
        };
        constexpr std::array<PlanetDefinition, 8> planets{{
            {"Mercury", 0.387, 1.660e-7, 2439.7},
            {"Venus", 0.723, 2.447e-6, 6051.8},
            {"Earth", 1.000, 3.003e-6, 6371.0},
            {"Mars", 1.524, 3.227e-7, 3389.5},
            {"Jupiter", 5.203, 9.545e-4, 69911.0},
            {"Saturn", 9.537, 2.857e-4, 58232.0},
            {"Uranus", 19.191, 4.366e-5, 25362.0},
            {"Neptune", 30.070, 5.151e-5, 24622.0}
        }};
        constexpr double neptune_orbit = 400.0;
        constexpr double scale = neptune_orbit / 30.070;
        constexpr double kilometers_per_au = 149597870.7;
        constexpr double radius_scale = neptune_orbit / (30.070 * kilometers_per_au);
        // Calibrated so the Earth orbit is approximately one Julian year in
        // simulation seconds after Neptune is mapped to radius 400.
        constexpr double gravitational_constant = 9.33076e-11;
        constexpr double sun_mass = 1.0;
        constexpr double pi = 3.14159265358979323846;
        std::mt19937 generator(20260918);
        std::uniform_real_distribution<double> phase(0.0, 2.0 * pi);
        Vec3 earth_position{};
        Vec3 earth_velocity{};
        Vec3 jupiter_position{};
        Vec3 jupiter_velocity{};
        constexpr double earth_mass = 3.003e-6;
        constexpr double jupiter_mass = 9.545e-4;

        commands_.dispatch(nbody::app::ClearBodies{});
        BodyState sun;
        sun.mass = sun_mass;
        sun.radius = 696340.0 * radius_scale;
        sun.is_static = true;
        sun.kind = BodyKind::Star;
        commands_.dispatch(nbody::app::CreateBody{sun});

        for (const PlanetDefinition& planet : planets) {
            const double orbit = planet.orbit_au * scale;
            const double angle = phase(generator);
            const double speed = std::sqrt(gravitational_constant * sun_mass / orbit);
            BodyState body;
            body.position = {orbit * std::cos(angle), orbit * std::sin(angle), 0.0};
            body.velocity = {-speed * std::sin(angle), speed * std::cos(angle), 0.0};
            body.mass = planet.mass_ratio;
            body.radius = planet.radius_km * radius_scale;
            body.kind = BodyKind::Ordinary;
            commands_.dispatch(nbody::app::CreateBody{body});
            if (std::string_view(planet.name) == "Earth") {
                earth_position = body.position;
                earth_velocity = body.velocity;
            } else if (std::string_view(planet.name) == "Jupiter") {
                jupiter_position = body.position;
                jupiter_velocity = body.velocity;
            }
        }

        const auto addMoon = [&](const Vec3& parent_position, const Vec3& parent_velocity,
                                 double parent_mass, double distance_km, double radius_km,
                                 double mass_ratio, double orbit_phase) {
            const double distance_world = distance_km * radius_scale;
            const double orbital_speed = std::sqrt(gravitational_constant * parent_mass / distance_world);
            const Vec3 offset{distance_world * std::cos(orbit_phase),
                              distance_world * std::sin(orbit_phase), 0.0};
            const Vec3 tangent{-std::sin(orbit_phase) * orbital_speed,
                               std::cos(orbit_phase) * orbital_speed, 0.0};
            BodyState moon;
            moon.position = parent_position + offset;
            moon.velocity = parent_velocity + tangent;
            moon.mass = mass_ratio;
            moon.radius = radius_km * radius_scale;
            moon.kind = BodyKind::Ordinary;
            commands_.dispatch(nbody::app::CreateBody{moon});
        };

        // Values are physical present-day means, expressed in the demo's
        // solar-mass and kilometre-to-world-unit scales.
        addMoon(earth_position, earth_velocity, earth_mass,
                384400.0, 1737.4, 3.694e-8, phase(generator));
        addMoon(jupiter_position, jupiter_velocity, jupiter_mass,
                421700.0, 1821.6, 4.49e-8, phase(generator)); // Io
        addMoon(jupiter_position, jupiter_velocity, jupiter_mass,
                671034.0, 1560.8, 2.41e-8, phase(generator)); // Europa
        addMoon(jupiter_position, jupiter_velocity, jupiter_mass,
                1070412.0, 2634.1, 7.57e-8, phase(generator)); // Ganymede
        addMoon(jupiter_position, jupiter_velocity, jupiter_mass,
                1882709.0, 2410.3, 5.70e-8, phase(generator)); // Callisto
        accumulator_ = 0.0;
        scaled_solar_system_ = true;
        simulation_timestep_ = 3600.0;
        time_scale_value_ = 1.0;
        time_scale_unit_ = 1;
        camera_x_ = 0.0f;
        camera_y_ = 0.0f;
        view_scale_ = 1.5f;
    }

    void drawWorld(GLFWwindow* window) {
        // The bottom bar owns camera status and controls. Keep this hook for
        // the future world-space overlay layer.
        (void)window;
    }
};

} // namespace

} // namespace nbody::frontend

namespace nbody::rendering {

struct VulkanRenderer::Impl {
    ::nbody::frontend::VulkanFrontendContext context;
};

VulkanRenderer::VulkanRenderer() : impl_(new Impl) {}

VulkanRenderer::~VulkanRenderer() {
    delete impl_;
}

void VulkanRenderer::initialize(GLFWwindow* window) {
    impl_->context.initialize(window);
}

void VulkanRenderer::render(GLFWwindow* window, ImDrawData* draw_data,
                            const ::nbody::rendering::RenderScene& scene) {
    impl_->context.render(window, draw_data, scene);
}

} // namespace rendering

namespace nbody::frontend {

int VulkanFrontend::run() {
    if (!glfwInit()) throw std::runtime_error("GLFW initialization failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "N-Body Simulator", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("GLFW window creation failed");
    }
    try {
        rendering::VulkanRenderer renderer;
        renderer.initialize(window);
        VulkanFrontendState application;
        glfwSetWindowUserPointer(window, &application);
        glfwSetScrollCallback(window, VulkanFrontendState::scrollCallback);
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            application.draw(window, renderer);
            // Swapchain recreation reinitializes ImGui's GLFW backend, which
            // installs its own scroll callback. Restore the frontend callback
            // after each frame so zoom input remains owned by the frontend.
            glfwSetScrollCallback(window, VulkanFrontendState::scrollCallback);
        }
    } catch (...) {
        glfwDestroyWindow(window);
        glfwTerminate();
        throw;
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

}
