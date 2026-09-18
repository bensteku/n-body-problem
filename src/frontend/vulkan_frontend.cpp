#include "frontend/vulkan_frontend.hpp"

#include "simulation/physics_session.hpp"

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
#include <deque>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace nbody::frontend {

namespace {

void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(operation) + " failed");
}

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

    void initialize(GLFWwindow* window) {
        createInstance();
        check(glfwCreateWindowSurface(instance, window, nullptr, &surface), "window surface");
        selectDevice();
        createDevice();
        createSwapchain(window);
        createRenderPass();
        createFramebuffers();
        createCommands();
        createSync();
        initializeImGui(window);
    }

    void render(ImDrawData* draw_data) {
        check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "wait fence");
        check(vkResetFences(device, 1, &fence), "reset fence");
        std::uint32_t image = 0;
        const VkResult acquired = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                                         image_available, VK_NULL_HANDLE, &image);
        if (acquired == VK_ERROR_OUT_OF_DATE_KHR) return;
        check(acquired, "acquire swapchain image");
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
        if (presented != VK_SUCCESS && presented != VK_SUBOPTIMAL_KHR) check(presented, "present");
    }

    ~VulkanFrontendContext() {
        if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device);
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
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

class ApplicationState {
public:
    void draw(GLFWwindow* window, VulkanFrontendContext& context) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - last_frame_).count();
        last_frame_ = now;
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        if (title_screen_) drawTitle();
        else drawSimulation(window, std::min(elapsed, 0.1));
        ImGui::Render();
        context.render(ImGui::GetDrawData());
    }

    void onScroll(double offset) { pending_scroll_ += offset; }

    static void scrollCallback(GLFWwindow* window, double x_offset, double y_offset) {
        ImGui_ImplGlfw_ScrollCallback(window, x_offset, y_offset);
        if (auto* state = static_cast<ApplicationState*>(glfwGetWindowUserPointer(window))) {
            state->onScroll(y_offset);
        }
    }

private:
    PhysicsSession session_;
    WorldState world_{Dimension::Two};
    SimulationParameters parameters_;
    std::optional<FramePublication> publication_;
    std::deque<WorldState> history_;
    std::chrono::steady_clock::time_point last_frame_{std::chrono::steady_clock::now()};
    double accumulator_{};
    double time_scale_value_{1.0};
    double simulation_timestep_{0.001};
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
    float tangential_velocity_{};
    bool title_screen_{true};
    bool running_{};
    bool reversing_{};
    bool grid_{};
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

    static constexpr std::size_t max_history_frames_ = 600;

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
            title_screen_ = false;
        }
        ImGui::TextDisabled("Press Enter to begin");
        ImGui::End();
    }

    void drawSimulation(GLFWwindow* window, double elapsed) {
        updateCameraInput(window);
        parameters_.dimension = Dimension::Two;
        parameters_.timestep = std::clamp(simulation_timestep_, 1.0e-5, 86400.0);
        parameters_.gravitational_constant = scaled_solar_system_ ? 9.33076e-11 : 0.1;
        parameters_.softening_length = 0.05;
        parameters_.collision.model = CollisionModel::Transparent;
        const double requested_rate = timeScaleSecondsPerRealSecond();
        accumulator_ += elapsed * requested_rate;
        std::size_t steps_this_frame = 0;
        while (running_ && accumulator_ >= parameters_.timestep && steps_this_frame < 256) {
            history_.push_back(world_);
            if (history_.size() > max_history_frames_) history_.pop_front();
            session_.step(world_, parameters_);
            accumulator_ -= parameters_.timestep;
            ++steps_this_frame;
        }
        if (reversing_ && !history_.empty()) {
            world_ = std::move(history_.back());
            history_.pop_back();
            accumulator_ = 0.0;
        }
        publication_ = session_.publishFrame(world_);
        ImGui::BeginMainMenuBar();
        if (ImGui::Button(running_ ? "Pause" : "Start")) {
            running_ = !running_;
            reversing_ = false;
        }
        ImGui::SameLine();
        if (ImGui::Button(reversing_ ? "Stop reverse" : "Reverse")) {
            reversing_ = !reversing_;
            running_ = false;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110.0f);
        ImGui::InputDouble("##time-scale", &time_scale_value_, 0.1, 1.0, "%.3f");
        ImGui::SameLine();
        const char* units[] = {"seconds / real second", "days / real second",
                               "months / real second", "years / real second"};
        ImGui::SetNextItemWidth(170.0f);
        ImGui::Combo("##time-scale-unit", &time_scale_unit_, units, 4);
        ImGui::SameLine();
        ImGui::Text("sim time %.3f | bodies %zu", world_.time(), world_.bodyCount());
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90.0f);
        ImGui::InputDouble("dt", &simulation_timestep_, 0.0001, 0.001, "%.5f");
        ImGui::SameLine();
        if (ImGui::Button(grid_ ? "Hide grid" : "Show grid")) grid_ = !grid_;
        ImGui::SameLine();
        if (ImGui::Button("Add body")) add_body_ = true;
        ImGui::SameLine();
        if (ImGui::Button("Debug solar system")) createDebugSolarSystem();
        ImGui::EndMainMenuBar();
        if (steps_this_frame == 256 && accumulator_ >= parameters_.timestep) {
            ImGui::Begin("Playback status", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::Text("Playback is rate-limited: reduce timescale or timestep.");
            ImGui::End();
        }
        if (add_body_) drawAddBody();
        drawWorld(window);
    }

    double timeScaleSecondsPerRealSecond() const {
        const double value = std::max(0.0, time_scale_value_);
        switch (time_scale_unit_) {
        case 1: return value * 86400.0;
        case 2: return value * 30.0 * 86400.0;
        case 3: return value * 365.25 * 86400.0;
        default: return value;
        }
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
            history_.clear();
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
                body.velocity.x += -std::sin(angle) * tangential_velocity_;
                body.velocity.y += std::cos(angle) * tangential_velocity_;
            }
            body.mass = generator_ == 0 ? std::max(0.001f, body_mass_) : std::max(0.001f, mass(generator));
            body.radius = generator_ == 0 ? std::max(0.001f, body_radius_) : std::max(0.001f, radius(generator));
            body.is_static = static_body_;
            world_.addBody(body);
        }
    }

    void updateCameraInput(GLFWwindow* window) {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        const ImVec2 work_min = viewport->WorkPos;
        const ImVec2 work_max{viewport->WorkPos.x + viewport->WorkSize.x,
                              viewport->WorkPos.y + viewport->WorkSize.y};
        double cursor_x = 0.0;
        double cursor_y = 0.0;
        glfwGetCursorPos(window, &cursor_x, &cursor_y);
        const bool over_world = cursor_x >= work_min.x && cursor_x <= work_max.x
            && cursor_y >= work_min.y + 28.0 && cursor_y <= work_max.y;
        const bool ui_using_mouse = ImGui::IsAnyItemActive() || add_body_;
        if (over_world && !ui_using_mouse && pending_scroll_ != 0.0) {
            const float old_scale = view_scale_;
            view_scale_ = std::clamp(view_scale_ * std::pow(1.15f, static_cast<float>(pending_scroll_)),
                                      0.25f, 200.0f);
            const ImVec2 mouse{static_cast<float>(cursor_x), static_cast<float>(cursor_y)};
            const float center_x = work_min.x + viewport->WorkSize.x * 0.5f;
            const float center_y = work_min.y + viewport->WorkSize.y * 0.5f + 20.0f;
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
            camera_x_ -= static_cast<float>(cursor_x - last_cursor_x_) / view_scale_;
            camera_y_ += static_cast<float>(cursor_y - last_cursor_y_) / view_scale_;
        }
        last_cursor_x_ = cursor_x;
        last_cursor_y_ = cursor_y;
        have_cursor_position_ = true;
    }

    void createDebugSolarSystem() {
        struct PlanetDefinition {
            const char* name;
            double orbit_au;
            double mass_ratio;
        };
        constexpr std::array<PlanetDefinition, 8> planets{{
            {"Mercury", 0.387, 1.660e-7},
            {"Venus", 0.723, 2.447e-6},
            {"Earth", 1.000, 3.003e-6},
            {"Mars", 1.524, 3.227e-7},
            {"Jupiter", 5.203, 9.545e-4},
            {"Saturn", 9.537, 2.857e-4},
            {"Uranus", 19.191, 4.366e-5},
            {"Neptune", 30.070, 5.151e-5}
        }};
        constexpr double neptune_orbit = 400.0;
        constexpr double scale = neptune_orbit / 30.070;
        // Calibrated so the Earth orbit is approximately one Julian year in
        // simulation seconds after Neptune is mapped to radius 400.
        constexpr double gravitational_constant = 9.33076e-11;
        constexpr double sun_mass = 1.0;
        constexpr double pi = 3.14159265358979323846;
        std::mt19937 generator(20260918);
        std::uniform_real_distribution<double> phase(0.0, 2.0 * pi);

        world_ = WorldState(Dimension::Two);
        BodyState sun;
        sun.mass = sun_mass;
        sun.radius = 5.0;
        sun.is_static = true;
        sun.kind = BodyKind::Star;
        world_.addBody(sun);

        for (const PlanetDefinition& planet : planets) {
            const double orbit = planet.orbit_au * scale;
            const double angle = phase(generator);
            const double speed = std::sqrt(gravitational_constant * sun_mass / orbit);
            BodyState body;
            body.position = {orbit * std::cos(angle), orbit * std::sin(angle), 0.0};
            body.velocity = {-speed * std::sin(angle), speed * std::cos(angle), 0.0};
            body.mass = planet.mass_ratio;
            body.radius = 1.5;
            body.kind = BodyKind::Ordinary;
            world_.addBody(body);
        }
        history_.clear();
        accumulator_ = 0.0;
        scaled_solar_system_ = true;
        simulation_timestep_ = 3600.0;
        time_scale_value_ = 1.0;
        time_scale_unit_ = 1;
        running_ = false;
        reversing_ = false;
        camera_x_ = 0.0f;
        camera_y_ = 0.0f;
        view_scale_ = 1.5f;
    }

    void drawWorld(GLFWwindow* window) {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        const ImVec2 origin{viewport->WorkPos.x + viewport->WorkSize.x * 0.5f
                                - camera_x_ * view_scale_,
                            viewport->WorkPos.y + viewport->WorkSize.y * 0.5f + 20.0f
                                + camera_y_ * view_scale_};
        const ImVec2 work_min = viewport->WorkPos;
        const ImVec2 work_max{viewport->WorkPos.x + viewport->WorkSize.x,
                              viewport->WorkPos.y + viewport->WorkSize.y};
        ImGui::SetNextWindowPos({work_min.x + 12.0f, work_max.y - 42.0f}, ImGuiCond_Always);
        ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize
            | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing);
        ImGui::Text("Zoom %.2f  |  Scroll zooms, drag pans", view_scale_);
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset view")) {
            camera_x_ = 0.0f;
            camera_y_ = 0.0f;
            view_scale_ = 20.0f;
        }
        ImGui::End();
        ImDrawList* draw = ImGui::GetForegroundDrawList();
        if (grid_) {
            const float world_width = viewport->WorkSize.x / view_scale_;
            const float world_height = viewport->WorkSize.y / view_scale_;
            const float grid_step = view_scale_ < 2.0f ? 50.0f : (view_scale_ < 8.0f ? 10.0f : 1.0f);
            const int min_x = static_cast<int>(std::floor(camera_x_ - world_width * 0.5f / grid_step)) - 1;
            const int max_x = static_cast<int>(std::ceil(camera_x_ + world_width * 0.5f / grid_step)) + 1;
            const int min_y = static_cast<int>(std::floor(camera_y_ - world_height * 0.5f / grid_step)) - 1;
            const int max_y = static_cast<int>(std::ceil(camera_y_ + world_height * 0.5f / grid_step)) + 1;
            for (int line = min_x; line <= max_x; ++line) {
                const float world = static_cast<float>(line) * grid_step;
                const float x = origin.x + world * view_scale_;
                draw->AddLine({x, work_min.y}, {x, work_max.y}, 0x30384A55);
            }
            for (int line = min_y; line <= max_y; ++line) {
                const float world = static_cast<float>(line) * grid_step;
                const float y = origin.y - world * view_scale_;
                draw->AddLine({work_min.x, y}, {work_max.x, y}, 0x30384A55);
            }
        }
        if (publication_ && publication_->published()) {
            for (const RenderBody& body : publication_->cpu_snapshot->bodies) {
                const ImVec2 position{origin.x + static_cast<float>(body.position.x) * view_scale_,
                                      origin.y - static_cast<float>(body.position.y) * view_scale_};
                const float radius = std::clamp(static_cast<float>(body.radius * view_scale_), 2.0f, 24.0f);
                const ImU32 color = body.kind == BodyKind::Star ? 0xFF60D0FFFF
                    : (body.is_static ? 0xFFB080FF : 0x80D8FFFF);
                draw->AddCircleFilled(position, radius, color);
                draw->AddCircle(position, radius, 0xFFFFFFFF, 16, 1.0f);
            }
        }
        (void)window;
    }
};

} // namespace

int VulkanFrontend::run() {
    if (!glfwInit()) throw std::runtime_error("GLFW initialization failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // Swapchain recreation is deliberately deferred to the next frontend
    // slice; keep the prototype window fixed until that lifecycle is present.
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "N-Body Simulator", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("GLFW window creation failed");
    }
    try {
        VulkanFrontendContext context;
        context.initialize(window);
        ApplicationState application;
        glfwSetWindowUserPointer(window, &application);
        glfwSetScrollCallback(window, ApplicationState::scrollCallback);
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            application.draw(window, context);
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
