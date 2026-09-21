#pragma once

#include "rendering/render_scene.hpp"

struct GLFWwindow;
struct ImDrawData;

namespace nbody::rendering {

// Native Vulkan rendering boundary. The frontend supplies logical scenes;
// Vulkan handles and swapchain lifetime remain private to the implementation.
class VulkanRenderer {
public:
    VulkanRenderer();
    ~VulkanRenderer();

    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;

    void initialize(GLFWwindow* window);
    void render(GLFWwindow* window, ImDrawData* draw_data,
                const ::nbody::rendering::RenderScene& scene);

private:
    struct Impl;
    Impl* impl_{};
};

}
