#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace nbody::rendering {

enum class VulkanShaderId : std::uint8_t {
    BodyVertex,
    BodyFragment,
    GridVertex,
    GridFragment,
};

enum class VulkanShaderInterface : std::uint8_t {
    Body2D,
    Grid2D,
};

struct VulkanPipelineDescription {
    std::string_view name;
    VulkanShaderId vertex_shader;
    VulkanShaderId fragment_shader;
    VulkanShaderInterface interface;
    VkRenderPass render_pass;
    VkPipelineVertexInputStateCreateInfo vertex_input;
    VkPrimitiveTopology topology;
    std::uint32_t subpass{};
    bool depth_test;
    bool depth_write;
    bool alpha_blend;
};

struct VulkanPipelineHandle {
    VkPipeline pipeline{};
    VkPipelineLayout layout{};
};

class VulkanPipelineManager {
public:
    VulkanPipelineManager() = default;
    ~VulkanPipelineManager();

    VulkanPipelineManager(const VulkanPipelineManager&) = delete;
    VulkanPipelineManager& operator=(const VulkanPipelineManager&) = delete;

    void initialize(VkDevice device, std::filesystem::path shader_directory);
    VulkanPipelineHandle createGraphicsPipeline(const VulkanPipelineDescription& description);
    void destroyAll();

private:
    struct ShaderModule {
        VkShaderModule module{};
        VkShaderStageFlagBits stage{};
        VulkanShaderInterface interface{};
    };

    ShaderModule loadShader(VulkanShaderId id);
    static const char* shaderFileName(VulkanShaderId id);
    static VkShaderStageFlagBits shaderStage(VulkanShaderId id);
    static VulkanShaderInterface shaderInterface(VulkanShaderId id);
    static void check(VkResult result, const char* operation);

    VkDevice device_{};
    std::filesystem::path shader_directory_;
    std::vector<VulkanPipelineHandle> pipelines_;
    std::vector<VkShaderModule> shaders_;
};

}
