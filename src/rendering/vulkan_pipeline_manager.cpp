#include "rendering/vulkan_pipeline_manager.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace nbody::rendering {

VulkanPipelineManager::~VulkanPipelineManager() {
    destroyAll();
}

void VulkanPipelineManager::initialize(VkDevice device, std::filesystem::path shader_directory) {
    destroyAll();
    device_ = device;
    shader_directory_ = std::move(shader_directory);
}

VulkanPipelineHandle VulkanPipelineManager::createGraphicsPipeline(
    const VulkanPipelineDescription& description) {
    const ShaderModule vertex = loadShader(description.vertex_shader);
    const ShaderModule fragment = loadShader(description.fragment_shader);
    if (vertex.interface != description.interface || fragment.interface != description.interface) {
        throw std::runtime_error(std::string("shader interface mismatch for pipeline ")
            + std::string(description.name));
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = vertex.stage;
    stages[0].module = vertex.module;
    stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = fragment.stage;
    stages[1].module = fragment.module;
    stages[1].pName = "main";

    VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = description.topology;

    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterization.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = description.depth_test ? VK_TRUE : VK_FALSE;
    depth.depthWriteEnable = description.depth_write ? VK_TRUE : VK_FALSE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
        | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blend.blendEnable = description.alpha_blend ? VK_TRUE : VK_FALSE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.alphaBlendOp = VK_BLEND_OP_ADD;
    VkPipelineColorBlendStateCreateInfo color_blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &blend;

    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;

    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout layout{};
    check(vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout),
          "create graphics pipeline layout");

    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &description.vertex_input;
    pipeline.pInputAssemblyState = &assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &rasterization;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depth;
    pipeline.pColorBlendState = &color_blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = layout;
    pipeline.renderPass = description.render_pass;
    pipeline.subpass = description.subpass;

    VkPipeline result{};
    const VkResult created = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline,
                                                       nullptr, &result);
    if (created != VK_SUCCESS) {
        vkDestroyPipelineLayout(device_, layout, nullptr);
        check(created, "create graphics pipeline");
    }
    pipelines_.push_back({result, layout});
    return {result, layout};
}

void VulkanPipelineManager::destroyAll() {
    if (!device_) return;
    for (const VulkanPipelineHandle handle : pipelines_) {
        if (handle.pipeline) vkDestroyPipeline(device_, handle.pipeline, nullptr);
        if (handle.layout) vkDestroyPipelineLayout(device_, handle.layout, nullptr);
    }
    for (const VkShaderModule shader : shaders_) {
        if (shader) vkDestroyShaderModule(device_, shader, nullptr);
    }
    pipelines_.clear();
    shaders_.clear();
}

VulkanPipelineManager::ShaderModule VulkanPipelineManager::loadShader(VulkanShaderId id) {
    const std::filesystem::path path = shader_directory_ / shaderFileName(id);
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("could not open shader " + path.string());
    const std::streamsize size = input.tellg();
    if (size <= 0 || size % 4 != 0) throw std::runtime_error("invalid SPIR-V shader size: " + path.string());
    std::vector<char> bytes(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(bytes.data(), size);
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = bytes.size();
    info.pCode = reinterpret_cast<const std::uint32_t*>(bytes.data());
    VkShaderModule module{};
    check(vkCreateShaderModule(device_, &info, nullptr, &module), "create shader module");
    shaders_.push_back(module);
    return {module, shaderStage(id), shaderInterface(id)};
}

const char* VulkanPipelineManager::shaderFileName(VulkanShaderId id) {
    switch (id) {
    case VulkanShaderId::BodyVertex: return "body.vert.spv";
    case VulkanShaderId::BodyFragment: return "body.frag.spv";
    case VulkanShaderId::GridVertex: return "grid.vert.spv";
    case VulkanShaderId::GridFragment: return "grid.frag.spv";
    }
    return "";
}

VkShaderStageFlagBits VulkanPipelineManager::shaderStage(VulkanShaderId id) {
    return id == VulkanShaderId::BodyVertex || id == VulkanShaderId::GridVertex
        ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
}

VulkanShaderInterface VulkanPipelineManager::shaderInterface(VulkanShaderId id) {
    return id == VulkanShaderId::BodyVertex || id == VulkanShaderId::BodyFragment
        ? VulkanShaderInterface::Body2D : VulkanShaderInterface::Grid2D;
}

void VulkanPipelineManager::check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(operation) + " failed");
}

}
