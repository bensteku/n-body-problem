#include "rendering/vulkan_upload_arena.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace nbody::rendering {

VulkanUploadArena::~VulkanUploadArena() {
    shutdown();
}

void VulkanUploadArena::initialize(VkDevice device, VkPhysicalDevice physical_device,
                                   VkDeviceSize initial_capacity) {
    shutdown();
    device_ = device;
    physical_device_ = physical_device;
    createBuffer(initial_capacity);
}

void VulkanUploadArena::shutdown() {
    if (!device_) return;
    if (mapping_) vkUnmapMemory(device_, memory_);
    if (buffer_) vkDestroyBuffer(device_, buffer_, nullptr);
    if (memory_) vkFreeMemory(device_, memory_, nullptr);
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    mapping_ = nullptr;
    capacity_ = 0;
    cursor_ = 0;
    owner_fence_ = VK_NULL_HANDLE;
}

void VulkanUploadArena::beginFrame(VkFence completed_fence) {
    if (owner_fence_ && owner_fence_ != completed_fence
        && vkGetFenceStatus(device_, owner_fence_) != VK_SUCCESS) {
        throw std::runtime_error("upload arena reused before its fence completed");
    }
    cursor_ = 0;
    owner_fence_ = completed_fence;
}

void VulkanUploadArena::endFrame(VkFence submitted_fence) {
    owner_fence_ = submitted_fence;
}

VulkanUploadAllocation VulkanUploadArena::allocate(VkDeviceSize size, VkDeviceSize alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::runtime_error("upload alignment must be a non-zero power of two");
    }
    const VkDeviceSize offset = (cursor_ + alignment - 1) / alignment * alignment;
    ensureCapacity(offset + size);
    cursor_ = offset + size;
    return {offset, size};
}

void VulkanUploadArena::write(VulkanUploadAllocation allocation,
                              std::span<const std::byte> data) {
    if (data.size_bytes() > allocation.size
        || allocation.offset + data.size_bytes() > cursor_) {
        throw std::runtime_error("upload write exceeds active arena allocation");
    }
    if (!data.empty()) {
        std::memcpy(static_cast<std::byte*>(mapping_) + allocation.offset,
                    data.data(), data.size_bytes());
    }
}

void VulkanUploadArena::ensureCapacity(VkDeviceSize required) {
    if (required <= capacity_) return;
    const VkDeviceSize expanded = std::max<VkDeviceSize>(
        required, capacity_ == 0 ? 4 * 1024 * 1024 : capacity_ * 2);
    if (mapping_) vkUnmapMemory(device_, memory_);
    if (buffer_) vkDestroyBuffer(device_, buffer_, nullptr);
    if (memory_) vkFreeMemory(device_, memory_, nullptr);
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    mapping_ = nullptr;
    capacity_ = 0;
    createBuffer(expanded);
}

void VulkanUploadArena::createBuffer(VkDeviceSize capacity) {
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = capacity;
    buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device_, &buffer_info, nullptr, &buffer_), "create upload buffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, buffer_, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkAllocateMemory(device_, &allocation, nullptr, &memory_), "allocate upload memory");
    check(vkBindBufferMemory(device_, buffer_, memory_, 0), "bind upload memory");
    check(vkMapMemory(device_, memory_, 0, capacity, 0, &mapping_), "map upload memory");
    capacity_ = capacity;
}

std::uint32_t VulkanUploadArena::findMemoryType(std::uint32_t filter,
                                                 VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory);
    for (std::uint32_t index = 0; index < memory.memoryTypeCount; ++index) {
        if ((filter & (1u << index))
            && (memory.memoryTypes[index].propertyFlags & properties) == properties) {
            return index;
        }
    }
    throw std::runtime_error("no compatible upload memory type");
}

void VulkanUploadArena::check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(operation) + " failed");
}

}
