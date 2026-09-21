#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace nbody::rendering {

struct VulkanUploadAllocation {
    VkDeviceSize offset{};
    VkDeviceSize size{};
};

// A persistently mapped, fence-owned upload resource for one in-flight frame.
// The renderer waits the owner fence before beginning a new frame, allowing
// the allocation cursor and backing buffer to be reused without extra copies.
class VulkanUploadArena {
public:
    VulkanUploadArena() = default;
    ~VulkanUploadArena();

    VulkanUploadArena(const VulkanUploadArena&) = delete;
    VulkanUploadArena& operator=(const VulkanUploadArena&) = delete;

    void initialize(VkDevice device, VkPhysicalDevice physical_device,
                    VkDeviceSize initial_capacity);
    void shutdown();

    void beginFrame(VkFence completed_fence);
    void endFrame(VkFence submitted_fence);
    VulkanUploadAllocation allocate(VkDeviceSize size, VkDeviceSize alignment = 256);
    void write(VulkanUploadAllocation allocation, std::span<const std::byte> data);

    VkBuffer buffer() const { return buffer_; }
    VkDeviceSize bytesUsed() const { return cursor_; }

private:
    void ensureCapacity(VkDeviceSize required);
    void createBuffer(VkDeviceSize capacity);
    std::uint32_t findMemoryType(std::uint32_t filter, VkMemoryPropertyFlags properties) const;
    static void check(VkResult result, const char* operation);

    VkDevice device_{};
    VkPhysicalDevice physical_device_{};
    VkBuffer buffer_{};
    VkDeviceMemory memory_{};
    void* mapping_{};
    VkDeviceSize capacity_{};
    VkDeviceSize cursor_{};
    VkFence owner_fence_{};
};

}
