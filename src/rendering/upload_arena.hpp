#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace nbody::rendering {

// Backend-neutral staging storage policy. Vulkan maps this into a persistently
// mapped upload buffer; WebGPU can use the same bounded frame-local policy
// with queue writes. It deliberately owns no graphics handles.
class UploadArena {
public:
    explicit UploadArena(std::size_t capacity = 0) : storage_(capacity) {}

    void reset() { cursor_ = 0; }

    std::span<std::byte> allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        const std::size_t aligned = (cursor_ + alignment - 1) & ~(alignment - 1);
        if (aligned + size > storage_.size()) storage_.resize(std::max(storage_.size() * 2, aligned + size));
        cursor_ = aligned + size;
        return {storage_.data() + aligned, size};
    }

    template<typename T>
    std::span<T> copy(std::span<const T> values) {
        const auto bytes = allocate(values.size_bytes(), alignof(T));
        std::memcpy(bytes.data(), values.data(), values.size_bytes());
        return {reinterpret_cast<T*>(bytes.data()), values.size()};
    }

    std::size_t bytesUsed() const { return cursor_; }
    std::size_t capacity() const { return storage_.size(); }

private:
    std::vector<std::byte> storage_;
    std::size_t cursor_{};
};

}
