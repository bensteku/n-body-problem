#pragma once

#include "simulation/body_id.hpp"
#include "simulation/world_state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace nbody::rendering {

using FrameId = std::uint64_t;

// This is a logical storage classification. It deliberately does not name a
// graphics API: Vulkan and WebGPU resources are owned by their respective
// application targets and are never mixed in one binary.
enum class FrameStorageKind {
    CpuSnapshot,
    ExternalGpuResource
};

struct FrameDescription {
    FrameId id{};
    Dimension dimension{Dimension::Two};
    double simulation_time{};
    std::size_t body_count{};
    std::size_t previous_body_count{};
    bool body_count_changed{};
};

// The schema describes the byte-level/logical contents of a published frame.
// It is part of the contract between simulation producers and renderers, not
// a renderer-specific vertex declaration.
struct FrameSchema {
    std::uint32_t version{1};
    bool includes_velocity{true};
    bool includes_mass{true};
    bool includes_radius{true};
    bool includes_stable_body_id{true};
    bool includes_visual_kind{true};
};

enum class ReadinessKind {
    Immediate,
    ExternalTimeline
};

// `value` is interpreted by the target backend. For CPU frames it is zero. For
// a native GPU target it may represent a timeline/fence value without placing
// a Vulkan or WebGPU synchronization type in shared code.
struct FrameReadiness {
    ReadinessKind kind{ReadinessKind::Immediate};
    std::uint64_t value{};
};

// Opaque target-owned resource identity and lifetime. The owning backend may
// associate `resource_id` with a Vulkan or WebGPU buffer in its own registry.
// Holding `lifetime` keeps the underlying allocation alive while the frame is
// retained by the renderer, without exposing backend objects here.
struct ExternalFrameResource {
    std::uint64_t resource_id{};
    std::uint32_t backend_tag{};
    std::shared_ptr<const void> lifetime;

    explicit operator bool() const {
        return resource_id != 0 && static_cast<bool>(lifetime);
    }
};

struct FrameStorageDescription {
    FrameStorageKind kind{FrameStorageKind::CpuSnapshot};
    ExternalFrameResource external_gpu;
};

}
