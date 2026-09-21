#pragma once

#include "simulation/simulation_frame.hpp"

#include <cstdint>
#include <span>
#include <string_view>
#include <utility>

namespace nbody::rendering {

using MaterialId = std::uint32_t;

struct RenderCamera {
    Vec3 position{};
    Vec3 target{};
    Vec3 up{0.0, 1.0, 0.0};
    double vertical_field_of_view{0.7853981633974483};
    double orthographic_scale{1.0};
    double zoom{1.0};
    bool orthographic{true};
};

enum class RenderObjectFlags : std::uint32_t {
    None = 0,
    Selected = 1u << 0,
    Focused = 1u << 1,
    Static = 1u << 2,
    Emissive = 1u << 3
};

constexpr RenderObjectFlags operator|(RenderObjectFlags left, RenderObjectFlags right) {
    return static_cast<RenderObjectFlags>(static_cast<std::uint32_t>(left)
        | static_cast<std::uint32_t>(right));
}

struct RenderMaterial {
    MaterialId id{};
    Vec3 base_color{1.0, 1.0, 1.0};
    Vec3 emissive_color{};
    double emissive_strength{};
    double roughness{0.8};
    double metallic{};
};

struct GridOverlay {
    bool visible{};
    bool major_lines{true};
    double spacing{1.0};
    Vec3 origin{};
    Vec3 normal{0.0, 0.0, 1.0};
};

struct RenderSceneSettings {
    GridOverlay grid;
    bool show_debug_overlays{true};
    bool show_trajectories{};
    bool show_selection_outline{true};
};

// A logical scene is a view over a published frame. It keeps the publication
// by value so its CPU lease or future GPU resource lifetime remains valid while
// a backend consumes the scene. No per-body render objects are allocated here.
struct RenderScene {
    FramePublication frame;
    RenderCamera camera;
    RenderSceneSettings settings;
    std::span<const RenderMaterial> materials;
    float viewport_width{};
    float viewport_height{};

    std::span<const RenderBody> cpuBodies() const {
        if (!frame.cpu_snapshot) return {};
        return frame.cpu_snapshot->bodies;
    }

    bool hasCpuBodies() const {
        return frame.storage.kind == FrameStorageKind::CpuSnapshot
            && static_cast<bool>(frame.cpu_snapshot);
    }

    bool hasExternalGpuResource() const {
        return frame.storage.kind == FrameStorageKind::ExternalGpuResource
            && static_cast<bool>(frame.storage.external_gpu);
    }
};

inline RenderScene makeRenderScene(FramePublication frame,
                                   RenderCamera camera = {},
                                   RenderSceneSettings settings = {},
                                   std::span<const RenderMaterial> materials = {},
                                   float viewport_width = 0.0f,
                                   float viewport_height = 0.0f) {
    return {std::move(frame), camera, settings, materials, viewport_width, viewport_height};
}

}
