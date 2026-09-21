#pragma once

#include "rendering/render_camera.hpp"

#include <cmath>
#include <limits>
#include <optional>

namespace nbody::rendering {

struct PickingQuery {
    Dimension dimension{Dimension::Two};
    float screen_x{};
    float screen_y{};
    RenderViewport viewport{};
    float tolerance_pixels{6.0f};
};

struct PickingHit {
    BodyId body;
    float distance_pixels{};
};

struct PickingResult {
    std::optional<PickingHit> hit;
    bool supported{true};

    explicit operator bool() const { return hit.has_value(); }
};

// The request/result pair is dimension-neutral. The 2D implementation below
// is deliberately CPU-side; a future 3D backend can replace it with ray or
// ID-buffer picking without changing frontend selection commands.
inline PickingResult pickBody2D(const RenderScene& scene, const PickingQuery& query) {
    if (!scene.hasCpuBodies()) return {};
    const RenderViewport viewport = query.viewport;
    std::optional<PickingHit> result;
    float nearest = std::numeric_limits<float>::max();
    for (const RenderBody& body : scene.cpuBodies()) {
        const auto clip = worldToClip(scene.camera, viewport, body.position.x, body.position.y);
        const float body_x = (clip[0] + 1.0f) * viewport.width * 0.5f;
        const float body_y = (clip[1] + 1.0f) * viewport.height * 0.5f;
        const float radius = std::max(query.tolerance_pixels,
            static_cast<float>(body.radius) * static_cast<float>(scene.camera.zoom));
        const float dx = query.screen_x - body_x;
        const float dy = query.screen_y - body_y;
        const float distance_squared = dx * dx + dy * dy;
        if (distance_squared <= radius * radius
            && (distance_squared < nearest
                || (distance_squared == nearest && result
                    && body.id.value < result->body.value))) {
            nearest = distance_squared;
            result = PickingHit{body.id, std::sqrt(distance_squared)};
        }
    }
    return {result, true};
}

inline PickingResult pickBody(const RenderScene& scene, const PickingQuery& query) {
    if (query.dimension == Dimension::Two) return pickBody2D(scene, query);
    return {std::nullopt, false};
}

inline std::optional<BodyId> pickBody(const RenderScene& scene,
                                      float screen_x, float screen_y,
                                      float tolerance_pixels = 6.0f) {
    const PickingQuery query{Dimension::Two, screen_x, screen_y,
                             {scene.viewport_width, scene.viewport_height},
                             tolerance_pixels};
    const PickingResult result = pickBody(scene, query);
    if (!result.hit) return std::nullopt;
    return result.hit->body;
}

}
