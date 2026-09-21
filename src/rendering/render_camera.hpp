#pragma once

#include "rendering/render_scene.hpp"

#include <algorithm>
#include <array>

namespace nbody::rendering {

struct RenderViewport {
    float width{1.0f};
    float height{1.0f};
};

inline std::array<float, 2> worldToClip(const RenderCamera& camera,
                                        const RenderViewport& viewport,
                                        double x, double y) {
    const float width = viewport.width > 0.0f ? viewport.width : 1.0f;
    const float height = viewport.height > 0.0f ? viewport.height : 1.0f;
    const float zoom = static_cast<float>(camera.zoom);
    const float screen_x = width * 0.5f
        + (static_cast<float>(x) - static_cast<float>(camera.position.x)) * zoom;
    const float screen_y = height * 0.5f
        - (static_cast<float>(y) - static_cast<float>(camera.position.y)) * zoom;
    return {2.0f * screen_x / width - 1.0f, 2.0f * screen_y / height - 1.0f};
}

inline std::array<float, 2> worldRadiusToClip(const RenderCamera& camera,
                                              const RenderViewport& viewport,
                                              double radius) {
    const float width = viewport.width > 0.0f ? viewport.width : 1.0f;
    const float height = viewport.height > 0.0f ? viewport.height : 1.0f;
    const float pixels = std::max(0.75f, static_cast<float>(radius) * static_cast<float>(camera.zoom));
    return {2.0f * pixels / width, 2.0f * pixels / height};
}

}
