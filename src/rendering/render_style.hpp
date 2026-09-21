#pragma once

#include "rendering/render_scene.hpp"

namespace nbody::rendering {

struct RenderStyle {
    Vec3 color{0.35, 0.85, 1.0};
    float alpha{0.85f};
    bool emissive{};
};

inline RenderStyle styleFor(const RenderBody& body, bool selected, bool focused) {
    if (selected) return {{1.0, 0.85, 0.15}, 1.0f, true};
    if (body.kind == BodyKind::Star) return {{1.0, 0.75, 0.25}, 1.0f, true};
    if (focused) return {{0.45, 1.0, 0.95}, 1.0f, false};
    if (body.is_static) return {{0.75, 0.50, 1.0}, 0.85f, false};
    return {{0.35, 0.85, 1.0}, 0.85f, false};
}

}
