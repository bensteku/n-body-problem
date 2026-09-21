#pragma once

#include "rendering/render_scene.hpp"

#include <algorithm>

namespace nbody::rendering {

struct CameraZoomPolicy {
    float minimum{0.25f};
    float maximum{1.0e6f};
};

inline CameraZoomPolicy zoomPolicy(bool physical_scale, double world_kilometers_per_unit,
                                   float viewport_span) {
    if (!physical_scale || world_kilometers_per_unit <= 0.0) return {};
    constexpr double moon_diameter_km = 2.0 * 1737.4;
    constexpr double margin_factor = 1.3;
    const double moon_span_world = moon_diameter_km * margin_factor / world_kilometers_per_unit;
    return {0.25f, std::max(0.25f, static_cast<float>(viewport_span / (4.0 * moon_span_world)))};
}

}
