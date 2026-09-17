#pragma once

#include "vec3.hpp"

namespace nbody {

enum class BoundaryResponse {
    Reflective,
    RepulsivePotential
};

struct BoundarySettings {
    bool enabled{false};
    BoundaryResponse response{BoundaryResponse::Reflective};
    Vec3 minimum{-100.0, -100.0, -100.0};
    Vec3 maximum{100.0, 100.0, 100.0};
    double restitution{1.0};
    double repulsion_strength{0.0};
    double repulsion_falloff{1.0};
};

}
