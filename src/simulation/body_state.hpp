#pragma once

#include "body_id.hpp"
#include "body_kind.hpp"
#include "material_properties.hpp"
#include "vec3.hpp"

namespace nbody {

struct BodyState {
    BodyId id;
    Vec3 position;
    Vec3 velocity;
    double mass{1.0};
    double radius{1.0};
    bool is_static{false};
    MaterialProperties material;
    double accumulated_damage{0.0};
    BodyKind kind{BodyKind::Ordinary};
};

}
