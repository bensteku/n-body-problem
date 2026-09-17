#pragma once

#include "body_id.hpp"
#include "vec3.hpp"

namespace nbody {

struct BodyState {
    BodyId id;
    Vec3 position;
    Vec3 velocity;
    double mass{1.0};
    double radius{1.0};
    bool is_static{false};
};

}
