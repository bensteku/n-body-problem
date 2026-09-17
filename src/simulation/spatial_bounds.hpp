#pragma once

#include "vec3.hpp"

namespace nbody {

struct SpatialBounds {
    Vec3 minimum{};
    Vec3 maximum{};
};

}
