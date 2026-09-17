#pragma once

#include "boundary.hpp"
#include "world_state.hpp"

namespace nbody {

class BoundarySystem {
public:
    static void resolve(WorldState& world, const BoundarySettings& settings);
};

}
