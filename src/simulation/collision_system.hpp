#pragma once

#include "collision.hpp"
#include "world_state.hpp"

namespace nbody {

class CollisionSystem {
public:
    static void resolve(WorldState& world, const CollisionSettings& settings);
};

}
