#pragma once

#include "collision.hpp"
#include "world_state.hpp"

namespace nbody {

class CollisionSystem {
public:
    static void resolveContacts(WorldState& world, const CollisionSettings& settings, double gravitational_constant);
    static void applyDeferredOutcomes(WorldState& world, const CollisionSettings& settings);
};

}
