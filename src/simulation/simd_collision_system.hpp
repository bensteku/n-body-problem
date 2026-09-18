#pragma once

#include "collision.hpp"
#include "world_state.hpp"

namespace nbody {

// SIMD collision phase boundary. Contact outcomes and deferred mutations remain
// identical to the scalar reference until vectorized contact kernels replace
// the narrow-phase implementation.
class SimdCollisionSystem {
public:
    static void resolveContacts(WorldState& world, const CollisionSettings& settings,
                                double gravitational_constant);
    static void applyDeferredOutcomes(WorldState& world, const CollisionSettings& settings);
};

}
