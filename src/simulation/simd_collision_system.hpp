#pragma once

#include "collision.hpp"
#include "scalar_collision_system.hpp"
#include "world_state.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace nbody {

class SimdCollisionWorkspace {
public:
    void prepare(Dimension dimension, const CollisionBroadPhaseSettings& settings);
    SpatialTree& tree();
    std::vector<std::pair<std::size_t, std::size_t>>& contacts();

private:
    std::unique_ptr<SpatialTree> tree_;
    std::vector<std::pair<std::size_t, std::size_t>> contacts_;
    std::size_t leaf_capacity_{};
    std::size_t maximum_depth_{};
    double looseness_{};
};

// SIMD collision phase boundary. Contact outcomes and deferred mutations remain
// identical to the scalar reference until vectorized contact kernels replace
// the narrow-phase implementation.
class SimdCollisionSystem {
public:
    static void resolveContacts(WorldState& world, const CollisionSettings& settings,
                                double gravitational_constant);
    static void resolveContacts(WorldState& world, const CollisionSettings& settings,
                                double gravitational_constant, SimdCollisionWorkspace& simd_workspace,
                                ScalarCollisionWorkspace& scalar_workspace);
    static void applyDeferredOutcomes(WorldState& world, const CollisionSettings& settings);
};

}
