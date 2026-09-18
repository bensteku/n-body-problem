#pragma once

#include "collision.hpp"
#include "spatial_tree.hpp"
#include "world_state.hpp"

#include <memory>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace nbody {

class ScalarCollisionWorkspace {
public:
    void prepare(Dimension dimension, const CollisionBroadPhaseSettings& settings);
    SpatialTree& tree();
    std::vector<std::pair<std::size_t, std::size_t>>& candidatePairs();
    std::vector<std::uint8_t>& pendingRemoval();

private:
    std::unique_ptr<SpatialTree> tree_;
    std::vector<std::pair<std::size_t, std::size_t>> candidate_pairs_;
    std::vector<std::uint8_t> pending_removal_;
    std::size_t leaf_capacity_{};
    std::size_t maximum_depth_{};
    double looseness_{};
};

// Scalar reference collision phase. Backend-specific collision implementations
// should conform to this phase boundary without entering the contact hot loop
// through virtual dispatch.
class ScalarCollisionSystem {
public:
    static void resolveContacts(WorldState& world, const CollisionSettings& settings,
                                double gravitational_constant);
    static void resolveContacts(WorldState& world, const CollisionSettings& settings,
                                double gravitational_constant, ScalarCollisionWorkspace& workspace);
    static void resolveContactsForPairs(WorldState& world, const CollisionSettings& settings,
                                        double gravitational_constant,
                                        std::span<const std::pair<std::size_t, std::size_t>> pairs);
    static void resolveContactsForPairs(WorldState& world, const CollisionSettings& settings,
                                        double gravitational_constant,
                                        std::span<const std::pair<std::size_t, std::size_t>> pairs,
                                        ScalarCollisionWorkspace& workspace);
    static void applyDeferredOutcomes(WorldState& world, const CollisionSettings& settings);

private:
    static void resolveContactsInternal(WorldState& world, const CollisionSettings& settings,
                                        double gravitational_constant,
                                        ScalarCollisionWorkspace& workspace,
                                        const std::span<const std::pair<std::size_t, std::size_t>>* pairs);
};

}
