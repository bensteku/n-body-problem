#pragma once

#include "collision.hpp"
#include "world_state.hpp"

#include <span>
#include <utility>
#include <vector>

namespace nbody {

// Scalar reference collision phase. Backend-specific collision implementations
// should conform to this phase boundary without entering the contact hot loop
// through virtual dispatch.
class ScalarCollisionSystem {
public:
    static void resolveContacts(WorldState& world, const CollisionSettings& settings,
                                double gravitational_constant);
    static void resolveContactsForPairs(WorldState& world, const CollisionSettings& settings,
                                        double gravitational_constant,
                                        std::span<const std::pair<std::size_t, std::size_t>> pairs);
    static void applyDeferredOutcomes(WorldState& world, const CollisionSettings& settings);

private:
    static void resolveContactsInternal(WorldState& world, const CollisionSettings& settings,
                                        double gravitational_constant,
                                        const std::span<const std::pair<std::size_t, std::size_t>>* pairs);
};

}
