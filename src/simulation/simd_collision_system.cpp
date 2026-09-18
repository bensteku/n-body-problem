#include "simulation/simd_collision_system.hpp"

#include "simulation/scalar_collision_system.hpp"
#include "simulation/simd_capabilities.hpp"
#include "simulation/spatial_tree.hpp"

#include <algorithm>
#include <array>
#include <immintrin.h>
#include <memory>
#include <span>
#include <vector>

namespace nbody {

namespace {

struct SimdCollisionWorkspace {
    SimdCollisionWorkspace(Dimension dimension, const CollisionBroadPhaseSettings& settings)
        : tree(dimension, settings.leaf_capacity, settings.maximum_depth, settings.looseness),
          leaf_capacity(settings.leaf_capacity), maximum_depth(settings.maximum_depth),
          looseness(settings.looseness) {}

    SpatialTree tree;
    std::vector<std::pair<std::size_t, std::size_t>> contacts;
    std::size_t leaf_capacity;
    std::size_t maximum_depth;
    double looseness;
};

SimdCollisionWorkspace& collisionWorkspace(Dimension dimension,
    const CollisionBroadPhaseSettings& settings) {
    thread_local std::unique_ptr<SimdCollisionWorkspace> workspace;
    if (!workspace || workspace->tree.dimension() != dimension
        || workspace->leaf_capacity != settings.leaf_capacity
        || workspace->maximum_depth != settings.maximum_depth
        || workspace->looseness != settings.looseness) {
        workspace = std::make_unique<SimdCollisionWorkspace>(dimension, settings);
    }
    return *workspace;
}

void filterContactPairsAvx2(const WorldState& world, Dimension dimension,
    std::span<const std::pair<std::size_t, std::size_t>> candidates,
    std::vector<std::pair<std::size_t, std::size_t>>& contacts) {
    const std::size_t required_capacity = contacts.size() + candidates.size();
    if (contacts.capacity() < required_capacity) {
        const std::size_t doubled_capacity = contacts.capacity() == 0
            ? 64 : contacts.capacity() * 2;
        contacts.reserve(std::max(required_capacity, doubled_capacity));
    }
    const BodyStorage& storage = world.bodyStorage();
    const double* position_x = storage.positionX().data();
    const double* position_y = storage.positionY().data();
    const double* position_z = storage.positionZ().data();
    const double* radii = storage.radii().data();
    std::size_t index = 0;
    for (; index + 4 <= candidates.size(); index += 4) {
        const auto& first0 = candidates[index];
        const auto& first1 = candidates[index + 1];
        const auto& first2 = candidates[index + 2];
        const auto& first3 = candidates[index + 3];
        const __m256i first_indices = _mm256_set_epi64x(
            static_cast<long long>(first3.first), static_cast<long long>(first2.first),
            static_cast<long long>(first1.first), static_cast<long long>(first0.first));
        const __m256i second_indices = _mm256_set_epi64x(
            static_cast<long long>(first3.second), static_cast<long long>(first2.second),
            static_cast<long long>(first1.second), static_cast<long long>(first0.second));
        const __m256d dx = _mm256_sub_pd(
            _mm256_i64gather_pd(position_x, second_indices, 8),
            _mm256_i64gather_pd(position_x, first_indices, 8));
        const __m256d dy = _mm256_sub_pd(
            _mm256_i64gather_pd(position_y, second_indices, 8),
            _mm256_i64gather_pd(position_y, first_indices, 8));
        __m256d distance_squared = _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy));
        if (dimension == Dimension::Three) {
            const __m256d dz = _mm256_sub_pd(
                _mm256_i64gather_pd(position_z, second_indices, 8),
                _mm256_i64gather_pd(position_z, first_indices, 8));
            distance_squared = _mm256_add_pd(distance_squared, _mm256_mul_pd(dz, dz));
        }
        const __m256d combined_radius = _mm256_add_pd(
            _mm256_i64gather_pd(radii, first_indices, 8),
            _mm256_i64gather_pd(radii, second_indices, 8));
        const __m256d radius_squared = _mm256_mul_pd(combined_radius, combined_radius);
        const int mask = _mm256_movemask_pd(_mm256_cmp_pd(distance_squared, radius_squared, _CMP_LE_OQ));
        if (mask & 1) contacts.push_back(first0);
        if (mask & 2) contacts.push_back(first1);
        if (mask & 4) contacts.push_back(first2);
        if (mask & 8) contacts.push_back(first3);
    }
    for (; index < candidates.size(); ++index) {
        const auto [first, second] = candidates[index];
        const Vec3 displacement = world.body(second).position - world.body(first).position;
        const double combined_radius = world.body(first).radius + world.body(second).radius;
        if (displacement.lengthSquared(dimension) <= combined_radius * combined_radius) {
            contacts.push_back(candidates[index]);
        }
    }
}

}

void SimdCollisionSystem::resolveContacts(WorldState& world,
    const CollisionSettings& settings, double gravitational_constant) {
    static const bool avx2_available = detectSimdCapabilities().avx2;
    if (!avx2_available || settings.model != CollisionModel::HardBody
        || world.activeCollisionModel() != CollisionModel::HardBody) {
        ScalarCollisionSystem::resolveContacts(world, settings, gravitational_constant);
        return;
    }

    SimdCollisionWorkspace& workspace = collisionWorkspace(world.dimension(), settings.broad_phase);
    auto& contacts = workspace.contacts;
    contacts.clear();

    // Collision response mutates the authoritative Vec3 positions. Keep the
    // component arrays coherent before the AVX2 gather phase.
    world.bodyStorage().synchronizePositionComponents();
    if (settings.broad_phase.spatial_tree_enabled) {
        workspace.tree.rebuild(world.bodyStorage());
        workspace.tree.forEachPotentialBodyPairBatch(world.bodyStorage(),
            [&](std::span<const std::pair<std::size_t, std::size_t>> candidates) {
                filterContactPairsAvx2(world, world.dimension(), candidates, contacts);
            });
    } else {
        constexpr std::size_t batch_capacity = 64;
        std::array<std::pair<std::size_t, std::size_t>, batch_capacity> batch{};
        std::size_t batch_size = 0;
        const auto flush = [&]() {
            filterContactPairsAvx2(world, world.dimension(),
                                   std::span<const std::pair<std::size_t, std::size_t>>(
                                       batch.data(), batch_size), contacts);
            batch_size = 0;
        };
        for (std::size_t first = 0; first < world.bodyCount(); ++first) {
            for (std::size_t second = first + 1; second < world.bodyCount(); ++second) {
                batch[batch_size++] = {first, second};
                if (batch_size == batch_capacity) flush();
            }
        }
        if (batch_size != 0) flush();
    }
    ScalarCollisionSystem::resolveContactsForPairs(world, settings, gravitational_constant,
                                                   contacts);
}

void SimdCollisionSystem::applyDeferredOutcomes(WorldState& world,
    const CollisionSettings& settings) {
    ScalarCollisionSystem::applyDeferredOutcomes(world, settings);
}

}
