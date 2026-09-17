#include "simulation/solvers/simd_barnes_hut_kernel.hpp"

#include <immintrin.h>
#include <algorithm>
#include <cmath>

namespace nbody {

namespace {

bool containsPoint(const SpatialBounds& bounds, const Vec3& position, Dimension dimension) {
    return position.x >= bounds.minimum.x && position.x <= bounds.maximum.x
        && position.y >= bounds.minimum.y && position.y <= bounds.maximum.y
        && (dimension == Dimension::Two
            || (position.z >= bounds.minimum.z && position.z <= bounds.maximum.z));
}

double extent(const SpatialBounds& bounds, Dimension dimension) {
    return std::max({bounds.maximum.x - bounds.minimum.x,
                     bounds.maximum.y - bounds.minimum.y,
                     dimension == Dimension::Three ? bounds.maximum.z - bounds.minimum.z : 0.0});
}

Vec3 accelerationFromNode(const BarnesHutTree& tree, std::size_t node_index, std::size_t target,
                          const Vec3& target_position, const WorldState& world, double gravitational_constant,
                          double softening_squared, double opening_angle,
                          std::vector<std::size_t>& traversal_stack) {
    const Dimension dimension = world.dimension();
    traversal_stack.clear();
    traversal_stack.push_back(node_index);
    Vec3 acceleration{};
    while (!traversal_stack.empty()) {
        const std::size_t current_node = traversal_stack.back();
        traversal_stack.pop_back();
        const BarnesHutTree::NodeView node = tree.node(current_node);
        if (node.mass <= 0.0) continue;
        const double distance_squared = (node.center_of_mass - target_position).lengthSquared(dimension)
            + softening_squared;
        const double distance = std::sqrt(distance_squared);
        const bool target_inside = containsPoint(node.bounds, target_position, dimension);
        if (node.child_count != 0 && !target_inside && distance > 0.0
            && extent(node.bounds, dimension) / distance < opening_angle) {
            acceleration += (node.center_of_mass - target_position)
                * (gravitational_constant * node.mass / (distance_squared * distance));
            continue;
        }

        std::size_t source = 0;
        if (node.packed_x.size() >= 4) {
            __m256d acceleration_x = _mm256_setzero_pd();
            __m256d acceleration_y = _mm256_setzero_pd();
            __m256d acceleration_z = _mm256_setzero_pd();
            const __m256d target_x = _mm256_set1_pd(target_position.x);
            const __m256d target_y = _mm256_set1_pd(target_position.y);
            const __m256d target_z = _mm256_set1_pd(target_position.z);
            const __m256d softening = _mm256_set1_pd(softening_squared);
            const __m256d minimum_distance = _mm256_set1_pd(1e-30);
            const __m256d gravity = _mm256_set1_pd(gravitational_constant);
            for (; source + 3 < node.packed_x.size(); source += 4) {
                const __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(node.packed_x.data() + source), target_x);
                const __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(node.packed_y.data() + source), target_y);
                const __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(node.packed_z.data() + source), target_z);
                __m256d distance_squared = _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy));
                if (dimension == Dimension::Three) distance_squared = _mm256_add_pd(distance_squared, _mm256_mul_pd(dz, dz));
                distance_squared = _mm256_max_pd(_mm256_add_pd(distance_squared, softening), minimum_distance);
                const __m256d distance = _mm256_sqrt_pd(distance_squared);
                const __m256d scale = _mm256_mul_pd(gravity,
                    _mm256_div_pd(_mm256_loadu_pd(node.packed_masses.data() + source),
                                  _mm256_mul_pd(distance_squared, distance)));
                acceleration_x = _mm256_add_pd(acceleration_x, _mm256_mul_pd(dx, scale));
                acceleration_y = _mm256_add_pd(acceleration_y, _mm256_mul_pd(dy, scale));
                if (dimension == Dimension::Three) acceleration_z = _mm256_add_pd(acceleration_z, _mm256_mul_pd(dz, scale));
            }
            alignas(32) double lanes_x[4];
            alignas(32) double lanes_y[4];
            alignas(32) double lanes_z[4];
            _mm256_store_pd(lanes_x, acceleration_x);
            _mm256_store_pd(lanes_y, acceleration_y);
            _mm256_store_pd(lanes_z, acceleration_z);
            for (double lane : lanes_x) acceleration.x += lane;
            for (double lane : lanes_y) acceleration.y += lane;
            for (double lane : lanes_z) acceleration.z += lane;
        }
        for (; source < node.packed_x.size(); ++source) {
            const std::size_t body_index = node.packed_indices[source];
            if (body_index == target) continue;
            const Vec3 displacement{node.packed_x[source] - target_position.x,
                                    node.packed_y[source] - target_position.y,
                                    node.packed_z[source] - target_position.z};
            const double body_distance_squared = displacement.lengthSquared(dimension) + softening_squared;
            if (body_distance_squared == 0.0) continue;
            const double body_distance = std::sqrt(body_distance_squared);
            acceleration += displacement * (gravitational_constant * node.packed_masses[source]
                / (body_distance_squared * body_distance));
        }
        for (std::size_t child = node.child_count; child > 0; --child) {
            traversal_stack.push_back(node.children[child - 1]);
        }
    }
    return acceleration;
}

}

void calculateAvx2BarnesHutAccelerations(const WorldState& world, const SimulationParameters& parameters,
                                         const BarnesHutTree& tree, std::vector<Vec3>& output) {
    output.assign(world.bodyCount(), Vec3{});
    const double opening_angle = std::clamp(parameters.solver.barnes_hut.opening_angle, 0.0, 10.0);
    const double softening_squared = parameters.softening_length * parameters.softening_length;
    std::vector<std::size_t> traversal_stack;
    traversal_stack.reserve(tree.nodeCount());
    for (std::size_t target = 0; target < world.bodyCount(); ++target) {
        if (!world.body(target).is_static()) {
            output[target] = accelerationFromNode(tree, 0, target, world.body(target).position, world,
                                                  parameters.gravitational_constant,
                                                  softening_squared, opening_angle, traversal_stack);
        }
    }
}

}
