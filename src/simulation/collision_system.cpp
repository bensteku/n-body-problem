#include "simulation/collision_system.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nbody {

void CollisionSystem::resolve(WorldState& world, const CollisionSettings& settings) {
    switch (settings.model) {
    case CollisionModel::Transparent:
        // Transparent bodies deliberately receive no contact impulse or overlap correction.
        return;
    case CollisionModel::HardBody: {
        auto bodies = world.mutableBodies();
        const Dimension dimension = world.dimension();
        const double restitution = std::clamp(settings.restitution, 0.0, 1.0);

        for (std::size_t first = 0; first < bodies.size(); ++first) {
            for (std::size_t second = first + 1; second < bodies.size(); ++second) {
                BodyState& first_body = bodies[first];
                BodyState& second_body = bodies[second];
                const Vec3 displacement = second_body.position - first_body.position;
                const double distance_squared = displacement.lengthSquared(dimension);
                const double combined_radius = first_body.radius + second_body.radius;
                if (distance_squared > combined_radius * combined_radius) continue;

                const double distance = std::sqrt(distance_squared);
                const Vec3 normal = distance > 1e-12
                    ? displacement * (1.0 / distance)
                    : Vec3{1.0, 0.0, 0.0};
                const double penetration = std::max(0.0, combined_radius - distance);
                const double first_inverse_mass = first_body.is_static || first_body.mass <= 0.0 ? 0.0 : 1.0 / first_body.mass;
                const double second_inverse_mass = second_body.is_static || second_body.mass <= 0.0 ? 0.0 : 1.0 / second_body.mass;
                const double inverse_mass_sum = first_inverse_mass + second_inverse_mass;

                if (inverse_mass_sum > 0.0 && penetration > 0.0) {
                    const Vec3 correction = normal * (penetration / inverse_mass_sum);
                    first_body.position -= correction * first_inverse_mass;
                    second_body.position += correction * second_inverse_mass;
                    if (dimension == Dimension::Two) {
                        first_body.position.z = 0.0;
                        second_body.position.z = 0.0;
                    }
                }

                if (inverse_mass_sum == 0.0) continue;
                const Vec3 relative_velocity = second_body.velocity - first_body.velocity;
                const double normal_velocity = relative_velocity.dot(normal, dimension);
                if (normal_velocity >= 0.0) continue;

                const double material_restitution = std::min(first_body.material.restitution, second_body.material.restitution);
                const double effective_restitution = std::clamp(std::min(restitution, material_restitution), 0.0, 1.0);
                const double impulse_magnitude = -(1.0 + effective_restitution) * normal_velocity / inverse_mass_sum;
                const Vec3 impulse = normal * impulse_magnitude;
                if (!first_body.is_static) first_body.velocity -= impulse * first_inverse_mass;
                if (!second_body.is_static) second_body.velocity += impulse * second_inverse_mass;
                if (dimension == Dimension::Two) {
                    first_body.velocity.z = 0.0;
                    second_body.velocity.z = 0.0;
                }
            }
        }
        return;
    }
    case CollisionModel::HeuristicFragmentation:
    case CollisionModel::HeuristicAbsorption:
        throw std::logic_error("Requested collision model is not implemented yet");
    }
    throw std::logic_error("Unknown collision model");
}

}
