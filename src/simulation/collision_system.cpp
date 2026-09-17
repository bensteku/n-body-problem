#include "simulation/collision_system.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nbody {

namespace {

CollisionOutcome classify(double specific_energy, double damage_limit, double fragmentation_limit,
                          double accumulated_damage, const CollisionClassifierSettings& settings) {
    if (settings.force_fragmentation || specific_energy >= fragmentation_limit) return CollisionOutcome::Fragment;
    if (accumulated_damage >= 1.0) return CollisionOutcome::Fragment;
    if (settings.force_damage || specific_energy >= damage_limit) return CollisionOutcome::Damage;
    return CollisionOutcome::Bounce;
}

CollisionAssessment assessBody(const BodyState& body, double impact_energy,
                               double gravitational_constant, const CollisionClassifierSettings& settings) {
    const double specific_energy = body.mass > 0.0 ? impact_energy / body.mass : 0.0;
    const double density = std::max(body.material.density, 1e-12);
    const double strength_energy = settings.strength_scale
        * (body.material.compressive_strength + body.material.tensile_strength)
        / (2.0 * density);
    const double binding_energy = settings.binding_scale * gravitational_constant * body.mass
        / std::max(body.radius, 1e-12);
    const double damage_limit = settings.damage_threshold_scale
        * std::max(body.material.damage_threshold, strength_energy + binding_energy);
    const double fragmentation_limit = settings.fragmentation_threshold_scale
        * std::max(body.material.fragmentation_threshold, strength_energy + binding_energy);

    return {body.id, classify(specific_energy, damage_limit, fragmentation_limit, body.accumulated_damage, settings),
            specific_energy, damage_limit, fragmentation_limit};
}

void applyDeferredDamage(WorldState& world) {
    std::unordered_map<std::uint64_t, double> damage_by_body;
    for (const CollisionEvent& event : world.collisionEvents()) {
        const CollisionAssessment assessments[] = {event.first, event.second};
        for (const CollisionAssessment& assessment : assessments) {
            if (assessment.outcome != CollisionOutcome::Damage) continue;
            const double increment = std::clamp(
                assessment.specific_impact_energy / std::max(assessment.fragmentation_limit, 1e-12),
                0.0, 1.0);
            damage_by_body[assessment.body.value] += increment;
        }
    }

    for (BodyState& body : world.mutableBodies()) {
        const auto damage = damage_by_body.find(body.id.value);
        if (damage != damage_by_body.end()) {
            body.accumulated_damage = std::clamp(body.accumulated_damage + damage->second, 0.0, 1.0);
        }
    }
}

std::vector<BodyState> fragmentBody(const BodyState& body, std::size_t count,
                                    double specific_energy, Dimension dimension) {
    std::vector<BodyState> fragments;
    fragments.reserve(count);
    const double angle_step = 2.0 * 3.14159265358979323846 / static_cast<double>(count);
    const double spread = 0.1 * std::sqrt(std::max(0.0, specific_energy));
    const double radius_scale = dimension == Dimension::Two
        ? 1.0 / std::sqrt(static_cast<double>(count))
        : 1.0 / std::cbrt(static_cast<double>(count));

    for (std::size_t index = 0; index < count; ++index) {
        const double angle = angle_step * static_cast<double>(index);
        BodyState fragment = body;
        fragment.id = {};
        fragment.mass = body.mass / static_cast<double>(count);
        fragment.radius = body.radius * radius_scale;
        fragment.accumulated_damage = 0.0;
        fragment.position += Vec3{std::cos(angle), std::sin(angle), 0.0} * (body.radius * 0.25);
        fragment.velocity += Vec3{std::cos(angle), std::sin(angle), 0.0} * spread;
        if (dimension == Dimension::Two) {
            fragment.position.z = 0.0;
            fragment.velocity.z = 0.0;
        }
        fragments.push_back(fragment);
    }
    return fragments;
}

void applyDeferredFragmentation(WorldState& world, const CollisionSettings& settings) {
    if (settings.maximum_fragment_count == 0) return;

    std::unordered_set<std::uint64_t> fragment_ids;
    std::unordered_map<std::uint64_t, double> fragment_energy;
    for (const CollisionEvent& event : world.collisionEvents()) {
        if (event.first.outcome == CollisionOutcome::Fragment) {
            fragment_ids.insert(event.first.body.value);
            fragment_energy[event.first.body.value] = std::max(fragment_energy[event.first.body.value], event.first.specific_impact_energy);
        }
        if (event.second.outcome == CollisionOutcome::Fragment) {
            fragment_ids.insert(event.second.body.value);
            fragment_energy[event.second.body.value] = std::max(fragment_energy[event.second.body.value], event.second.specific_impact_energy);
        }
    }
    if (fragment_ids.empty()) return;

    const std::size_t minimum = std::max<std::size_t>(2, settings.minimum_fragments);
    const std::size_t maximum = std::max(minimum, settings.maximum_fragments);
    std::vector<BodyState> replacement;
    replacement.reserve(world.bodies().size() + fragment_ids.size() * minimum);
    std::size_t fragment_count_total = 0;
    for (const BodyState& body : world.bodies()) {
        if (!fragment_ids.contains(body.id.value)) {
            replacement.push_back(body);
            continue;
        }

        const std::size_t available = settings.maximum_fragment_count > fragment_count_total
            ? settings.maximum_fragment_count - fragment_count_total : 0;
        if (available < minimum) {
            replacement.push_back(body);
            continue;
        }
        const std::size_t range = maximum - minimum + 1;
        const std::size_t count = std::min(minimum + (body.id.value % range), std::min(maximum, available));
        std::vector<BodyState> fragments = fragmentBody(body, count, fragment_energy[body.id.value], world.dimension());
        replacement.insert(replacement.end(), fragments.begin(), fragments.end());
        fragment_count_total += count;
    }
    world.replaceBodies(std::move(replacement));
}

}

void CollisionSystem::resolveContacts(WorldState& world, const CollisionSettings& settings, double gravitational_constant) {
    world.clearCollisionEvents();
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
                const double normal_speed = std::max(0.0, -normal_velocity);
                const Vec3 tangential_velocity = relative_velocity - normal * normal_velocity;
                const double tangential_speed = tangential_velocity.length(dimension);
                const double reduced_mass = first_body.mass > 0.0 && second_body.mass > 0.0
                    ? (first_body.mass * second_body.mass) / (first_body.mass + second_body.mass)
                    : 0.0;
                const double impact_energy = 0.5 * reduced_mass
                    * (normal_speed * normal_speed
                       + settings.classifier.tangential_energy_scale * tangential_speed * tangential_speed);
                world.recordCollisionEvent({
                    first_body.id,
                    second_body.id,
                    normal_speed,
                    tangential_speed,
                    impact_energy,
                    assessBody(first_body, impact_energy, gravitational_constant, settings.classifier),
                    assessBody(second_body, impact_energy, gravitational_constant, settings.classifier)
                });
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

void CollisionSystem::applyDeferredOutcomes(WorldState& world, const CollisionSettings& settings) {
    applyDeferredDamage(world);
    applyDeferredFragmentation(world, settings);
}

}
