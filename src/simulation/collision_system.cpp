#include "simulation/collision_system.hpp"
#include "simulation/spatial_tree.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nbody {

namespace {

struct CollisionWorkspace {
    CollisionWorkspace(Dimension dimension, const CollisionBroadPhaseSettings& settings)
        : tree(dimension, settings.leaf_capacity, settings.maximum_depth, settings.looseness),
          leaf_capacity(settings.leaf_capacity), maximum_depth(settings.maximum_depth),
          looseness(settings.looseness) {}

    SpatialTree tree;
    std::vector<std::pair<std::size_t, std::size_t>> candidate_pairs;
    std::size_t leaf_capacity;
    std::size_t maximum_depth;
    double looseness;
};

struct AbsorptionDecision {
    BodyId survivor;
    BodyId removed;
};

bool approximatelyEqual(double left, double right, double relative_tolerance) {
    return std::abs(left - right) <= relative_tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

template <typename FirstBody, typename SecondBody>
AbsorptionDecision chooseAbsorption(const FirstBody& first, const SecondBody& second,
    const CollisionSettings& settings) {
    const bool first_black_hole = isBlackHole(first.kind);
    const bool second_black_hole = isBlackHole(second.kind);
    if (!isAbsorber(first.kind) && !isAbsorber(second.kind)) return {};

    if (first_black_hole != second_black_hole) {
        const BodyId survivor = first_black_hole ? first.id : second.id;
        const BodyId removed = first_black_hole ? second.id : first.id;
        return {survivor, removed};
    }

    if (isAbsorber(first.kind) && isAbsorber(second.kind)) {
        if (!approximatelyEqual(first.mass, second.mass, settings.absorber_mass_tolerance)) {
            return first.mass > second.mass ? AbsorptionDecision{first.id, second.id}
                                            : AbsorptionDecision{second.id, first.id};
        }
        if (!approximatelyEqual(first.radius, second.radius, settings.absorber_size_tolerance)) {
            return first.radius > second.radius ? AbsorptionDecision{first.id, second.id}
                                                 : AbsorptionDecision{second.id, first.id};
        }
        return {first.id, second.id};
    }

    const auto& absorber = isAbsorber(first.kind) ? first : second;
    const auto& solid = isAbsorber(first.kind) ? second : first;
    const bool solid_can_be_absorbed = solid.mass < settings.solid_absorption_mass_ratio * absorber.mass
        && solid.radius < settings.solid_absorption_size_ratio * absorber.radius;
    return solid_can_be_absorbed ? AbsorptionDecision{absorber.id, solid.id}
                                 : AbsorptionDecision{solid.id, absorber.id};
}

CollisionWorkspace& collisionWorkspace(Dimension dimension, const CollisionBroadPhaseSettings& settings) {
    thread_local std::unique_ptr<CollisionWorkspace> workspace;
    if (!workspace || workspace->tree.dimension() != dimension
        || workspace->leaf_capacity != settings.leaf_capacity
        || workspace->maximum_depth != settings.maximum_depth
        || workspace->looseness != settings.looseness) {
        workspace = std::make_unique<CollisionWorkspace>(dimension, settings);
    }
    return *workspace;
}

CollisionOutcome classify(double specific_energy, double damage_limit, double fragmentation_limit,
                          double accumulated_damage, const CollisionClassifierSettings& settings) {
    const bool fragmentation_active = settings.fragmentation_enabled || settings.force_fragmentation;
    if (fragmentation_active && (settings.force_fragmentation || specific_energy >= fragmentation_limit)) {
        return CollisionOutcome::Fragment;
    }
    if (fragmentation_active && accumulated_damage >= 1.0) return CollisionOutcome::Fragment;
    if (settings.force_damage || specific_energy >= damage_limit) return CollisionOutcome::Damage;
    return CollisionOutcome::Bounce;
}

std::vector<BodyId> overlappingBodies(const WorldState& world) {
    std::unordered_set<std::uint64_t> ids;
    const Dimension dimension = world.dimension();
    for (std::size_t first = 0; first < world.bodyCount(); ++first) {
        const ConstBodyView first_body = world.body(first);
        for (std::size_t second = first + 1; second < world.bodyCount(); ++second) {
            const ConstBodyView second_body = world.body(second);
            const Vec3 displacement = second_body.position - first_body.position;
            const double combined_radius = first_body.radius + second_body.radius;
            if (displacement.lengthSquared(dimension) < combined_radius * combined_radius) {
                ids.insert(first_body.id.value);
                ids.insert(second_body.id.value);
            }
        }
    }
    std::vector<BodyId> result;
    result.reserve(ids.size());
    for (const std::uint64_t id : ids) result.push_back({id});
    std::sort(result.begin(), result.end(), [](BodyId left, BodyId right) {
        return left.value < right.value;
    });
    return result;
}

bool untangleBodies(WorldState& world, std::size_t maximum_passes, std::size_t& passes,
    std::vector<BodyId>& unresolved_bodies) {
    passes = 0;
    for (; passes < maximum_passes; ++passes) {
        bool moved_body = false;
        const Dimension dimension = world.dimension();
        for (std::size_t first = 0; first < world.bodyCount(); ++first) {
            for (std::size_t second = first + 1; second < world.bodyCount(); ++second) {
                MutableBodyView first_body = world.mutableBody(first);
                MutableBodyView second_body = world.mutableBody(second);
                const Vec3 displacement = second_body.position - first_body.position;
                const double combined_radius = first_body.radius + second_body.radius;
                const double distance_squared = displacement.lengthSquared(dimension);
                if (distance_squared >= combined_radius * combined_radius) continue;

                moved_body = true;
                const double distance = std::sqrt(distance_squared);
                const Vec3 normal = distance > 1e-12
                    ? displacement * (1.0 / distance)
                    : Vec3{1.0, 0.0, 0.0};
                const double first_inverse_mass = first_body.is_static() || first_body.mass <= 0.0
                    ? 0.0 : 1.0 / first_body.mass;
                const double second_inverse_mass = second_body.is_static() || second_body.mass <= 0.0
                    ? 0.0 : 1.0 / second_body.mass;
                const double inverse_mass_sum = first_inverse_mass + second_inverse_mass;
                if (inverse_mass_sum == 0.0) continue;

                const double penetration = combined_radius - distance;
                const Vec3 correction = normal * (penetration / inverse_mass_sum);
                first_body.position -= correction * first_inverse_mass;
                second_body.position += correction * second_inverse_mass;
                if (dimension == Dimension::Two) {
                    first_body.position.z = 0.0;
                    second_body.position.z = 0.0;
                }
            }
        }

        unresolved_bodies = overlappingBodies(world);
        if (unresolved_bodies.empty()) return true;
        if (!moved_body) break;
    }

    unresolved_bodies = overlappingBodies(world);
    return unresolved_bodies.empty();
}

template <typename Body>
CollisionAssessment assessBody(const Body& body, double impact_energy,
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

    const CollisionOutcome outcome = isAbsorber(body.kind)
        ? CollisionOutcome::Bounce
        : classify(specific_energy, damage_limit, fragmentation_limit, body.accumulated_damage, settings);
    return {body.id, outcome,
            specific_energy, damage_limit, fragmentation_limit};
}

void applyDeferredAbsorption(WorldState& world, const CollisionSettings& settings) {
    if (!settings.classifier.absorption_enabled) return;

    struct Request { BodyId survivor; BodyId removed; bool destroys_absorber; };
    std::vector<Request> requests;
    requests.reserve(world.collisionEvents().size());
    std::unordered_set<std::uint64_t> removed_ids;
    for (const CollisionEvent& event : world.collisionEvents()) {
        if (!event.absorption_survivor.isValid() || !event.absorption_removed.isValid()) continue;
        if (removed_ids.contains(event.absorption_survivor.value)
            || removed_ids.contains(event.absorption_removed.value)) continue;
        requests.push_back({event.absorption_survivor, event.absorption_removed, event.absorber_destroyed});
        removed_ids.insert(event.absorption_removed.value);
    }
    if (requests.empty()) return;

    std::unordered_map<std::uint64_t, std::size_t> index_by_id;
    index_by_id.reserve(world.bodyCount());
    for (std::size_t index = 0; index < world.bodyCount(); ++index) {
        index_by_id.emplace(world.body(index).id.value, index);
    }
    const auto find_index = [&index_by_id, body_count = world.bodyCount()](BodyId id) -> std::size_t {
        const auto found = index_by_id.find(id.value);
        return found == index_by_id.end() ? body_count : found->second;
    };

    for (const Request request : requests) {
        const std::size_t survivor_index = find_index(request.survivor);
        const std::size_t removed_index = find_index(request.removed);
        if (survivor_index == world.bodyCount() || removed_index == world.bodyCount()) continue;
        MutableBodyView survivor = world.mutableBody(survivor_index);
        const ConstBodyView removed = world.body(removed_index);
        if (request.destroys_absorber) {
            const double combined_mass = survivor.mass + removed.mass;
            if (combined_mass > 0.0 && !survivor.is_static()) {
                const Vec3 total_momentum = survivor.velocity * survivor.mass + removed.velocity * removed.mass;
                survivor.velocity = total_momentum * (1.0 / combined_mass);
            }
        } else {
            const double combined_mass = survivor.mass + removed.mass;
            if (combined_mass > 0.0) {
                const Vec3 total_momentum = survivor.velocity * survivor.mass + removed.velocity * removed.mass;
                const Vec3 center_of_mass = (survivor.position * survivor.mass + removed.position * removed.mass)
                    * (1.0 / combined_mass);
                survivor.mass = combined_mass;
                if (!survivor.is_static()) {
                    survivor.velocity = total_momentum * (1.0 / combined_mass);
                    survivor.position = center_of_mass;
                }
            }
            const double combined_volume = survivor.radius * survivor.radius * survivor.radius
                + removed.radius * removed.radius * removed.radius;
            survivor.radius = std::cbrt(std::max(0.0, combined_volume));
        }
        if (!request.destroys_absorber) survivor.accumulated_damage = 0.0;
    }

    std::vector<BodyState> replacement;
    replacement.reserve(world.bodyCount() - removed_ids.size());
    for (std::size_t index = 0; index < world.bodyCount(); ++index) {
        const BodyState body = world.body(index).snapshot();
        if (!removed_ids.contains(body.id.value)) replacement.push_back(body);
    }
    world.replaceBodies(std::move(replacement));
}

void applyDeferredDamage(WorldState& world) {
    std::unordered_map<std::uint64_t, double> damage_by_body;
    for (const CollisionEvent& event : world.collisionEvents()) {
        const CollisionAssessment assessments[] = {event.first, event.second};
        for (const CollisionAssessment& assessment : assessments) {
            if (assessment.outcome != CollisionOutcome::Damage) continue;
            const double increment = std::clamp(
                assessment.specific_impact_energy / std::max(assessment.damage_limit, 1e-12),
                0.0, 1.0);
            damage_by_body[assessment.body.value] += increment;
        }
    }

    for (std::size_t index = 0; index < world.bodyCount(); ++index) {
        MutableBodyView body = world.mutableBody(index);
        const auto damage = damage_by_body.find(body.id.value);
        if (damage != damage_by_body.end()) {
            body.accumulated_damage = std::clamp(body.accumulated_damage + damage->second, 0.0, 1.0);
        }
    }
}

std::vector<Vec3> fragmentDirections(std::size_t count, Dimension dimension) {
    constexpr double pi = 3.14159265358979323846;
    std::vector<Vec3> directions;
    directions.reserve(count);
    Vec3 mean{};
    for (std::size_t index = 0; index < count; ++index) {
        Vec3 direction;
        if (dimension == Dimension::Two) {
            const double angle = 2.0 * pi * static_cast<double>(index) / static_cast<double>(count);
            direction = {std::cos(angle), std::sin(angle), 0.0};
        } else {
            constexpr double golden_angle = 2.39996322972865332223;
            const double normalized_index = (static_cast<double>(index) + 0.5) / static_cast<double>(count);
            const double z = 1.0 - 2.0 * normalized_index;
            const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double angle = golden_angle * static_cast<double>(index);
            direction = {radial * std::cos(angle), radial * std::sin(angle), z};
        }
        directions.push_back(direction);
        mean += direction;
    }
    mean = mean * (1.0 / static_cast<double>(count));
    for (Vec3& direction : directions) direction -= mean;
    return directions;
}

std::vector<BodyState> fragmentBody(const BodyState& body, std::size_t count,
                                    double specific_energy, Dimension dimension) {
    std::vector<BodyState> fragments;
    fragments.reserve(count);
    const double spread = 0.1 * std::sqrt(std::max(0.0, specific_energy));
    const double radius_scale = dimension == Dimension::Two
        ? 1.0 / std::sqrt(static_cast<double>(count))
        : 1.0 / std::cbrt(static_cast<double>(count));
    const std::vector<Vec3> directions = fragmentDirections(count, dimension);

    for (std::size_t index = 0; index < count; ++index) {
        BodyState fragment = body;
        fragment.id = {};
        fragment.mass = body.mass / static_cast<double>(count);
        fragment.radius = body.radius * radius_scale;
        fragment.accumulated_damage = 0.0;
        fragment.position += directions[index] * (body.radius * 0.25);
        fragment.velocity += directions[index] * spread;
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
    const std::vector<BodyState> current_bodies = world.snapshotBodies();
    std::vector<BodyState> replacement;
    replacement.reserve(current_bodies.size() + fragment_ids.size() * minimum);
    std::size_t fragment_count_total = 0;
    for (const BodyState& body : current_bodies) {
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
    CollisionSettings effective_settings = settings;
    world.clearCollisionEvents();
    world.resetCollisionTransitionDiagnostics(settings.model);
    const bool classifier_fragmentation_requested = effective_settings.classifier.fragmentation_enabled
        || effective_settings.classifier.force_fragmentation;
    const bool classifier_outcome_requested = classifier_fragmentation_requested
        || effective_settings.classifier.absorption_enabled;
    if (effective_settings.model == CollisionModel::Transparent && classifier_outcome_requested) {
        throw std::invalid_argument("Fragmentation and absorption settings require hard-body collision mode");
    }
    if (world.activeCollisionModel() == CollisionModel::Transparent
        && effective_settings.model == CollisionModel::HardBody) {
        std::vector<Vec3> original_positions;
        original_positions.reserve(world.bodyCount());
        for (std::size_t index = 0; index < world.bodyCount(); ++index) {
            original_positions.push_back(world.body(index).position);
        }

        std::size_t passes = 0;
        std::vector<BodyId> unresolved_bodies;
        if (!untangleBodies(world, effective_settings.maximum_consistency_passes, passes, unresolved_bodies)) {
            for (std::size_t index = 0; index < original_positions.size(); ++index) {
                world.mutableBody(index).position = original_positions[index];
            }
            world.recordCollisionTransitionFailure(passes, std::move(unresolved_bodies));
            return;
        }
        world.recordCollisionTransitionSuccess(passes);
        world.setActiveCollisionModel(CollisionModel::HardBody);
    }

    switch (effective_settings.model) {
    case CollisionModel::Transparent:
        // Transparent bodies deliberately receive no contact impulse or overlap correction.
        world.setActiveCollisionModel(CollisionModel::Transparent);
        return;
    case CollisionModel::HardBody: {
        const Dimension dimension = world.dimension();
        const double restitution = std::clamp(effective_settings.restitution, 0.0, 1.0);
        CollisionWorkspace& workspace = collisionWorkspace(dimension, effective_settings.broad_phase);
        if (effective_settings.broad_phase.spatial_tree_enabled) {
            workspace.tree.rebuild(world.bodyStorage());
            workspace.tree.potentialContactPairs(world.bodyStorage(), workspace.candidate_pairs);
        } else {
            workspace.candidate_pairs.clear();
            for (std::size_t first = 0; first < world.bodyCount(); ++first) {
                for (std::size_t second = first + 1; second < world.bodyCount(); ++second) {
                    workspace.candidate_pairs.emplace_back(first, second);
                }
            }
        }

        world.reserveCollisionEvents(workspace.candidate_pairs.size());
        std::vector<std::uint8_t> pending_removal;
        if (effective_settings.classifier.absorption_enabled) {
            pending_removal.assign(world.bodyCount(), 0);
        }
        for (const auto [first, second] : workspace.candidate_pairs) {
                MutableBodyView first_body = world.mutableBody(first);
                MutableBodyView second_body = world.mutableBody(second);
                if (!pending_removal.empty() && (pending_removal[first] || pending_removal[second])) continue;
                const Vec3 displacement = second_body.position - first_body.position;
                const double distance_squared = displacement.lengthSquared(dimension);
                const double combined_radius = first_body.radius + second_body.radius;
                if (distance_squared > combined_radius * combined_radius) continue;

                const double distance = std::sqrt(distance_squared);
                const Vec3 normal = distance > 1e-12
                    ? displacement * (1.0 / distance)
                    : Vec3{1.0, 0.0, 0.0};
                const double penetration = std::max(0.0, combined_radius - distance);
                const double first_inverse_mass = first_body.is_static() || first_body.mass <= 0.0 ? 0.0 : 1.0 / first_body.mass;
                const double second_inverse_mass = second_body.is_static() || second_body.mass <= 0.0 ? 0.0 : 1.0 / second_body.mass;
                const double inverse_mass_sum = first_inverse_mass + second_inverse_mass;

                AbsorptionDecision absorption;
                if (effective_settings.classifier.absorption_enabled) {
                    absorption = chooseAbsorption(first_body, second_body, effective_settings);
                }

                if (absorption.survivor.isValid()) {
                    const std::size_t removed_index = absorption.survivor == first_body.id ? second : first;
                    pending_removal[removed_index] = 1;
                } else if (inverse_mass_sum > 0.0 && penetration > 0.0) {
                    const Vec3 correction = normal * (penetration / inverse_mass_sum);
                    first_body.position -= correction * first_inverse_mass;
                    second_body.position += correction * second_inverse_mass;
                    if (dimension == Dimension::Two) {
                        first_body.position.z = 0.0;
                        second_body.position.z = 0.0;
                    }
                }

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
                       + effective_settings.classifier.tangential_energy_scale * tangential_speed * tangential_speed);
                CollisionEvent event{
                    first_body.id,
                    second_body.id,
                    normal_speed,
                    tangential_speed,
                    impact_energy,
                    assessBody(first_body, impact_energy, gravitational_constant, effective_settings.classifier),
                    assessBody(second_body, impact_energy, gravitational_constant, effective_settings.classifier)
                };
                if (absorption.survivor.isValid()) {
                    event.absorption_survivor = absorption.survivor;
                    event.absorption_removed = absorption.removed;
                    event.absorber_destroyed = !isAbsorber(
                        absorption.survivor == first_body.id ? first_body.kind : second_body.kind);
                    if (absorption.survivor == first_body.id) {
                        event.first.outcome = isAbsorber(first_body.kind) ? CollisionOutcome::Absorb
                                                                          : CollisionOutcome::Bounce;
                        event.second.outcome = isAbsorber(second_body.kind) ? CollisionOutcome::DestroyAbsorber
                                                                            : CollisionOutcome::Absorb;
                    } else {
                        event.second.outcome = isAbsorber(second_body.kind) ? CollisionOutcome::Absorb
                                                                             : CollisionOutcome::Bounce;
                        event.first.outcome = isAbsorber(first_body.kind) ? CollisionOutcome::DestroyAbsorber
                                                                           : CollisionOutcome::Absorb;
                    }
                }
                world.recordCollisionEvent(std::move(event));
                if (absorption.survivor.isValid() || inverse_mass_sum == 0.0) continue;
                if (normal_velocity >= 0.0) continue;

                const double material_restitution = std::min(first_body.material.restitution, second_body.material.restitution);
                const double effective_restitution = std::clamp(std::min(restitution, material_restitution), 0.0, 1.0);
                const double impulse_magnitude = -(1.0 + effective_restitution) * normal_velocity / inverse_mass_sum;
                const Vec3 impulse = normal * impulse_magnitude;
                if (!first_body.is_static()) first_body.velocity -= impulse * first_inverse_mass;
                if (!second_body.is_static()) second_body.velocity += impulse * second_inverse_mass;
                if (dimension == Dimension::Two) {
                    first_body.velocity.z = 0.0;
                    second_body.velocity.z = 0.0;
                }
        }
        world.setActiveCollisionModel(CollisionModel::HardBody);
        return;
    }
    }
    throw std::logic_error("Unknown collision model");
}

void CollisionSystem::applyDeferredOutcomes(WorldState& world, const CollisionSettings& settings) {
    applyDeferredAbsorption(world, settings);
    applyDeferredDamage(world);
    applyDeferredFragmentation(world, settings);
}

}
