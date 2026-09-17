#include "simulation/world_state.hpp"

#include <random>
#include <utility>

namespace nbody {

WorldState::WorldState(Dimension dimension) : dimension_(dimension) {}

BodyId WorldState::addBody(BodyState body) {
    body.id = next_id_;
    ++next_id_.value;
    if (dimension_ == Dimension::Two) {
        body.position.z = 0.0;
        body.velocity.z = 0.0;
    }
    storage_.append(body);
    return body.id;
}

void WorldState::advance(double timestep) {
    for (std::size_t index = 0; index < storage_.size(); ++index) {
        MutableBodyView body = storage_.mutableView(index);
        if (!body.is_static()) {
            body.position += body.velocity * timestep;
            if (dimension_ == Dimension::Two) {
                body.position.z = 0.0;
                body.velocity.z = 0.0;
            }
        }
    }
    time_ += timestep;
}

void WorldState::replaceBodies(std::vector<BodyState> bodies) {
    storage_.clear();
    storage_.reserve(bodies.size());
    for (BodyState body : bodies) {
        if (dimension_ == Dimension::Two) {
            body.position.z = 0.0;
            body.velocity.z = 0.0;
        }
        storage_.append(body);
    }
    std::uint64_t largest_id = 0;
    for (const BodyState& body : bodies) {
        if (body.id.value > largest_id) largest_id = body.id.value;
    }
    next_id_.value = largest_id + 1;
    for (std::size_t index = 0; index < storage_.size(); ++index) {
        MutableBodyView body = storage_.mutableView(index);
        if (!body.id.isValid()) {
            body.id = next_id_;
            ++next_id_.value;
        }
    }
}

WorldDiagnostics WorldState::diagnostics() const {
    WorldDiagnostics result;
    Vec3 weighted_position{};
    for (std::size_t index = 0; index < storage_.size(); ++index) {
        ConstBodyView body = storage_.view(index);
        result.total_mass += body.mass;
        weighted_position += body.position * body.mass;
        result.finite = result.finite && body.position.isFinite() && body.velocity.isFinite();
    }
    if (result.total_mass != 0.0) {
        result.center_of_mass = weighted_position * (1.0 / result.total_mass);
    }
    return result;
}

bool WorldState::isValid() const {
    if (next_id_.value == 0) return false;
    for (std::size_t index = 0; index < storage_.size(); ++index) {
        ConstBodyView body = storage_.view(index);
        if (!body.id.isValid() || body.mass < 0.0 || body.radius < 0.0
            || body.accumulated_damage < 0.0 || body.accumulated_damage > 1.0) return false;
        if (!body.position.isFinite() || !body.velocity.isFinite()) return false;
        if (dimension_ == Dimension::Two && (body.position.z != 0.0 || body.velocity.z != 0.0)) return false;
    }
    return true;
}

WorldState WorldState::deterministic(std::size_t body_count, Dimension dimension, unsigned long long seed) {
    WorldState world(dimension);
    world.reserveBodies(body_count);
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> position(-10.0, 10.0);
    std::uniform_real_distribution<double> velocity(-0.1, 0.1);
    std::uniform_real_distribution<double> mass(0.5, 2.0);

    for (std::size_t index = 0; index < body_count; ++index) {
        BodyState body;
        body.position = {position(generator), position(generator), dimension == Dimension::Three ? position(generator) : 0.0};
        body.velocity = {velocity(generator), velocity(generator), dimension == Dimension::Three ? velocity(generator) : 0.0};
        body.mass = mass(generator);
        body.radius = 0.1;
        world.addBody(body);
    }
    return world;
}

}
