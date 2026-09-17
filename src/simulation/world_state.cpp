#include "simulation/world_state.hpp"

#include <random>

namespace nbody {

WorldState::WorldState(Dimension dimension) : dimension_(dimension) {}

BodyId WorldState::addBody(BodyState body) {
    body.id = next_id_;
    ++next_id_.value;
    if (dimension_ == Dimension::Two) {
        body.position.z = 0.0;
        body.velocity.z = 0.0;
    }
    bodies_.push_back(body);
    return body.id;
}

void WorldState::advance(double timestep) {
    for (BodyState& body : bodies_) {
        if (!body.is_static) {
            body.position += body.velocity * timestep;
            if (dimension_ == Dimension::Two) {
                body.position.z = 0.0;
                body.velocity.z = 0.0;
            }
        }
    }
    time_ += timestep;
}

WorldDiagnostics WorldState::diagnostics() const {
    WorldDiagnostics result;
    Vec3 weighted_position{};
    for (const BodyState& body : bodies_) {
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
    for (const BodyState& body : bodies_) {
        if (!body.id.isValid() || body.mass < 0.0 || body.radius < 0.0) return false;
        if (!body.position.isFinite() || !body.velocity.isFinite()) return false;
        if (dimension_ == Dimension::Two && (body.position.z != 0.0 || body.velocity.z != 0.0)) return false;
    }
    return true;
}

WorldState WorldState::deterministic(std::size_t body_count, Dimension dimension, unsigned long long seed) {
    WorldState world(dimension);
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
