#pragma once

#include "body_state.hpp"
#include "simulation_parameters.hpp"

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace nbody {

struct WorldDiagnostics {
    double total_mass{};
    Vec3 center_of_mass{};
    bool finite{true};
};

class WorldState {
public:
    explicit WorldState(Dimension dimension = Dimension::Two);

    Dimension dimension() const { return dimension_; }
    double time() const { return time_; }
    const std::vector<BodyState>& bodies() const { return bodies_; }
    const std::vector<CollisionEvent>& collisionEvents() const { return collision_events_; }
    std::span<BodyState> mutableBodies() { return bodies_; }
    void clearCollisionEvents() { collision_events_.clear(); }
    void recordCollisionEvent(CollisionEvent event) { collision_events_.push_back(event); }
    void replaceBodies(std::vector<BodyState> bodies);

    BodyId addBody(BodyState body);
    void advance(double timestep);
    void advanceTime(double timestep) { time_ += timestep; }
    WorldDiagnostics diagnostics() const;
    bool isValid() const;

    static WorldState deterministic(std::size_t body_count, Dimension dimension, unsigned long long seed);

private:
    Dimension dimension_;
    double time_{};
    BodyId next_id_{1};
    std::vector<BodyState> bodies_;
    std::vector<CollisionEvent> collision_events_;
};

}
