#pragma once

#include "body_state.hpp"
#include "body_storage.hpp"
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
    std::size_t bodyCount() const { return storage_.size(); }
    void reserveBodies(std::size_t count) { storage_.reserve(count); }
    // Snapshotting is intentionally explicit: simulation hot paths should use body()/mutableBody().
    std::vector<BodyState> snapshotBodies() const { return storage_.snapshot(); }
    // Compatibility name for callers that still expect a value snapshot.
    std::vector<BodyState> bodies() const { return storage_.snapshot(); }
    const BodyStorage& bodyStorage() const { return storage_; }
    const std::vector<CollisionEvent>& collisionEvents() const { return collision_events_; }
    ConstBodyView body(std::size_t index) const { return storage_.view(index); }
    MutableBodyView mutableBody(std::size_t index) { return storage_.mutableView(index); }
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
    BodyStorage storage_;
    std::vector<CollisionEvent> collision_events_;
};

}
