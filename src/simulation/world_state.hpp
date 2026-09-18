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
    CollisionModel activeCollisionModel() const { return active_collision_model_; }
    const CollisionTransitionDiagnostics& collisionTransitionDiagnostics() const {
        return collision_transition_diagnostics_;
    }
    void setActiveCollisionModel(CollisionModel model) {
        active_collision_model_ = model;
        collision_transition_diagnostics_.active_model = model;
    }
    void resetCollisionTransitionDiagnostics(CollisionModel requested_model) {
        collision_transition_diagnostics_ = {};
        collision_transition_diagnostics_.active_model = active_collision_model_;
        collision_transition_diagnostics_.requested_model = requested_model;
    }
    void recordCollisionTransitionFailure(std::size_t passes, std::vector<BodyId> unresolved_bodies) {
        collision_transition_diagnostics_.failed = true;
        collision_transition_diagnostics_.passes = passes;
        collision_transition_diagnostics_.unresolved_bodies = std::move(unresolved_bodies);
    }
    void recordCollisionTransitionSuccess(std::size_t passes) {
        collision_transition_diagnostics_.failed = false;
        collision_transition_diagnostics_.passes = passes;
        collision_transition_diagnostics_.unresolved_bodies.clear();
    }
    ConstBodyView body(std::size_t index) const { return storage_.view(index); }
    MutableBodyView mutableBody(std::size_t index) { return storage_.mutableView(index); }
    void clearCollisionEvents() { collision_events_.clear(); }
    void reserveCollisionEvents(std::size_t count) { collision_events_.reserve(count); }
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
    CollisionModel active_collision_model_{CollisionModel::Transparent};
    CollisionTransitionDiagnostics collision_transition_diagnostics_;
};

}
