#include "app/simulation_session.hpp"

#include <memory>

namespace nbody::app {

SimulationSession::SimulationSession()
    : SimulationSession(SimulationState{}) {}

SimulationSession::SimulationSession(SimulationState state)
    : state_(std::move(state)),
      physics_(std::make_unique<PhysicsSession>(state_.parameters.solver)) {}

PhysicsStepResult SimulationSession::step() {
    return physics_->step(state_.world, state_.parameters);
}

FramePublication SimulationSession::publishFrame(FramePublicationRequest request) {
    return physics_->publishFrame(state_.world, request);
}

void SimulationSession::captureInitialState() {
    initial_state_ = state_;
}

bool SimulationSession::restoreInitialState() {
    if (!initial_state_) return false;
    state_ = *initial_state_;
    recreatePhysicsSession();
    mode_ = SimulationMode::Paused;
    return true;
}

void SimulationSession::recreatePhysicsSession() {
    physics_ = std::make_unique<PhysicsSession>(state_.parameters.solver);
}

}
