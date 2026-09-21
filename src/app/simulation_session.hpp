#pragma once

#include "simulation/physics_session.hpp"

#include <optional>
#include <memory>

namespace nbody::app {

enum class SimulationMode {
    StartupEdit,
    Running,
    Paused,
    Edit,
    EnteringEdit,
    LeavingEdit
};

struct SimulationState {
    WorldState world{Dimension::Two};
    SimulationParameters parameters;
};

class SimulationSession {
public:
    SimulationSession();
    explicit SimulationSession(SimulationState state);

    SimulationState& state() { return state_; }
    const SimulationState& state() const { return state_; }

    SimulationMode mode() const { return mode_; }
    void setMode(SimulationMode mode) { mode_ = mode; }

    PhysicsStepResult step();
    FramePublication publishFrame(FramePublicationRequest request = {});

    void captureInitialState();
    bool hasInitialState() const { return initial_state_.has_value(); }
    bool restoreInitialState();

    const PhysicsSession& physics() const { return *physics_; }
    PhysicsSession& physics() { return *physics_; }

private:
    void recreatePhysicsSession();

    SimulationState state_;
    std::optional<SimulationState> initial_state_;
    SimulationMode mode_{SimulationMode::StartupEdit};
    std::unique_ptr<PhysicsSession> physics_;
};

}
