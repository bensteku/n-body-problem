#pragma once

#include "physics_engine_factory.hpp"

#include <memory>
#include <optional>
#include <string_view>

namespace nbody {

enum class EngineSwitchStatus {
    Switched,
    Rejected
};

struct EngineSwitchResult {
    EngineSwitchStatus status{EngineSwitchStatus::Rejected};
    std::string_view message{};

    bool switched() const { return status == EngineSwitchStatus::Switched; }
};

// Owns the selected engine and the presentation publisher. Switching is an
// atomic boundary: the candidate engine is created and validated before it
// replaces the current engine. Callers should invoke it only at a completed
// simulation-frame boundary (paused/edit mode in an interactive application).
class PhysicsSession {
public:
    explicit PhysicsSession(SolverConfiguration configuration = {});

    PhysicsStepResult step(WorldState& world, const SimulationParameters& parameters);
    FrameLease publishFrame(const WorldState& world);
    EngineSwitchResult switchEngine(const SolverConfiguration& configuration,
                                    const WorldState& world);

    const SolverConfiguration& configuration() const { return configuration_; }
    const IPhysicsEngine& engine() const { return *engine_; }
    IPhysicsEngine& engine() { return *engine_; }

private:
    SolverConfiguration configuration_;
    std::unique_ptr<IPhysicsEngine> engine_;
    FramePublisher frame_publisher_;
    std::optional<Dimension> dimension_;
};

}
