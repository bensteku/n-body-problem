#pragma once

#include "simulation_parameters.hpp"
#include "simulation_frame.hpp"
#include "world_state.hpp"

#include <cstddef>
#include <string_view>

namespace nbody {

struct PhysicsEngineInfo {
    ComputeBackend backend{ComputeBackend::Scalar};
    ForceModel force_model{ForceModel::Full};
    Dimension dimension{Dimension::Two};
    std::string_view name{"Scalar Full"};
    SolverKind kind{SolverKind::Full};
};

struct PhysicsEngineCapabilities {
    SolverConfiguration requested;
    PhysicsEngineInfo effective;
    bool fallback{false};
    std::string_view status{"available"};
};

struct PhysicsPhaseTimings {
    double force_ms{};
    double integration_ms{};
    double collision_ms{};
    double boundary_ms{};
    double deferred_outcomes_ms{};
    double total_ms{};
};

enum class PhysicsStepStatus {
    Advanced,
    Skipped,
    Rejected
};

struct PhysicsStepResult {
    PhysicsStepStatus status{PhysicsStepStatus::Skipped};
    double simulation_time{};
    std::size_t body_count{};
    std::string_view message{};
    PhysicsPhaseTimings timings;

    bool advanced() const { return status == PhysicsStepStatus::Advanced; }
    bool rejected() const { return status == PhysicsStepStatus::Rejected; }
};

class IPhysicsEngine {
public:
    virtual ~IPhysicsEngine() = default;
    virtual PhysicsStepResult step(WorldState& world,
                                   const SimulationParameters& parameters) = 0;
    virtual PhysicsEngineInfo info(Dimension dimension) const = 0;

    // Publication is intentionally shared at the API boundary while the
    // numerical engine remains free to use backend-specific internal storage.
    FramePublication publishFrame(const WorldState& world, IFramePublisher& publisher,
                                  FramePublicationRequest request = {}) const {
        return publisher.publish(world, request);
    }
};

}
