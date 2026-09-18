#pragma once

#include "simulation_parameters.hpp"
#include "world_state.hpp"

#include <string_view>

namespace nbody {

struct PhysicsEngineInfo {
    ComputeBackend backend{ComputeBackend::Scalar};
    ForceModel force_model{ForceModel::Full};
    Dimension dimension{Dimension::Two};
    std::string_view name{"Scalar Full"};
    SolverKind kind{SolverKind::Full};
};

class IPhysicsEngine {
public:
    virtual ~IPhysicsEngine() = default;
    virtual void step(WorldState& world, const SimulationParameters& parameters) = 0;
    virtual PhysicsEngineInfo info(Dimension dimension) const = 0;
};

}
