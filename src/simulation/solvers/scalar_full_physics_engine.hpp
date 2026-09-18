#pragma once

#include "simulation/physics_engine.hpp"

#include <vector>

namespace nbody {

class ScalarFullPhysicsEngine final : public IPhysicsEngine {
public:
    void step(WorldState& world, const SimulationParameters& parameters) override;
    PhysicsEngineInfo info(Dimension dimension) const override;

private:
    void calculateAccelerations(const WorldState& world, const SimulationParameters& parameters,
                               std::vector<Vec3>& accelerations) const;

    std::vector<Vec3> initial_accelerations_;
    std::vector<Vec3> final_accelerations_;
};

}
