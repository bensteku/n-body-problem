#pragma once

#include "simulation/barnes_hut_tree.hpp"
#include "simulation/physics_engine.hpp"
#include "simulation/simd_capabilities.hpp"

#include <memory>
#include <vector>

namespace nbody {

class SimdApproximatedPhysicsEngine final : public IPhysicsEngine {
public:
    SimdApproximatedPhysicsEngine();

    void step(WorldState& world, const SimulationParameters& parameters) override;
    PhysicsEngineInfo info(Dimension dimension) const override;

private:
    void calculateAccelerations(const WorldState& world, const SimulationParameters& parameters,
                               std::vector<Vec3>& output) const;
    mutable std::vector<Vec3> initial_accelerations_;
    mutable std::vector<Vec3> final_accelerations_;
    mutable std::unique_ptr<BarnesHutTree> tree_;
    SimdCapabilities capabilities_;
};

}
