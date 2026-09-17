#pragma once

#include "simulation/simd_capabilities.hpp"
#include "simulation/solver.hpp"

#include <vector>

namespace nbody {

class SimdFullSolver final : public ISolver {
public:
    SimdFullSolver();

    void step(WorldState& world, const SimulationParameters& parameters) override;
    SolverInfo info(Dimension dimension) const override;
    SimdCapabilities capabilities() const { return capabilities_; }

private:
    void calculateAccelerations(const WorldState& world, const SimulationParameters& parameters,
                                std::vector<Vec3>& output) const;
    mutable std::vector<Vec3> initial_accelerations_;
    mutable std::vector<Vec3> final_accelerations_;
    SimdCapabilities capabilities_;
};

}
