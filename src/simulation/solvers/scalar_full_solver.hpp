#pragma once

#include "simulation/solver.hpp"

namespace nbody {

class ScalarFullSolver final : public ISolver {
public:
    void step(WorldState& world, const SimulationParameters& parameters) override;
    SolverInfo info(Dimension dimension) const override;
};

}
