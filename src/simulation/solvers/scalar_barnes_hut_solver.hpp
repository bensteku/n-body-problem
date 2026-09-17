#pragma once

#include "simulation/barnes_hut_tree.hpp"
#include "simulation/solver.hpp"

#include <memory>
#include <vector>

namespace nbody {

class ScalarBarnesHutSolver final : public ISolver {
public:
    void step(WorldState& world, const SimulationParameters& parameters) override;
    SolverInfo info(Dimension dimension) const override;

private:
    void calculateAccelerations(const WorldState& world, const SimulationParameters& parameters,
                                std::vector<Vec3>& output) const;
    mutable std::vector<Vec3> initial_accelerations_;
    mutable std::vector<Vec3> final_accelerations_;
    mutable std::unique_ptr<BarnesHutTree> tree_;
};

}
