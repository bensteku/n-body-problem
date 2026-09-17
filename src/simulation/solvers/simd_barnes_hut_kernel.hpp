#pragma once

#include "simulation/barnes_hut_tree.hpp"
#include "simulation/simulation_parameters.hpp"
#include "simulation/world_state.hpp"

#include <vector>

namespace nbody {

void calculateAvx2BarnesHutAccelerations(const WorldState& world, const SimulationParameters& parameters,
                                         const BarnesHutTree& tree, std::vector<Vec3>& output);

}
