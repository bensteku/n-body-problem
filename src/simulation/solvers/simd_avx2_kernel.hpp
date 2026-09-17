#pragma once

#include "simulation/simulation_parameters.hpp"
#include "simulation/world_state.hpp"

#include <vector>

namespace nbody {

void calculateAvx2Accelerations(const WorldState& world, const SimulationParameters& parameters,
                                std::vector<Vec3>& output);

}
