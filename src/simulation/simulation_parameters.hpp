#pragma once

#include "dimension.hpp"
#include "solver_configuration.hpp"

namespace nbody {

struct SimulationParameters {
    Dimension dimension{Dimension::Two};
    double gravitational_constant{1.0};
    double timestep{0.01};
    unsigned long long seed{1};
    SolverConfiguration solver;
};

}
