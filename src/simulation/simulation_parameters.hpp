#pragma once

#include "dimension.hpp"
#include "collision.hpp"
#include "solver_configuration.hpp"

namespace nbody {

enum class Integrator { VelocityVerlet };

struct SimulationParameters {
    Dimension dimension{Dimension::Two};
    double gravitational_constant{1.0};
    double timestep{0.01};
    double softening_length{1e-6};
    Integrator integrator{Integrator::VelocityVerlet};
    unsigned long long seed{1};
    SolverConfiguration solver;
    CollisionSettings collision;
};

}
