#include "simulation/world_state.hpp"
#include "simulation/solver.hpp"
#include "simulation/solvers/scalar_full_solver.hpp"
#include "simulation/solvers/scalar_barnes_hut_solver.hpp"

#include <iostream>
#include <memory>
#include <string_view>

namespace {

nbody::Dimension parseDimension(int argc, char** argv) {
    if (argc > 1 && (std::string_view(argv[1]) == "3" || std::string_view(argv[1]) == "3D")) {
        return nbody::Dimension::Three;
    }
    return nbody::Dimension::Two;
}

}

int main(int argc, char** argv) {
    constexpr unsigned long long seed = 42;
    const nbody::Dimension dimension = parseDimension(argc, argv);
    nbody::WorldState world = nbody::WorldState::deterministic(3, dimension, seed);
    nbody::SimulationParameters parameters;
    parameters.dimension = dimension;
    parameters.timestep = 0.01;
    std::unique_ptr<nbody::ISolver> solver;
#ifdef NBODY_FORCE_MODEL_BARNES_HUT
    parameters.solver.force_model = nbody::ForceModel::BarnesHut;
    solver = std::make_unique<nbody::ScalarBarnesHutSolver>();
#else
    parameters.solver.force_model = nbody::ForceModel::Full;
    solver = std::make_unique<nbody::ScalarFullSolver>();
#endif
    solver->step(world, parameters);
    const nbody::WorldDiagnostics diagnostics = world.diagnostics();

    std::cout << "nbody greenfield scaffold\n"
              << "dimension=" << (dimension == nbody::Dimension::Two ? "2D" : "3D") << "\n"
              << "bodies=" << world.bodyCount() << "\n"
              << "time=" << world.time() << "\n"
              << "solver=" << solver->info(dimension).name << "\n"
              << "total_mass=" << diagnostics.total_mass << "\n"
              << "finite=" << (diagnostics.finite ? "true" : "false") << '\n';
    return world.isValid() ? 0 : 1;
}
