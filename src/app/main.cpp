#include "simulation/world_state.hpp"
#include "simulation/solver.hpp"
#include "simulation/solvers/scalar_full_solver.hpp"
#include "simulation/solvers/scalar_barnes_hut_solver.hpp"
#include "simulation/solvers/simd_barnes_hut_solver.hpp"
#include "simulation/solvers/simd_full_solver.hpp"

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
    bool gpu_fallback = false;
#ifdef NBODY_FORCE_MODEL_BARNES_HUT
    parameters.solver.force_model = nbody::ForceModel::BarnesHut;
#ifdef NBODY_BACKEND_SIMD
    solver = std::make_unique<nbody::SimdBarnesHutSolver>();
#else
    solver = std::make_unique<nbody::ScalarBarnesHutSolver>();
#endif
#elif defined(NBODY_BACKEND_SIMD)
    parameters.solver.force_model = nbody::ForceModel::Full;
    solver = std::make_unique<nbody::SimdFullSolver>();
#elif defined(NBODY_BACKEND_GPU)
    // Vulkan GPU support is the selected future backend; keep this build honest
    // until the Vulkan device and compute implementation are available.
    gpu_fallback = true;
    parameters.solver.force_model = nbody::ForceModel::Full;
    solver = std::make_unique<nbody::ScalarFullSolver>();
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
              << "solver=";
    if (gpu_fallback) {
        std::cout << "GPU unavailable (fallback: " << solver->info(dimension).name << ")\n";
    } else {
        std::cout << solver->info(dimension).name << "\n";
    }
    std::cout << "total_mass=" << diagnostics.total_mass << "\n"
              << "finite=" << (diagnostics.finite ? "true" : "false") << '\n';
    return world.isValid() ? 0 : 1;
}
