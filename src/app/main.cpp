#include "simulation/world_state.hpp"
#include "simulation/physics_engine_factory.hpp"

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
    std::unique_ptr<nbody::IPhysicsEngine> physics_engine;
#ifdef NBODY_FORCE_MODEL_BARNES_HUT
    parameters.solver.force_model = nbody::ForceModel::BarnesHut;
    parameters.solver.kind = nbody::SolverKind::Approximated;
#ifdef NBODY_BACKEND_SIMD
    parameters.solver.backend = nbody::ComputeBackend::SIMD;
#else
    parameters.solver.backend = nbody::ComputeBackend::Scalar;
#endif
#elif defined(NBODY_BACKEND_SIMD)
    parameters.solver.force_model = nbody::ForceModel::Full;
    parameters.solver.kind = nbody::SolverKind::Full;
    parameters.solver.backend = nbody::ComputeBackend::SIMD;
#elif defined(NBODY_BACKEND_GPU)
    // Vulkan GPU support is the selected future backend; keep this build honest
    // until the Vulkan device and compute implementation are available.
    parameters.solver.force_model = nbody::ForceModel::Full;
    parameters.solver.kind = nbody::SolverKind::Full;
    parameters.solver.backend = nbody::ComputeBackend::GPU;
#else
    parameters.solver.force_model = nbody::ForceModel::Full;
    parameters.solver.kind = nbody::SolverKind::Full;
    parameters.solver.backend = nbody::ComputeBackend::Scalar;
#endif
    physics_engine = nbody::createPhysicsEngine(parameters.solver);
    const bool gpu_fallback = parameters.solver.backend == nbody::ComputeBackend::GPU
        && physics_engine->info(dimension).backend != nbody::ComputeBackend::GPU;
    physics_engine->step(world, parameters);
    const nbody::WorldDiagnostics diagnostics = world.diagnostics();

    std::cout << "nbody greenfield scaffold\n"
              << "dimension=" << (dimension == nbody::Dimension::Two ? "2D" : "3D") << "\n"
              << "bodies=" << world.bodyCount() << "\n"
              << "time=" << world.time() << "\n"
              << "solver=";
    if (gpu_fallback) {
        std::cout << "GPU unavailable (fallback: " << physics_engine->info(dimension).name << ")\n";
    } else {
        std::cout << physics_engine->info(dimension).name << "\n";
    }
    std::cout << "total_mass=" << diagnostics.total_mass << "\n"
              << "finite=" << (diagnostics.finite ? "true" : "false") << '\n';
    return world.isValid() ? 0 : 1;
}
