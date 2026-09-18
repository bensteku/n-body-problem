#include "simulation/world_state.hpp"
#include "simulation/physics_session.hpp"

#include <iostream>
#include <string_view>

namespace {

nbody::Dimension parseDimension(int argc, char** argv) {
    if (argc > 1 && (std::string_view(argv[1]) == "3" || std::string_view(argv[1]) == "3D")) {
        return nbody::Dimension::Three;
    }
    return nbody::Dimension::Two;
}

nbody::ComputeBackend parseBackend(int argc, char** argv) {
    if (argc > 2 && std::string_view(argv[2]) == "SIMD") return nbody::ComputeBackend::SIMD;
    if (argc > 2 && std::string_view(argv[2]) == "GPU") return nbody::ComputeBackend::GPU;
    return nbody::ComputeBackend::Scalar;
}

nbody::ForceModel parseForceModel(int argc, char** argv) {
    return argc > 3 && (std::string_view(argv[3]) == "BaHu"
        || std::string_view(argv[3]) == "BarnesHut")
        ? nbody::ForceModel::BarnesHut : nbody::ForceModel::Full;
}

}

int main(int argc, char** argv) {
    constexpr unsigned long long seed = 42;
    const nbody::Dimension dimension = parseDimension(argc, argv);
    nbody::WorldState world = nbody::WorldState::deterministic(3, dimension, seed);
    nbody::SimulationParameters parameters;
    parameters.dimension = dimension;
    parameters.timestep = 0.01;
    parameters.solver.backend = parseBackend(argc, argv);
    parameters.solver.force_model = parseForceModel(argc, argv);
    parameters.solver.kind = parameters.solver.force_model == nbody::ForceModel::BarnesHut
        ? nbody::SolverKind::Approximated : nbody::SolverKind::Full;
    nbody::PhysicsSession physics_session(parameters.solver);
    const bool gpu_fallback = parameters.solver.backend == nbody::ComputeBackend::GPU
        && physics_session.engine().info(dimension).backend != nbody::ComputeBackend::GPU;
    physics_session.step(world, parameters);
    const nbody::FramePublication publication = physics_session.publishFrame(world);
    const nbody::FrameLease& frame = publication.cpu_snapshot;
    const nbody::WorldDiagnostics diagnostics = world.diagnostics();

    std::cout << "nbody greenfield scaffold\n"
              << "dimension=" << (dimension == nbody::Dimension::Two ? "2D" : "3D") << "\n"
              << "bodies=" << world.bodyCount() << "\n"
              << "time=" << world.time() << "\n"
              << "solver=";
    if (gpu_fallback) {
        std::cout << "GPU unavailable (fallback: " << physics_session.engine().info(dimension).name << ")\n";
    } else {
        std::cout << physics_session.engine().info(dimension).name << "\n";
    }
    std::cout << "total_mass=" << diagnostics.total_mass << "\n"
              << "published_bodies=" << frame->body_count << "\n"
              << "finite=" << (diagnostics.finite ? "true" : "false") << '\n';
    return world.isValid() ? 0 : 1;
}
