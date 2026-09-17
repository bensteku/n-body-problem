#include "simulation/world_state.hpp"
#include "simulation/solvers/scalar_full_solver.hpp"
#include "simulation/solvers/scalar_barnes_hut_solver.hpp"
#include "simulation/solvers/simd_full_solver.hpp"
#include "simulation/solvers/simd_barnes_hut_solver.hpp"
#include "simulation/collision_system.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <string_view>

namespace {

template <typename Solver>
int runIsolatedSolver(std::string_view label, std::size_t body_count, int steps,
                      Solver& solver, nbody::ForceModel force_model) {
    nbody::WorldState world = nbody::WorldState::deterministic(body_count, nbody::Dimension::Two, 42);
    nbody::SimulationParameters parameters;
    parameters.dimension = nbody::Dimension::Two;
    parameters.gravitational_constant = 0.1;
    parameters.timestep = 0.001;
    parameters.collision.model = nbody::CollisionModel::Transparent;
    parameters.solver.force_model = force_model;

    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) solver.step(world, parameters);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
    std::cout << "isolated solver=" << label
              << " dimension=2D steps=" << steps
              << " bodies=" << body_count
              << " elapsed_ms=" << elapsed_ms
              << " implementation=" << solver.info(nbody::Dimension::Two).name
              << " finite=" << (world.diagnostics().finite ? "true" : "false") << '\n';
    return world.isValid() ? 0 : 1;
}

}

int main(int argc, char** argv) {
    const std::string_view mode = argc > 1 ? std::string_view(argv[1]) : std::string_view{};
    const bool isolated_mode = mode == "scalar-bh" || mode == "simd-bh"
        || mode == "scalar-full" || mode == "simd-full";
    if (isolated_mode) {
        const std::size_t body_count = argc > 2 ? std::stoull(argv[2]) : 2000;
        const int steps = argc > 3 ? std::stoi(argv[3]) : 20;
        if (mode == "scalar-bh") {
            nbody::ScalarBarnesHutSolver solver;
            return runIsolatedSolver(mode, body_count, steps, solver, nbody::ForceModel::BarnesHut);
        }
        if (mode == "simd-bh") {
            nbody::SimdBarnesHutSolver solver;
            return runIsolatedSolver(mode, body_count, steps, solver, nbody::ForceModel::BarnesHut);
        }
        if (mode == "scalar-full") {
            nbody::ScalarFullSolver solver;
            return runIsolatedSolver(mode, body_count, steps, solver, nbody::ForceModel::Full);
        }
        nbody::SimdFullSolver solver;
        return runIsolatedSolver(mode, body_count, steps, solver, nbody::ForceModel::Full);
    }

    const bool solver_comparison = argc > 1 && std::string_view(argv[1]) == "compare";
    if (solver_comparison) {
        constexpr std::size_t body_count = 2000;
        constexpr int steps = 20;
        const nbody::Dimension dimension = nbody::Dimension::Two;
        nbody::WorldState full_world = nbody::WorldState::deterministic(body_count, dimension, 42);
        nbody::WorldState barnes_world = nbody::WorldState::deterministic(body_count, dimension, 42);
        nbody::SimulationParameters parameters;
        parameters.dimension = dimension;
        parameters.gravitational_constant = 0.1;
        parameters.timestep = 0.001;
        parameters.collision.model = nbody::CollisionModel::Transparent;
        nbody::SimulationParameters barnes_parameters = parameters;
        barnes_parameters.solver.force_model = nbody::ForceModel::BarnesHut;
        nbody::ScalarFullSolver full_solver;
        nbody::ScalarBarnesHutSolver barnes_solver;

        const auto full_start = std::chrono::steady_clock::now();
        for (int step = 0; step < steps; ++step) full_solver.step(full_world, parameters);
        const auto full_elapsed = std::chrono::steady_clock::now() - full_start;

        const auto barnes_start = std::chrono::steady_clock::now();
        for (int step = 0; step < steps; ++step) barnes_solver.step(barnes_world, barnes_parameters);
        const auto barnes_elapsed = std::chrono::steady_clock::now() - barnes_start;

        const double full_ms = std::chrono::duration<double, std::milli>(full_elapsed).count();
        const double barnes_ms = std::chrono::duration<double, std::milli>(barnes_elapsed).count();
        std::cout << "solver comparison dimension=2D steps=" << steps
                  << " bodies=" << body_count
                  << " full_ms=" << full_ms
                  << " barnes_hut_ms=" << barnes_ms
                  << " speedup=" << (barnes_ms > 0.0 ? full_ms / barnes_ms : 0.0) << '\n';
        return 0;
    }

    const bool collision_comparison = argc > 1 && std::string_view(argv[1]) == "compare-collision";
    if (collision_comparison) {
        constexpr std::size_t body_count = 2000;
        constexpr int steps = 20;
        const nbody::Dimension dimension = nbody::Dimension::Two;
        nbody::WorldState full_world = nbody::WorldState::deterministic(body_count, dimension, 42);
        nbody::WorldState barnes_world = nbody::WorldState::deterministic(body_count, dimension, 42);
        nbody::SimulationParameters parameters;
        parameters.dimension = dimension;
        parameters.gravitational_constant = 0.1;
        parameters.timestep = 0.001;
        parameters.collision.model = nbody::CollisionModel::HardBody;
        nbody::SimulationParameters barnes_parameters = parameters;
        barnes_parameters.solver.force_model = nbody::ForceModel::BarnesHut;
        nbody::ScalarFullSolver full_solver;
        nbody::ScalarBarnesHutSolver barnes_solver;

        const auto full_start = std::chrono::steady_clock::now();
        for (int step = 0; step < steps; ++step) full_solver.step(full_world, parameters);
        const auto full_elapsed = std::chrono::steady_clock::now() - full_start;

        const auto barnes_start = std::chrono::steady_clock::now();
        for (int step = 0; step < steps; ++step) barnes_solver.step(barnes_world, barnes_parameters);
        const auto barnes_elapsed = std::chrono::steady_clock::now() - barnes_start;

        const double full_ms = std::chrono::duration<double, std::milli>(full_elapsed).count();
        const double barnes_ms = std::chrono::duration<double, std::milli>(barnes_elapsed).count();
        std::cout << "solver collision comparison dimension=2D steps=" << steps
                  << " bodies=" << body_count
                  << " full_ms=" << full_ms
                  << " barnes_hut_ms=" << barnes_ms
                  << " speedup=" << (barnes_ms > 0.0 ? full_ms / barnes_ms : 0.0)
                  << " full_contacts=" << full_world.collisionEvents().size()
                  << " barnes_hut_contacts=" << barnes_world.collisionEvents().size() << '\n';
        return 0;
    }

    const bool simd_comparison = argc > 1 && std::string_view(argv[1]) == "compare-simd";
    if (simd_comparison) {
        constexpr std::size_t body_count = 2000;
        constexpr int steps = 20;
        nbody::WorldState scalar_world = nbody::WorldState::deterministic(body_count, nbody::Dimension::Two, 42);
        nbody::WorldState simd_world = nbody::WorldState::deterministic(body_count, nbody::Dimension::Two, 42);
        nbody::SimulationParameters parameters;
        parameters.dimension = nbody::Dimension::Two;
        parameters.gravitational_constant = 0.1;
        parameters.timestep = 0.001;
        parameters.collision.model = nbody::CollisionModel::Transparent;
        nbody::ScalarFullSolver scalar_solver;
        nbody::SimdFullSolver simd_solver;
        const auto scalar_start = std::chrono::steady_clock::now();
        for (int step = 0; step < steps; ++step) scalar_solver.step(scalar_world, parameters);
        const auto scalar_elapsed = std::chrono::steady_clock::now() - scalar_start;
        const auto simd_start = std::chrono::steady_clock::now();
        for (int step = 0; step < steps; ++step) simd_solver.step(simd_world, parameters);
        const auto simd_elapsed = std::chrono::steady_clock::now() - simd_start;
        const double scalar_ms = std::chrono::duration<double, std::milli>(scalar_elapsed).count();
        const double simd_ms = std::chrono::duration<double, std::milli>(simd_elapsed).count();
        std::cout << "SIMD comparison dimension=2D steps=" << steps
                  << " bodies=" << body_count
                  << " scalar_ms=" << scalar_ms << " simd_ms=" << simd_ms
                  << " speedup=" << (simd_ms > 0.0 ? scalar_ms / simd_ms : 0.0)
                  << " implementation=" << simd_solver.info(nbody::Dimension::Two).name << '\n';
        return 0;
    }

    const bool simd_barnes_comparison = argc > 1 && std::string_view(argv[1]) == "compare-simd-bh";
    if (simd_barnes_comparison) {
        const std::size_t body_count = argc > 2 ? std::stoull(argv[2]) : 2000;
        const int steps = argc > 3 ? std::stoi(argv[3]) : 20;
        nbody::WorldState scalar_world = nbody::WorldState::deterministic(body_count, nbody::Dimension::Two, 42);
        nbody::WorldState simd_world = nbody::WorldState::deterministic(body_count, nbody::Dimension::Two, 42);
        nbody::SimulationParameters parameters;
        parameters.dimension = nbody::Dimension::Two;
        parameters.gravitational_constant = 0.1;
        parameters.timestep = 0.001;
        parameters.collision.model = nbody::CollisionModel::Transparent;
        parameters.solver.force_model = nbody::ForceModel::BarnesHut;
        nbody::ScalarBarnesHutSolver scalar_solver;
        nbody::SimdBarnesHutSolver simd_solver;
        const auto scalar_start = std::chrono::steady_clock::now();
        for (int step = 0; step < steps; ++step) scalar_solver.step(scalar_world, parameters);
        const auto scalar_elapsed = std::chrono::steady_clock::now() - scalar_start;
        const auto simd_start = std::chrono::steady_clock::now();
        for (int step = 0; step < steps; ++step) simd_solver.step(simd_world, parameters);
        const auto simd_elapsed = std::chrono::steady_clock::now() - simd_start;
        const double scalar_ms = std::chrono::duration<double, std::milli>(scalar_elapsed).count();
        const double simd_ms = std::chrono::duration<double, std::milli>(simd_elapsed).count();
        std::cout << "SIMD Barnes-Hut comparison dimension=2D steps=" << steps
                  << " bodies=" << body_count
                  << " scalar_ms=" << scalar_ms << " simd_ms=" << simd_ms
                  << " speedup=" << (simd_ms > 0.0 ? scalar_ms / simd_ms : 0.0)
                  << " implementation=" << simd_solver.info(nbody::Dimension::Two).name << '\n';
        return 0;
    }

    const bool interference_run = argc > 1 && std::string_view(argv[1]) == "interference";
    if (interference_run) {
        constexpr std::size_t body_count = 250;
        constexpr int steps_per_phase = 10;
        nbody::WorldState world = nbody::WorldState::deterministic(body_count, nbody::Dimension::Two, 42);
        nbody::SimulationParameters parameters;
        parameters.dimension = nbody::Dimension::Two;
        parameters.gravitational_constant = 0.1;
        parameters.timestep = 0.001;
        parameters.collision.model = nbody::CollisionModel::Transparent;
        nbody::ScalarBarnesHutSolver solver;
        const auto start = std::chrono::steady_clock::now();
        for (int step = 0; step < steps_per_phase; ++step) solver.step(world, parameters);
        parameters.collision.model = nbody::CollisionModel::HardBody;
        for (int step = 0; step < steps_per_phase; ++step) solver.step(world, parameters);
        parameters.collision.model = nbody::CollisionModel::Transparent;
        for (int step = 0; step < steps_per_phase; ++step) solver.step(world, parameters);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        std::cout << "mid-run collision switch solver=Barnes-Hut phases=Transparent/HardBody/Transparent"
                  << " bodies=" << body_count << " steps=" << steps_per_phase * 3
                  << " elapsed_ms=" << std::chrono::duration<double, std::milli>(elapsed).count()
                  << " finite=" << (world.diagnostics().finite ? "true" : "false") << '\n';
        return world.isValid() ? 0 : 1;
    }

    const bool collision_benchmark = argc > 1 && std::string_view(argv[1]) == "collision";
    if (collision_benchmark) {
        constexpr std::size_t body_count = 5000;
        constexpr int steps = 25;
        nbody::WorldState world = nbody::WorldState::deterministic(body_count, nbody::Dimension::Two, 42);
        nbody::CollisionSettings collision;
        collision.model = nbody::CollisionModel::HardBody;
        const auto start = std::chrono::steady_clock::now();
        for (int step = 0; step < steps; ++step) {
            nbody::CollisionSystem::resolveContacts(world, collision, 0.0);
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        std::cout << "collision broad-phase dimension=2D"
                  << " steps=" << steps << " bodies=" << body_count
                  << " elapsed_ms=" << std::chrono::duration<double, std::milli>(elapsed).count()
                  << " last_contacts=" << world.collisionEvents().size() << '\n';
        return 0;
    }

    const bool three_dimensional = argc > 1 && (std::string_view(argv[1]) == "3" || std::string_view(argv[1]) == "3D");
    const nbody::Dimension dimension = three_dimensional ? nbody::Dimension::Three : nbody::Dimension::Two;
    nbody::WorldState world = nbody::WorldState::deterministic(1000, dimension, 42);
    nbody::SimulationParameters parameters;
    parameters.dimension = dimension;
    parameters.timestep = 0.01;
    nbody::ScalarFullSolver solver;
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < 100; ++step) solver.step(world, parameters);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "greenfield scaffold dimension=" << (three_dimensional ? "3D" : "2D")
              << " steps=100 bodies=" << world.bodyCount()
              << " elapsed_ms=" << std::chrono::duration<double, std::milli>(elapsed).count() << '\n';
}
