#include "simulation/world_state.hpp"
#include "simulation/solvers/scalar_full_solver.hpp"
#include "simulation/solvers/scalar_barnes_hut_solver.hpp"
#include "simulation/collision_system.hpp"

#include <chrono>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
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
