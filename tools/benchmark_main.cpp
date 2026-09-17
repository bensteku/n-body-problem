#include "simulation/world_state.hpp"
#include "simulation/solvers/scalar_full_solver.hpp"

#include <chrono>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
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
              << " steps=100 bodies=" << world.bodies().size()
              << " elapsed_ms=" << std::chrono::duration<double, std::milli>(elapsed).count() << '\n';
}
