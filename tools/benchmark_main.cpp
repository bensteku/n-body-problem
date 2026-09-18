#include "simulation/world_state.hpp"
#include "simulation/solvers/scalar_full_physics_engine.hpp"
#include "simulation/solvers/scalar_approximated_physics_engine.hpp"
#include "simulation/solvers/simd_full_physics_engine.hpp"
#include "simulation/solvers/simd_approximated_physics_engine.hpp"
#include "simulation/scalar_collision_system.hpp"
#include "simulation/simd_collision_system.hpp"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

bool worldsNumericallyClose(const nbody::WorldState& first, const nbody::WorldState& second,
                            double tolerance = 1e-10) {
    if (first.bodyCount() != second.bodyCount() || first.dimension() != second.dimension()) return false;
    for (std::size_t index = 0; index < first.bodyCount(); ++index) {
        const nbody::ConstBodyView first_body = first.body(index);
        const nbody::ConstBodyView second_body = second.body(index);
        const auto close = [tolerance](double left, double right) {
            return std::abs(left - right) <= tolerance
                + tolerance * std::max(std::abs(left), std::abs(right));
        };
        if (!close(first_body.position.x, second_body.position.x)
            || !close(first_body.position.y, second_body.position.y)
            || !close(first_body.position.z, second_body.position.z)
            || !close(first_body.velocity.x, second_body.velocity.x)
            || !close(first_body.velocity.y, second_body.velocity.y)
            || !close(first_body.velocity.z, second_body.velocity.z)) return false;
    }
    return true;
}

template <typename Solver>
int runIsolatedSolver(std::string_view label, std::size_t body_count, int steps,
                      Solver& solver, nbody::ForceModel force_model,
                      nbody::ThreadingMode threading = nbody::ThreadingMode::SingleThreaded,
                      std::size_t worker_count = 0) {
    nbody::WorldState world = nbody::WorldState::deterministic(body_count, nbody::Dimension::Two, 42);
    nbody::SimulationParameters parameters;
    parameters.dimension = nbody::Dimension::Two;
    parameters.gravitational_constant = 0.1;
    parameters.timestep = 0.001;
    parameters.collision.model = nbody::CollisionModel::Transparent;
    parameters.solver.force_model = force_model;
    parameters.solver.threading = threading;
    parameters.solver.worker_count = worker_count;

    const auto start = std::chrono::steady_clock::now();
    nbody::PhysicsPhaseTimings timings;
    for (int step = 0; step < steps; ++step) {
        const nbody::PhysicsStepResult result = solver.step(world, parameters);
        timings.force_ms += result.timings.force_ms;
        timings.integration_ms += result.timings.integration_ms;
        timings.collision_ms += result.timings.collision_ms;
        timings.boundary_ms += result.timings.boundary_ms;
        timings.deferred_outcomes_ms += result.timings.deferred_outcomes_ms;
        timings.total_ms += result.timings.total_ms;
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
    std::cout << "isolated solver=" << label
              << " dimension=2D steps=" << steps
              << " bodies=" << body_count
              << " elapsed_ms=" << elapsed_ms
              << " force_ms=" << timings.force_ms
              << " integration_ms=" << timings.integration_ms
              << " collision_ms=" << timings.collision_ms
              << " boundary_ms=" << timings.boundary_ms
              << " deferred_ms=" << timings.deferred_outcomes_ms
              << " accounted_total_ms=" << timings.total_ms
              << " threading=" << (threading == nbody::ThreadingMode::MultiThreaded ? "MT" : "single")
              << " workers=" << worker_count
              << " implementation=" << solver.info(nbody::Dimension::Two).name
              << " finite=" << (world.diagnostics().finite ? "true" : "false") << '\n';
    return world.isValid() ? 0 : 1;
}

struct ScenarioDefaults {
    std::size_t bodies;
    int steps;
};

ScenarioDefaults scenarioDefaults(std::string_view name) {
    if (name == "sparse") return {256, 20};
    if (name == "dense") return {2000, 5};
    if (name == "large" || name == "large-bh") return {100000, 1};
    throw std::invalid_argument("unknown scenario; expected sparse, dense, or large");
}

int runConfiguredScenario(std::string_view scenario, std::string_view backend,
                          std::string_view force_model, std::size_t body_count, int steps,
                          bool multithreaded, std::size_t worker_count) {
    const nbody::ForceModel force = force_model == "BaHu" || force_model == "BarnesHut"
        ? nbody::ForceModel::BarnesHut : nbody::ForceModel::Full;
    const nbody::ThreadingMode threading = multithreaded
        ? nbody::ThreadingMode::MultiThreaded : nbody::ThreadingMode::SingleThreaded;
    const bool simd = backend == "SIMD";
    const std::string label = std::string(scenario) + ":" + std::string(backend)
        + ":" + std::string(force_model);

    // GPU is currently an explicitly reported scalar fallback. Keeping it in this
    // harness makes capability coverage visible without pretending it is a GPU run.
    if (!simd && force == nbody::ForceModel::Full) {
        nbody::ScalarFullPhysicsEngine solver;
        return runIsolatedSolver(label, body_count, steps, solver, force, threading, worker_count);
    }
    if (!simd) {
        nbody::ScalarApproximatedPhysicsEngine solver;
        return runIsolatedSolver(label, body_count, steps, solver, force, threading, worker_count);
    }
    if (force == nbody::ForceModel::Full) {
        nbody::SimdFullPhysicsEngine solver;
        return runIsolatedSolver(label, body_count, steps, solver, force, threading, worker_count);
    }
    nbody::SimdApproximatedPhysicsEngine solver;
    return runIsolatedSolver(label, body_count, steps, solver, force, threading, worker_count);
}

int runLargeBarnesHutComparison(std::size_t body_count, int steps) {
    constexpr nbody::Dimension dimension = nbody::Dimension::Two;
    nbody::WorldState scalar_world = nbody::WorldState::deterministic(body_count, dimension, 42);
    nbody::WorldState simd_world = nbody::WorldState::deterministic(body_count, dimension, 42);
    nbody::SimulationParameters parameters;
    parameters.dimension = dimension;
    parameters.gravitational_constant = 0.1;
    parameters.timestep = 0.001;
    parameters.collision.model = nbody::CollisionModel::Transparent;
    parameters.solver.force_model = nbody::ForceModel::BarnesHut;
    nbody::ScalarApproximatedPhysicsEngine scalar_solver;
    nbody::SimdApproximatedPhysicsEngine simd_solver;

    const auto scalar_start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) scalar_solver.step(scalar_world, parameters);
    const auto scalar_elapsed = std::chrono::steady_clock::now() - scalar_start;

    const auto simd_start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) simd_solver.step(simd_world, parameters);
    const auto simd_elapsed = std::chrono::steady_clock::now() - simd_start;

    const double scalar_ms = std::chrono::duration<double, std::milli>(scalar_elapsed).count();
    const double simd_ms = std::chrono::duration<double, std::milli>(simd_elapsed).count();
    std::cout << "large Barnes-Hut comparison dimension=2D"
              << " steps=" << steps
              << " bodies=" << body_count
              << " scalar_ms=" << scalar_ms
              << " simd_ms=" << simd_ms
              << " speedup=" << (simd_ms > 0.0 ? scalar_ms / simd_ms : 0.0)
              << " scalar_finite=" << (scalar_world.diagnostics().finite ? "true" : "false")
              << " simd_finite=" << (simd_world.diagnostics().finite ? "true" : "false")
              << '\n';
    return scalar_world.isValid() && simd_world.isValid() ? 0 : 1;
}

int runScalarThreadingComparison(std::size_t body_count, int steps, std::size_t worker_count) {
    nbody::WorldState single_world = nbody::WorldState::deterministic(body_count,
                                                                        nbody::Dimension::Two, 42);
    nbody::WorldState multi_world = nbody::WorldState::deterministic(body_count,
                                                                       nbody::Dimension::Two, 42);
    nbody::SimulationParameters single_parameters;
    single_parameters.dimension = nbody::Dimension::Two;
    single_parameters.gravitational_constant = 0.1;
    single_parameters.timestep = 0.001;
    single_parameters.collision.model = nbody::CollisionModel::Transparent;
    single_parameters.solver.threading = nbody::ThreadingMode::SingleThreaded;
    nbody::SimulationParameters multi_parameters = single_parameters;
    multi_parameters.solver.threading = nbody::ThreadingMode::MultiThreaded;
    multi_parameters.solver.worker_count = worker_count;
    nbody::ScalarFullPhysicsEngine single_solver;
    nbody::ScalarFullPhysicsEngine multi_solver;

    const auto single_start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) single_solver.step(single_world, single_parameters);
    const auto single_elapsed = std::chrono::steady_clock::now() - single_start;
    const auto multi_start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) multi_solver.step(multi_world, multi_parameters);
    const auto multi_elapsed = std::chrono::steady_clock::now() - multi_start;
    const double single_ms = std::chrono::duration<double, std::milli>(single_elapsed).count();
    const double multi_ms = std::chrono::duration<double, std::milli>(multi_elapsed).count();
    std::cout << "Scalar Full threading comparison dimension=2D"
              << " steps=" << steps << " bodies=" << body_count
              << " workers=" << worker_count
              << " single_ms=" << single_ms << " multi_ms=" << multi_ms
              << " speedup=" << (multi_ms > 0.0 ? single_ms / multi_ms : 0.0)
              << " parity=" << (worldsNumericallyClose(single_world, multi_world) ? "true" : "false")
              << '\n';
    return single_world.isValid() && multi_world.isValid() ? 0 : 1;
}

int runScalarBarnesThreadingComparison(std::size_t body_count, int steps,
                                       std::size_t worker_count) {
    nbody::WorldState single_world = nbody::WorldState::deterministic(body_count,
                                                                        nbody::Dimension::Two, 42);
    nbody::WorldState multi_world = nbody::WorldState::deterministic(body_count,
                                                                       nbody::Dimension::Two, 42);
    nbody::SimulationParameters single_parameters;
    single_parameters.dimension = nbody::Dimension::Two;
    single_parameters.gravitational_constant = 0.1;
    single_parameters.timestep = 0.001;
    single_parameters.collision.model = nbody::CollisionModel::Transparent;
    single_parameters.solver.force_model = nbody::ForceModel::BarnesHut;
    single_parameters.solver.kind = nbody::SolverKind::Approximated;
    single_parameters.solver.threading = nbody::ThreadingMode::SingleThreaded;
    nbody::SimulationParameters multi_parameters = single_parameters;
    multi_parameters.solver.threading = nbody::ThreadingMode::MultiThreaded;
    multi_parameters.solver.worker_count = worker_count;
    nbody::ScalarApproximatedPhysicsEngine single_solver;
    nbody::ScalarApproximatedPhysicsEngine multi_solver;

    const auto single_start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) single_solver.step(single_world, single_parameters);
    const auto single_elapsed = std::chrono::steady_clock::now() - single_start;
    const auto multi_start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) multi_solver.step(multi_world, multi_parameters);
    const auto multi_elapsed = std::chrono::steady_clock::now() - multi_start;
    const double single_ms = std::chrono::duration<double, std::milli>(single_elapsed).count();
    const double multi_ms = std::chrono::duration<double, std::milli>(multi_elapsed).count();
    std::cout << "Scalar Barnes-Hut threading comparison dimension=2D"
              << " steps=" << steps << " bodies=" << body_count
              << " workers=" << worker_count
              << " single_ms=" << single_ms << " multi_ms=" << multi_ms
              << " speedup=" << (multi_ms > 0.0 ? single_ms / multi_ms : 0.0)
              << " parity=" << (worldsNumericallyClose(single_world, multi_world) ? "true" : "false")
              << '\n';
    return single_world.isValid() && multi_world.isValid() ? 0 : 1;
}

int runSimdThreadingComparison(std::size_t body_count, int steps,
                               std::size_t worker_count, bool barnes_hut) {
    nbody::WorldState single_world = nbody::WorldState::deterministic(body_count,
                                                                        nbody::Dimension::Two, 42);
    nbody::WorldState multi_world = nbody::WorldState::deterministic(body_count,
                                                                       nbody::Dimension::Two, 42);
    nbody::SimulationParameters single_parameters;
    single_parameters.dimension = nbody::Dimension::Two;
    single_parameters.gravitational_constant = 0.1;
    single_parameters.timestep = 0.001;
    single_parameters.collision.model = nbody::CollisionModel::Transparent;
    single_parameters.solver.force_model = barnes_hut
        ? nbody::ForceModel::BarnesHut : nbody::ForceModel::Full;
    single_parameters.solver.kind = barnes_hut
        ? nbody::SolverKind::Approximated : nbody::SolverKind::Full;
    single_parameters.solver.threading = nbody::ThreadingMode::SingleThreaded;
    nbody::SimulationParameters multi_parameters = single_parameters;
    multi_parameters.solver.threading = nbody::ThreadingMode::MultiThreaded;
    multi_parameters.solver.worker_count = worker_count;
    nbody::SimdApproximatedPhysicsEngine simd_bh_single_solver;
    nbody::SimdApproximatedPhysicsEngine simd_bh_multi_solver;
    nbody::SimdFullPhysicsEngine simd_full_single_solver;
    nbody::SimdFullPhysicsEngine simd_full_multi_solver;

    const auto single_start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        if (barnes_hut) simd_bh_single_solver.step(single_world, single_parameters);
        else simd_full_single_solver.step(single_world, single_parameters);
    }
    const auto single_elapsed = std::chrono::steady_clock::now() - single_start;
    const auto multi_start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        if (barnes_hut) simd_bh_multi_solver.step(multi_world, multi_parameters);
        else simd_full_multi_solver.step(multi_world, multi_parameters);
    }
    const auto multi_elapsed = std::chrono::steady_clock::now() - multi_start;
    const double single_ms = std::chrono::duration<double, std::milli>(single_elapsed).count();
    const double multi_ms = std::chrono::duration<double, std::milli>(multi_elapsed).count();
    std::cout << (barnes_hut ? "SIMD Barnes-Hut" : "SIMD Full")
              << " threading comparison dimension=2D"
              << " steps=" << steps << " bodies=" << body_count
              << " workers=" << worker_count
              << " single_ms=" << single_ms << " multi_ms=" << multi_ms
              << " speedup=" << (multi_ms > 0.0 ? single_ms / multi_ms : 0.0)
              << " parity=" << (worldsNumericallyClose(single_world, multi_world) ? "true" : "false")
              << '\n';
    return single_world.isValid() && multi_world.isValid() ? 0 : 1;
}

}

int main(int argc, char** argv) {
    const std::string_view mode = argc > 1 ? std::string_view(argv[1]) : std::string_view{};
    const bool fragmentation_benchmark = mode == "fragmentation";
    if (fragmentation_benchmark) {
        const std::size_t body_count = argc > 2 ? std::stoull(argv[2]) : 256;
        const std::size_t fragment_count = argc > 3 ? std::stoull(argv[3]) : 4;
        const std::size_t reserved_capacity = argc > 4 ? std::stoull(argv[4]) : 0;
        const bool absorption_enabled = argc > 5 && std::stoull(argv[5]) != 0;
        if (body_count == 0 || fragment_count < 2) {
            std::cerr << "fragmentation benchmark requires bodies > 0 and fragments >= 2\n";
            return 2;
        }

        nbody::WorldState world(nbody::Dimension::Two);
        if (reserved_capacity != 0) world.reserveBodies(reserved_capacity);
        for (std::size_t index = 0; index < body_count; ++index) {
            world.addBody({{}, {0.0, 0.0, 0.0}, {}, 1.0, 1.0, false});
        }

        nbody::CollisionSettings settings;
        settings.model = nbody::CollisionModel::HardBody;
        settings.minimum_fragments = fragment_count;
        settings.maximum_fragments = fragment_count;
        settings.maximum_fragment_count = body_count * fragment_count;
        settings.classifier.force_fragmentation = true;
        settings.classifier.absorption_enabled = absorption_enabled;
        // This benchmark measures fragmentation while hard-body mode is already active.
        world.setActiveCollisionModel(nbody::CollisionModel::HardBody);

        const double initial_mass = world.diagnostics().total_mass;
        const auto resolve_start = std::chrono::steady_clock::now();
        nbody::ScalarCollisionSystem::resolveContacts(world, settings, 0.0);
        const auto resolve_elapsed = std::chrono::steady_clock::now() - resolve_start;
        const auto apply_start = std::chrono::steady_clock::now();
        nbody::ScalarCollisionSystem::applyDeferredOutcomes(world, settings);
        const auto apply_elapsed = std::chrono::steady_clock::now() - apply_start;
        const double resolve_ms = std::chrono::duration<double, std::milli>(resolve_elapsed).count();
        const double apply_ms = std::chrono::duration<double, std::milli>(apply_elapsed).count();
        const double final_mass = world.diagnostics().total_mass;
        std::cout << "fragmentation benchmark dimension=2D"
                  << " initial_bodies=" << body_count
                  << " requested_fragments=" << fragment_count
                  << " reserved_capacity=" << reserved_capacity
                  << " absorption_enabled=" << (absorption_enabled ? "true" : "false")
                  << " collision_events=" << world.collisionEvents().size()
                  << " final_bodies=" << world.bodyCount()
                  << " growth=" << (world.bodyCount() - body_count)
                  << " resolve_ms=" << resolve_ms
                  << " apply_ms=" << apply_ms
                  << " total_ms=" << (resolve_ms + apply_ms)
                  << " mass_error=" << (final_mass - initial_mass)
                  << " finite=" << (world.diagnostics().finite ? "true" : "false")
                  << " valid=" << (world.isValid() ? "true" : "false") << '\n';
        return world.isValid() && world.bodyCount() == body_count * fragment_count
            && std::abs(final_mass - initial_mass) < 1e-10 ? 0 : 1;
    }

    if (mode == "scenario") {
        if (argc < 4) {
            std::cerr << "scenario requires <sparse|dense|large> <Scalar|SIMD|GPU> <Full|BaHu> "
                         "[bodies] [steps] [mt] [workers]\n";
            return 2;
        }
        const std::string_view scenario = argv[2];
        const std::string_view backend = argv[3];
        const std::string_view force_model = argc > 4 ? std::string_view(argv[4]) : "Full";
        if (backend != "Scalar" && backend != "SIMD" && backend != "GPU") {
            std::cerr << "scenario backend must be Scalar, SIMD, or GPU\n";
            return 2;
        }
        if (force_model != "Full" && force_model != "BaHu" && force_model != "BarnesHut") {
            std::cerr << "scenario force model must be Full or BaHu\n";
            return 2;
        }
        const ScenarioDefaults defaults = scenarioDefaults(scenario);
        const std::size_t body_count = argc > 5 ? std::stoull(argv[5]) : defaults.bodies;
        const int steps = argc > 6 ? std::stoi(argv[6]) : defaults.steps;
        const bool multithreaded = argc > 7 && std::stoull(argv[7]) != 0;
        const std::size_t worker_count = argc > 8 ? std::stoull(argv[8]) : 0;
        if (body_count == 0 || steps <= 0 || (multithreaded && worker_count == 0)) {
            std::cerr << "scenario requires bodies > 0, steps > 0, and workers > 0 in MT mode\n";
            return 2;
        }
        return runConfiguredScenario(scenario, backend, force_model, body_count, steps,
                                     multithreaded, worker_count);
    }
    const bool isolated_mode = mode == "scalar-bh" || mode == "simd-bh"
        || mode == "scalar-full" || mode == "simd-full";
    if (isolated_mode) {
        const std::size_t body_count = argc > 2 ? std::stoull(argv[2]) : 2000;
        const int steps = argc > 3 ? std::stoi(argv[3]) : 20;
        if (mode == "scalar-bh") {
            nbody::ScalarApproximatedPhysicsEngine solver;
            return runIsolatedSolver(mode, body_count, steps, solver, nbody::ForceModel::BarnesHut);
        }
        if (mode == "simd-bh") {
            nbody::SimdApproximatedPhysicsEngine solver;
            return runIsolatedSolver(mode, body_count, steps, solver, nbody::ForceModel::BarnesHut);
        }
        if (mode == "scalar-full") {
            nbody::ScalarFullPhysicsEngine solver;
            return runIsolatedSolver(mode, body_count, steps, solver, nbody::ForceModel::Full);
        }
        nbody::SimdFullPhysicsEngine solver;
        return runIsolatedSolver(mode, body_count, steps, solver, nbody::ForceModel::Full);
    }

    if (mode == "large-bh") {
        const std::size_t body_count = argc > 2 ? std::stoull(argv[2]) : 100000;
        const int steps = argc > 3 ? std::stoi(argv[3]) : 1;
        if (body_count == 0 || steps <= 0) {
            std::cerr << "large-bh requires bodies > 0 and steps > 0\n";
            return 2;
        }
        return runLargeBarnesHutComparison(body_count, steps);
    }

    if (mode == "scalar-mt-compare") {
        const std::size_t body_count = argc > 2 ? std::stoull(argv[2]) : 256;
        const int steps = argc > 3 ? std::stoi(argv[3]) : 5;
        const std::size_t worker_count = argc > 4 ? std::stoull(argv[4]) : 4;
        if (body_count == 0 || steps <= 0 || worker_count == 0) {
            std::cerr << "scalar-mt-compare requires bodies > 0, steps > 0, and workers > 0\n";
            return 2;
        }
        return runScalarThreadingComparison(body_count, steps, worker_count);
    }

    if (mode == "scalar-bh-mt-compare") {
        const std::size_t body_count = argc > 2 ? std::stoull(argv[2]) : 256;
        const int steps = argc > 3 ? std::stoi(argv[3]) : 5;
        const std::size_t worker_count = argc > 4 ? std::stoull(argv[4]) : 4;
        if (body_count == 0 || steps <= 0 || worker_count == 0) {
            std::cerr << "scalar-bh-mt-compare requires bodies > 0, steps > 0, and workers > 0\n";
            return 2;
        }
        return runScalarBarnesThreadingComparison(body_count, steps, worker_count);
    }

    if (mode == "simd-mt-compare" || mode == "simd-bh-mt-compare") {
        const std::size_t body_count = argc > 2 ? std::stoull(argv[2]) : 256;
        const int steps = argc > 3 ? std::stoi(argv[3]) : 5;
        const std::size_t worker_count = argc > 4 ? std::stoull(argv[4]) : 4;
        if (body_count == 0 || steps <= 0 || worker_count == 0) {
            std::cerr << "simd threading comparison requires bodies > 0, steps > 0, and workers > 0\n";
            return 2;
        }
        return runSimdThreadingComparison(body_count, steps, worker_count,
                                          mode == "simd-bh-mt-compare");
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
        nbody::ScalarFullPhysicsEngine full_solver;
        nbody::ScalarApproximatedPhysicsEngine barnes_solver;

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
        nbody::ScalarFullPhysicsEngine full_solver;
        nbody::ScalarApproximatedPhysicsEngine barnes_solver;

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
        nbody::ScalarFullPhysicsEngine scalar_solver;
        nbody::SimdFullPhysicsEngine simd_solver;
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
        nbody::ScalarApproximatedPhysicsEngine scalar_solver;
        nbody::SimdApproximatedPhysicsEngine simd_solver;
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
        nbody::ScalarApproximatedPhysicsEngine solver;
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
            nbody::ScalarCollisionSystem::resolveContacts(world, collision, 0.0);
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        std::cout << "collision broad-phase dimension=2D"
                  << " steps=" << steps << " bodies=" << body_count
                  << " elapsed_ms=" << std::chrono::duration<double, std::milli>(elapsed).count()
                  << " last_contacts=" << world.collisionEvents().size() << '\n';
        return 0;
    }

    const bool simd_collision_comparison = argc > 1 && std::string_view(argv[1]) == "collision-compare";
    if (simd_collision_comparison) {
        constexpr std::size_t body_count = 5000;
        constexpr int steps = 25;
        nbody::WorldState scalar_world = nbody::WorldState::deterministic(body_count, nbody::Dimension::Two, 42);
        nbody::WorldState simd_world = nbody::WorldState::deterministic(body_count, nbody::Dimension::Two, 42);
        nbody::CollisionSettings collision;
        collision.model = nbody::CollisionModel::HardBody;
        scalar_world.setActiveCollisionModel(nbody::CollisionModel::HardBody);
        simd_world.setActiveCollisionModel(nbody::CollisionModel::HardBody);

        const auto scalar_start = std::chrono::steady_clock::now();
        for (int step = 0; step < steps; ++step) {
            nbody::ScalarCollisionSystem::resolveContacts(scalar_world, collision, 0.0);
        }
        const auto scalar_elapsed = std::chrono::steady_clock::now() - scalar_start;

        const auto simd_start = std::chrono::steady_clock::now();
        for (int step = 0; step < steps; ++step) {
            nbody::SimdCollisionSystem::resolveContacts(simd_world, collision, 0.0);
        }
        const auto simd_elapsed = std::chrono::steady_clock::now() - simd_start;
        const double scalar_ms = std::chrono::duration<double, std::milli>(scalar_elapsed).count();
        const double simd_ms = std::chrono::duration<double, std::milli>(simd_elapsed).count();
        std::cout << "collision SIMD comparison dimension=2D"
                  << " steps=" << steps << " bodies=" << body_count
                  << " scalar_ms=" << scalar_ms << " simd_ms=" << simd_ms
                  << " speedup=" << (simd_ms > 0.0 ? scalar_ms / simd_ms : 0.0)
                  << " scalar_contacts=" << scalar_world.collisionEvents().size()
                  << " simd_contacts=" << simd_world.collisionEvents().size() << '\n';
        return scalar_world.isValid() && simd_world.isValid() ? 0 : 1;
    }

    const bool three_dimensional = argc > 1 && (std::string_view(argv[1]) == "3" || std::string_view(argv[1]) == "3D");
    const nbody::Dimension dimension = three_dimensional ? nbody::Dimension::Three : nbody::Dimension::Two;
    nbody::WorldState world = nbody::WorldState::deterministic(1000, dimension, 42);
    nbody::SimulationParameters parameters;
    parameters.dimension = dimension;
    parameters.timestep = 0.01;
    nbody::ScalarFullPhysicsEngine solver;
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < 100; ++step) solver.step(world, parameters);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "greenfield scaffold dimension=" << (three_dimensional ? "3D" : "2D")
              << " steps=100 bodies=" << world.bodyCount()
              << " elapsed_ms=" << std::chrono::duration<double, std::milli>(elapsed).count() << '\n';
}
