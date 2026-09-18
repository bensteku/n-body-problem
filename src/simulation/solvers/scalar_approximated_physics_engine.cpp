#include "simulation/solvers/scalar_approximated_physics_engine.hpp"

#include "simulation/boundary_system.hpp"
#include "simulation/scalar_collision_system.hpp"

#include <algorithm>
#include <chrono>
#include <thread>
#include <stdexcept>

namespace nbody {

void ScalarApproximatedPhysicsEngine::calculateAccelerations(const WorldState& world,
    const SimulationParameters& parameters, std::vector<Vec3>& output) const {
    configureThreading(parameters.solver);
    output.assign(world.bodyCount(), Vec3{});
    const BarnesHutSettings& settings = parameters.solver.barnes_hut;
    if (!tree_ || tree_->dimension() != world.dimension()
        || tree_->leafCapacity() != settings.leaf_capacity
        || tree_->maximumDepth() != settings.maximum_depth) {
        tree_ = std::make_unique<BarnesHutTree>(world.dimension(), settings.leaf_capacity,
                                                settings.maximum_depth);
    }
    tree_->rebuild(world.bodyStorage());
    const double opening_angle = std::clamp(settings.opening_angle, 0.0, 10.0);
    if (parameters.solver.threading == ThreadingMode::MultiThreaded) {
        executor_.parallelFor(world.bodyCount(), [&](std::size_t worker,
                                                     std::size_t begin, std::size_t end) {
            std::vector<std::size_t>& traversal_stack = worker_traversal_stacks_[worker];
            for (std::size_t index = begin; index < end; ++index) {
                if (world.body(index).is_static()) continue;
                output[index] = tree_->accelerationOn(index, world.bodyStorage(),
                    parameters.gravitational_constant, parameters.softening_length,
                    opening_angle, traversal_stack);
            }
        });
        return;
    }
    for (std::size_t index = 0; index < world.bodyCount(); ++index) {
        if (world.body(index).is_static()) continue;
        output[index] = tree_->accelerationOn(index, world.bodyStorage(),
            parameters.gravitational_constant, parameters.softening_length,
            opening_angle, traversal_stack_);
    }
}

void ScalarApproximatedPhysicsEngine::configureThreading(
    const SolverConfiguration& configuration) const {
    const bool multithreaded = configuration.threading == ThreadingMode::MultiThreaded;
    std::size_t worker_count = configuration.worker_count;
    if (multithreaded && worker_count == 0) {
        worker_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }
    if (!multithreaded) worker_count = 0;
    if (worker_count == configured_worker_count_
        && configuration.threading == configured_threading_) return;

    executor_.setWorkerCount(worker_count);
    worker_traversal_stacks_.resize(worker_count);
    configured_worker_count_ = worker_count;
    configured_threading_ = configuration.threading;
}

PhysicsStepResult ScalarApproximatedPhysicsEngine::step(WorldState& world,
    const SimulationParameters& parameters) {
    if (parameters.integrator != Integrator::VelocityVerlet) {
        return {PhysicsStepStatus::Skipped, world.time(), world.bodyCount()};
    }
    if (parameters.dimension != world.dimension()) {
        throw std::invalid_argument("Simulation parameter dimension does not match WorldState dimension");
    }

    const double timestep = parameters.timestep;
    const auto step_start = std::chrono::steady_clock::now();
    PhysicsPhaseTimings timings;
    auto phase_start = step_start;
    calculateAccelerations(world, parameters, initial_accelerations_);
    timings.force_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phase_start).count();
    phase_start = std::chrono::steady_clock::now();
    world.integratePositions(initial_accelerations_, timestep);
    timings.integration_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phase_start).count();

    phase_start = std::chrono::steady_clock::now();
    calculateAccelerations(world, parameters, final_accelerations_);
    timings.force_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phase_start).count();
    phase_start = std::chrono::steady_clock::now();
    world.integrateVelocities(initial_accelerations_, final_accelerations_, timestep);
    timings.integration_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phase_start).count();

    phase_start = std::chrono::steady_clock::now();
    ScalarCollisionSystem::resolveContacts(world, parameters.collision,
                                           parameters.gravitational_constant,
                                           collision_workspace_);
    timings.collision_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phase_start).count();
    phase_start = std::chrono::steady_clock::now();
    BoundarySystem::resolve(world, parameters.boundary);
    timings.boundary_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phase_start).count();
    phase_start = std::chrono::steady_clock::now();
    ScalarCollisionSystem::applyDeferredOutcomes(world, parameters.collision);
    timings.deferred_outcomes_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phase_start).count();
    world.advanceTime(timestep);
    timings.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - step_start).count();
    return {PhysicsStepStatus::Advanced, world.time(), world.bodyCount(), {}, timings};
}

PhysicsEngineInfo ScalarApproximatedPhysicsEngine::info(Dimension dimension) const {
    return {ComputeBackend::Scalar, ForceModel::BarnesHut, dimension,
            configured_threading_ == ThreadingMode::MultiThreaded
                ? "Scalar Barnes-Hut (MT)" : "Scalar Barnes-Hut", SolverKind::Approximated};
}

}
