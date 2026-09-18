#include "simulation/solvers/simd_approximated_physics_engine.hpp"

#include "simulation/boundary_system.hpp"
#include "simulation/simd_collision_system.hpp"
#include "simulation/solvers/scalar_approximated_physics_engine.hpp"
#include "simulation/solvers/simd_barnes_hut_kernel.hpp"

#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <thread>

namespace nbody {

SimdApproximatedPhysicsEngine::SimdApproximatedPhysicsEngine()
    : capabilities_(detectSimdCapabilities()) {}

void SimdApproximatedPhysicsEngine::calculateAccelerations(const WorldState& world,
    const SimulationParameters& parameters, std::vector<Vec3>& output) const {
    const BarnesHutSettings& settings = parameters.solver.barnes_hut;
    if (!tree_ || tree_->dimension() != world.dimension()
        || tree_->leafCapacity() != settings.leaf_capacity
        || tree_->maximumDepth() != settings.maximum_depth) {
        tree_ = std::make_unique<BarnesHutTree>(world.dimension(), settings.leaf_capacity,
                                                settings.maximum_depth);
    }
    tree_->rebuild(world.bodyStorage());
    configureThreading(parameters.solver);
    output.assign(world.bodyCount(), Vec3{});
    if (parameters.solver.threading == ThreadingMode::MultiThreaded) {
        executor_.parallelFor(world.bodyCount(), [&](std::size_t worker,
                                                     std::size_t begin, std::size_t end) {
            calculateAvx2BarnesHutAccelerationsRange(world, parameters, *tree_, output,
                                                      worker_traversal_stacks_[worker], begin, end);
        });
    } else {
        calculateAvx2BarnesHutAccelerationsRange(world, parameters, *tree_, output,
                                                  traversal_stack_, 0, world.bodyCount());
    }
}

void SimdApproximatedPhysicsEngine::configureThreading(
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

PhysicsStepResult SimdApproximatedPhysicsEngine::step(WorldState& world,
    const SimulationParameters& parameters) {
    if (!capabilities_.avx2) {
        ScalarApproximatedPhysicsEngine fallback;
        return fallback.step(world, parameters);
    }
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
    SimdCollisionSystem::resolveContacts(world, parameters.collision,
                                         parameters.gravitational_constant,
                                         simd_collision_workspace_,
                                         scalar_collision_workspace_);
    timings.collision_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phase_start).count();
    phase_start = std::chrono::steady_clock::now();
    BoundarySystem::resolve(world, parameters.boundary);
    timings.boundary_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phase_start).count();
    phase_start = std::chrono::steady_clock::now();
    SimdCollisionSystem::applyDeferredOutcomes(world, parameters.collision);
    timings.deferred_outcomes_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - phase_start).count();
    world.advanceTime(timestep);
    timings.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - step_start).count();
    return {PhysicsStepStatus::Advanced, world.time(), world.bodyCount(), {}, timings};
}

PhysicsEngineInfo SimdApproximatedPhysicsEngine::info(Dimension dimension) const {
    return {ComputeBackend::SIMD, ForceModel::BarnesHut, dimension,
            configured_threading_ == ThreadingMode::MultiThreaded
                ? (capabilities_.avx2 ? "SIMD Barnes-Hut MT (AVX2)" : "SIMD Barnes-Hut MT (scalar fallback)")
                : (capabilities_.avx2 ? "SIMD Barnes-Hut (AVX2)" : "SIMD Barnes-Hut (scalar fallback)"),
            SolverKind::Approximated};
}

}
