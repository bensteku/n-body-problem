#include "simulation/solvers/simd_full_physics_engine.hpp"

#include "simulation/boundary_system.hpp"
#include "simulation/simd_collision_system.hpp"
#include "simulation/solvers/scalar_full_physics_engine.hpp"
#include "simulation/solvers/simd_avx2_kernel.hpp"

#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <thread>

namespace nbody {

SimdFullPhysicsEngine::SimdFullPhysicsEngine() : capabilities_(detectSimdCapabilities()) {}

void SimdFullPhysicsEngine::calculateAccelerations(const WorldState& world,
    const SimulationParameters& parameters, std::vector<Vec3>& output) const {
    if (capabilities_.avx2) {
        configureThreading(parameters.solver);
        world.bodyStorage().synchronizePositionComponents();
        output.assign(world.bodyCount(), Vec3{});
        if (parameters.solver.threading == ThreadingMode::MultiThreaded) {
            executor_.parallelFor(world.bodyCount(), [&](std::size_t,
                                                         std::size_t begin, std::size_t end) {
                calculateAvx2AccelerationsRange(world, parameters, output, begin, end);
            });
        } else {
            calculateAvx2AccelerationsRange(world, parameters, output, 0, world.bodyCount());
        }
        return;
    }
    output.assign(world.bodyCount(), Vec3{});
}

void SimdFullPhysicsEngine::configureThreading(const SolverConfiguration& configuration) const {
    const bool multithreaded = configuration.threading == ThreadingMode::MultiThreaded;
    std::size_t worker_count = configuration.worker_count;
    if (multithreaded && worker_count == 0) {
        worker_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }
    if (!multithreaded) worker_count = 0;
    if (worker_count == configured_worker_count_
        && configuration.threading == configured_threading_) return;
    executor_.setWorkerCount(worker_count);
    configured_worker_count_ = worker_count;
    configured_threading_ = configuration.threading;
}

PhysicsStepResult SimdFullPhysicsEngine::step(WorldState& world, const SimulationParameters& parameters) {
    if (!capabilities_.avx2) {
        ScalarFullPhysicsEngine fallback;
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

PhysicsEngineInfo SimdFullPhysicsEngine::info(Dimension dimension) const {
    return {ComputeBackend::SIMD, ForceModel::Full, dimension,
            configured_threading_ == ThreadingMode::MultiThreaded
                ? (capabilities_.avx2 ? "SIMD Full MT (AVX2)" : "SIMD Full MT (scalar fallback)")
                : (capabilities_.avx2 ? "SIMD Full (AVX2)" : "SIMD Full (scalar fallback)"),
            SolverKind::Full};
}

}
