#include "simulation/solvers/simd_full_physics_engine.hpp"

#include "simulation/boundary_system.hpp"
#include "simulation/simd_collision_system.hpp"
#include "simulation/solvers/scalar_full_physics_engine.hpp"
#include "simulation/solvers/simd_avx2_kernel.hpp"

#include <stdexcept>
#include <algorithm>
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
    calculateAccelerations(world, parameters, initial_accelerations_);
    for (std::size_t index = 0; index < world.bodyCount(); ++index) {
        MutableBodyView body = world.mutableBody(index);
        if (body.is_static()) continue;
        body.position += body.velocity * timestep
            + initial_accelerations_[index] * (0.5 * timestep * timestep);
        if (world.dimension() == Dimension::Two) body.position.z = 0.0;
    }

    calculateAccelerations(world, parameters, final_accelerations_);
    for (std::size_t index = 0; index < world.bodyCount(); ++index) {
        MutableBodyView body = world.mutableBody(index);
        if (body.is_static()) continue;
        body.velocity += (initial_accelerations_[index] + final_accelerations_[index])
            * (0.5 * timestep);
        if (world.dimension() == Dimension::Two) body.velocity.z = 0.0;
    }

    SimdCollisionSystem::resolveContacts(world, parameters.collision,
                                         parameters.gravitational_constant,
                                         simd_collision_workspace_,
                                         scalar_collision_workspace_);
    BoundarySystem::resolve(world, parameters.boundary);
    SimdCollisionSystem::applyDeferredOutcomes(world, parameters.collision);
    world.advanceTime(timestep);
    return {PhysicsStepStatus::Advanced, world.time(), world.bodyCount()};
}

PhysicsEngineInfo SimdFullPhysicsEngine::info(Dimension dimension) const {
    return {ComputeBackend::SIMD, ForceModel::Full, dimension,
            configured_threading_ == ThreadingMode::MultiThreaded
                ? (capabilities_.avx2 ? "SIMD Full MT (AVX2)" : "SIMD Full MT (scalar fallback)")
                : (capabilities_.avx2 ? "SIMD Full (AVX2)" : "SIMD Full (scalar fallback)"),
            SolverKind::Full};
}

}
