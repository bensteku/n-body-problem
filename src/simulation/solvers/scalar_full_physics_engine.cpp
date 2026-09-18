#include "simulation/solvers/scalar_full_physics_engine.hpp"

#include "simulation/boundary_system.hpp"
#include "simulation/scalar_collision_system.hpp"

#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>
#include <stdexcept>

namespace nbody {

void ScalarFullPhysicsEngine::calculateAccelerations(const WorldState& world,
    const SimulationParameters& parameters, std::vector<Vec3>& accelerations) const {
    configureThreading(parameters.solver);
    if (parameters.solver.threading == ThreadingMode::MultiThreaded) {
        calculateAccelerationsMultithreaded(world, parameters, accelerations);
        return;
    }
    const std::size_t body_count = world.bodyCount();
    const Dimension dimension = world.dimension();
    accelerations.assign(body_count, Vec3{});
    const double softening_squared = parameters.softening_length * parameters.softening_length;

    for (std::size_t first = 0; first < body_count; ++first) {
        const ConstBodyView first_body = world.body(first);
        for (std::size_t second = first + 1; second < body_count; ++second) {
            const ConstBodyView second_body = world.body(second);
            const Vec3 displacement = second_body.position - first_body.position;
            const double distance_squared = displacement.lengthSquared(dimension) + softening_squared;
            if (distance_squared == 0.0) continue;

            const double distance = std::sqrt(distance_squared);
            const double inverse_distance_cubed = 1.0 / (distance_squared * distance);
            const double first_scale = parameters.gravitational_constant
                * second_body.mass * inverse_distance_cubed;
            const double second_scale = parameters.gravitational_constant
                * first_body.mass * inverse_distance_cubed;
            accelerations[first] += displacement * first_scale;
            accelerations[second] -= displacement * second_scale;
        }
    }
}

void ScalarFullPhysicsEngine::configureThreading(const SolverConfiguration& configuration) const {
    const bool multithreaded = configuration.threading == ThreadingMode::MultiThreaded;
    std::size_t worker_count = configuration.worker_count;
    if (multithreaded && worker_count == 0) {
        worker_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }
    if (!multithreaded) worker_count = 0;
    if (worker_count == configured_worker_count_
        && configuration.threading == configured_threading_) return;

    executor_.setWorkerCount(worker_count);
    worker_accelerations_.resize(worker_count);
    configured_worker_count_ = worker_count;
    configured_threading_ = configuration.threading;
}

void ScalarFullPhysicsEngine::calculateAccelerationsMultithreaded(const WorldState& world,
    const SimulationParameters& parameters, std::vector<Vec3>& accelerations) const {
    const std::size_t body_count = world.bodyCount();
    const Dimension dimension = world.dimension();
    const double softening_squared = parameters.softening_length * parameters.softening_length;
    for (std::vector<Vec3>& local : worker_accelerations_) {
        local.assign(body_count, Vec3{});
    }

    executor_.parallelFor(body_count, [&](std::size_t worker, std::size_t begin, std::size_t end) {
        std::vector<Vec3>& local = worker_accelerations_[worker];
        for (std::size_t first = begin; first < end; ++first) {
            const ConstBodyView first_body = world.body(first);
            for (std::size_t second = first + 1; second < body_count; ++second) {
                const ConstBodyView second_body = world.body(second);
                const Vec3 displacement = second_body.position - first_body.position;
                const double distance_squared = displacement.lengthSquared(dimension)
                    + softening_squared;
                if (distance_squared == 0.0) continue;

                const double distance = std::sqrt(distance_squared);
                const double inverse_distance_cubed = 1.0 / (distance_squared * distance);
                const double first_scale = parameters.gravitational_constant
                    * second_body.mass * inverse_distance_cubed;
                const double second_scale = parameters.gravitational_constant
                    * first_body.mass * inverse_distance_cubed;
                local[first] += displacement * first_scale;
                local[second] -= displacement * second_scale;
            }
        }
    });

    accelerations.assign(body_count, Vec3{});
    for (const std::vector<Vec3>& local : worker_accelerations_) {
        for (std::size_t index = 0; index < body_count; ++index) {
            accelerations[index] += local[index];
        }
    }
}

PhysicsStepResult ScalarFullPhysicsEngine::step(WorldState& world, const SimulationParameters& parameters) {
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

PhysicsEngineInfo ScalarFullPhysicsEngine::info(Dimension dimension) const {
    return {ComputeBackend::Scalar, ForceModel::Full, dimension,
            configured_threading_ == ThreadingMode::MultiThreaded
                ? "Scalar Full (MT)" : "Scalar Full", SolverKind::Full};
}

}
