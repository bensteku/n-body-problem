#include "simulation/solvers/scalar_full_solver.hpp"
#include "simulation/collision_system.hpp"
#include "simulation/boundary_system.hpp"

#include <cmath>
#include <vector>

namespace nbody {

namespace {

std::vector<Vec3> calculateAccelerations(const WorldState& world, const SimulationParameters& parameters) {
    const auto& bodies = world.bodies();
    std::vector<Vec3> accelerations(bodies.size());
    const double softening_squared = parameters.softening_length * parameters.softening_length;

    for (std::size_t first = 0; first < bodies.size(); ++first) {
        for (std::size_t second = first + 1; second < bodies.size(); ++second) {
            const Vec3 displacement = bodies[second].position - bodies[first].position;
            const double distance_squared = displacement.lengthSquared(parameters.dimension) + softening_squared;
            if (distance_squared == 0.0) continue;

            const double distance = std::sqrt(distance_squared);
            const double inverse_distance_cubed = 1.0 / (distance_squared * distance);
            const double first_scale = parameters.gravitational_constant * bodies[second].mass * inverse_distance_cubed;
            const double second_scale = parameters.gravitational_constant * bodies[first].mass * inverse_distance_cubed;

            accelerations[first] += displacement * first_scale;
            accelerations[second] -= displacement * second_scale;
        }
    }
    return accelerations;
}

}

void ScalarFullSolver::step(WorldState& world, const SimulationParameters& parameters) {
    if (parameters.integrator != Integrator::VelocityVerlet) return;

    const double timestep = parameters.timestep;
    const auto initial_accelerations = calculateAccelerations(world, parameters);
    auto bodies = world.mutableBodies();

    for (std::size_t index = 0; index < bodies.size(); ++index) {
        BodyState& body = bodies[index];
        if (body.is_static) continue;
        body.position += body.velocity * timestep + initial_accelerations[index] * (0.5 * timestep * timestep);
        if (parameters.dimension == Dimension::Two) body.position.z = 0.0;
    }

    const auto final_accelerations = calculateAccelerations(world, parameters);
    for (std::size_t index = 0; index < bodies.size(); ++index) {
        BodyState& body = bodies[index];
        if (body.is_static) continue;
        body.velocity += (initial_accelerations[index] + final_accelerations[index]) * (0.5 * timestep);
        if (parameters.dimension == Dimension::Two) body.velocity.z = 0.0;
    }

    CollisionSystem::resolveContacts(world, parameters.collision, parameters.gravitational_constant);
    BoundarySystem::resolve(world, parameters.boundary);
    CollisionSystem::applyDeferredOutcomes(world, parameters.collision);
    world.advanceTime(timestep);
}

SolverInfo ScalarFullSolver::info(Dimension dimension) const {
    return {ComputeBackend::Scalar, ForceModel::Full, dimension, "Scalar Full"};
}

}
