#include "simulation/solvers/scalar_full_solver.hpp"
#include "simulation/collision_system.hpp"
#include "simulation/boundary_system.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace nbody {

void ScalarFullSolver::calculateAccelerations(const WorldState& world, const SimulationParameters& parameters,
                                              std::vector<Vec3>& accelerations) const {
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
            const double first_scale = parameters.gravitational_constant * second_body.mass * inverse_distance_cubed;
            const double second_scale = parameters.gravitational_constant * first_body.mass * inverse_distance_cubed;

            accelerations[first] += displacement * first_scale;
            accelerations[second] -= displacement * second_scale;
        }
    }
}

void ScalarFullSolver::step(WorldState& world, const SimulationParameters& parameters) {
    if (parameters.integrator != Integrator::VelocityVerlet) return;
    if (parameters.dimension != world.dimension()) {
        throw std::invalid_argument("Simulation parameter dimension does not match WorldState dimension");
    }

    const double timestep = parameters.timestep;
    calculateAccelerations(world, parameters, initial_accelerations_);

    for (std::size_t index = 0; index < world.bodyCount(); ++index) {
        MutableBodyView body = world.mutableBody(index);
        if (body.is_static()) continue;
        body.position += body.velocity * timestep + initial_accelerations_[index] * (0.5 * timestep * timestep);
        if (world.dimension() == Dimension::Two) body.position.z = 0.0;
    }

    calculateAccelerations(world, parameters, final_accelerations_);
    for (std::size_t index = 0; index < world.bodyCount(); ++index) {
        MutableBodyView body = world.mutableBody(index);
        if (body.is_static()) continue;
        body.velocity += (initial_accelerations_[index] + final_accelerations_[index]) * (0.5 * timestep);
        if (world.dimension() == Dimension::Two) body.velocity.z = 0.0;
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
