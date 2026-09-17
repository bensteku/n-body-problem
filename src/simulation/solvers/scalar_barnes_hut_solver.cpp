#include "simulation/solvers/scalar_barnes_hut_solver.hpp"

#include "simulation/boundary_system.hpp"
#include "simulation/collision_system.hpp"

#include <algorithm>
#include <stdexcept>

namespace nbody {

void ScalarBarnesHutSolver::calculateAccelerations(const WorldState& world,
                                                   const SimulationParameters& parameters,
                                                   std::vector<Vec3>& output) const {
    output.assign(world.bodyCount(), Vec3{});
    const BarnesHutSettings& settings = parameters.solver.barnes_hut;
    if (!tree_ || tree_->dimension() != world.dimension()
        || tree_->leafCapacity() != settings.leaf_capacity
        || tree_->maximumDepth() != settings.maximum_depth) {
        tree_ = std::make_unique<BarnesHutTree>(world.dimension(), settings.leaf_capacity,
                                                settings.maximum_depth);
    }
    tree_->rebuild(world.bodyStorage());
    const double opening_angle = std::clamp(parameters.solver.barnes_hut.opening_angle, 0.0, 10.0);
    for (std::size_t index = 0; index < world.bodyCount(); ++index) {
        if (world.body(index).is_static()) continue;
        output[index] = tree_->accelerationOn(index, world.bodyStorage(),
                                              parameters.gravitational_constant,
                                              parameters.softening_length, opening_angle);
    }
}

void ScalarBarnesHutSolver::step(WorldState& world, const SimulationParameters& parameters) {
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

SolverInfo ScalarBarnesHutSolver::info(Dimension dimension) const {
    return {ComputeBackend::Scalar, ForceModel::BarnesHut, dimension, "Scalar Barnes-Hut"};
}

}
