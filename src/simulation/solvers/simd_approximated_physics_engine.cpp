#include "simulation/solvers/simd_approximated_physics_engine.hpp"

#include "simulation/boundary_system.hpp"
#include "simulation/simd_collision_system.hpp"
#include "simulation/solvers/scalar_approximated_physics_engine.hpp"
#include "simulation/solvers/simd_barnes_hut_kernel.hpp"

#include <stdexcept>

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
    calculateAvx2BarnesHutAccelerations(world, parameters, *tree_, output);
}

void SimdApproximatedPhysicsEngine::step(WorldState& world,
    const SimulationParameters& parameters) {
    if (!capabilities_.avx2) {
        ScalarApproximatedPhysicsEngine fallback;
        fallback.step(world, parameters);
        return;
    }
    if (parameters.integrator != Integrator::VelocityVerlet) return;
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
                                         parameters.gravitational_constant);
    BoundarySystem::resolve(world, parameters.boundary);
    SimdCollisionSystem::applyDeferredOutcomes(world, parameters.collision);
    world.advanceTime(timestep);
}

PhysicsEngineInfo SimdApproximatedPhysicsEngine::info(Dimension dimension) const {
    return {ComputeBackend::SIMD, ForceModel::BarnesHut, dimension,
            capabilities_.avx2 ? "SIMD Barnes-Hut (AVX2)" : "SIMD Barnes-Hut (scalar fallback)",
            SolverKind::Approximated};
}

}
