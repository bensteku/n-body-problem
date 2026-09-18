#include "simulation/solvers/simd_full_physics_engine.hpp"

#include "simulation/boundary_system.hpp"
#include "simulation/simd_collision_system.hpp"
#include "simulation/solvers/scalar_full_physics_engine.hpp"
#include "simulation/solvers/simd_avx2_kernel.hpp"

#include <stdexcept>

namespace nbody {

SimdFullPhysicsEngine::SimdFullPhysicsEngine() : capabilities_(detectSimdCapabilities()) {}

void SimdFullPhysicsEngine::calculateAccelerations(const WorldState& world,
    const SimulationParameters& parameters, std::vector<Vec3>& output) const {
    if (capabilities_.avx2) {
        calculateAvx2Accelerations(world, parameters, output);
        return;
    }
    output.assign(world.bodyCount(), Vec3{});
}

void SimdFullPhysicsEngine::step(WorldState& world, const SimulationParameters& parameters) {
    if (!capabilities_.avx2) {
        ScalarFullPhysicsEngine fallback;
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

PhysicsEngineInfo SimdFullPhysicsEngine::info(Dimension dimension) const {
    return {ComputeBackend::SIMD, ForceModel::Full, dimension,
            capabilities_.avx2 ? "SIMD Full (AVX2)" : "SIMD Full (scalar fallback)",
            SolverKind::Full};
}

}
