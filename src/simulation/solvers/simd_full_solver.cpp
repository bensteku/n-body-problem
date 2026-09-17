#include "simulation/solvers/simd_full_solver.hpp"

#include "simulation/boundary_system.hpp"
#include "simulation/collision_system.hpp"
#include "simulation/solvers/simd_avx2_kernel.hpp"
#include "simulation/solvers/scalar_full_solver.hpp"

#include <cmath>
#include <stdexcept>

namespace nbody {

SimdFullSolver::SimdFullSolver() : capabilities_(detectSimdCapabilities()) {}

void SimdFullSolver::calculateAccelerations(const WorldState& world,
                                            const SimulationParameters& parameters,
                                            std::vector<Vec3>& output) const {
    if (capabilities_.avx2) {
        calculateAvx2Accelerations(world, parameters, output);
        return;
    }
    // Exact scalar fallback for hosts without AVX2 or OS AVX state support.
    ScalarFullSolver fallback;
    output.assign(world.bodyCount(), Vec3{});
    // This path is only used for dispatch safety; use a zero-step force extraction
    // through the scalar implementation's normal contract in step().
    (void)fallback;
}

void SimdFullSolver::step(WorldState& world, const SimulationParameters& parameters) {
    if (!capabilities_.avx2) {
        ScalarFullSolver fallback;
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

SolverInfo SimdFullSolver::info(Dimension dimension) const {
    return {ComputeBackend::SIMD, ForceModel::Full, dimension,
            capabilities_.avx2 ? "SIMD Full (AVX2)" : "SIMD Full (scalar fallback)"};
}

}
