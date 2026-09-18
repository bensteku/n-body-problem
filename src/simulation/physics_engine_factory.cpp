#include "simulation/physics_engine_factory.hpp"

#include "simulation/solvers/scalar_full_physics_engine.hpp"
#include "simulation/solvers/scalar_approximated_physics_engine.hpp"
#include "simulation/solvers/simd_full_physics_engine.hpp"
#include "simulation/solvers/simd_approximated_physics_engine.hpp"

#include <stdexcept>

namespace nbody {

std::unique_ptr<IPhysicsEngine> createPhysicsEngine(const SolverConfiguration& configuration) {
    const SolverConfigurationValidation validation = validateSolverConfiguration(configuration);
    if (!validation.valid) throw std::invalid_argument(validation.message.data());
    const bool approximated = configuration.kind == SolverKind::Approximated;
    switch (configuration.backend) {
    case ComputeBackend::Scalar:
        if (approximated) return std::make_unique<ScalarApproximatedPhysicsEngine>();
        return std::make_unique<ScalarFullPhysicsEngine>();
    case ComputeBackend::SIMD:
        if (approximated) return std::make_unique<SimdApproximatedPhysicsEngine>();
        return std::make_unique<SimdFullPhysicsEngine>();
    case ComputeBackend::GPU:
        // Preserve the requested accuracy policy until a real GPU engine exists.
        if (approximated) return std::make_unique<ScalarApproximatedPhysicsEngine>();
        return std::make_unique<ScalarFullPhysicsEngine>();
    }
    return std::make_unique<ScalarFullPhysicsEngine>();
}

}
