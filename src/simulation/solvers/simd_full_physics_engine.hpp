#pragma once

#include "simulation/physics_engine.hpp"
#include "simulation/parallel_executor.hpp"
#include "simulation/scalar_collision_system.hpp"
#include "simulation/simd_collision_system.hpp"
#include "simulation/simd_capabilities.hpp"

#include <vector>

namespace nbody {

class SimdFullPhysicsEngine final : public IPhysicsEngine {
public:
    SimdFullPhysicsEngine();

    PhysicsStepResult step(WorldState& world, const SimulationParameters& parameters) override;
    PhysicsEngineInfo info(Dimension dimension) const override;
    SimdCapabilities capabilities() const { return capabilities_; }

private:
    void calculateAccelerations(const WorldState& world, const SimulationParameters& parameters,
                               std::vector<Vec3>& output) const;
    void configureThreading(const SolverConfiguration& configuration) const;
    mutable std::vector<Vec3> initial_accelerations_;
    mutable std::vector<Vec3> final_accelerations_;
    mutable SimdCollisionWorkspace simd_collision_workspace_;
    mutable ScalarCollisionWorkspace scalar_collision_workspace_;
    mutable ParallelExecutor executor_;
    mutable std::size_t configured_worker_count_{};
    mutable ThreadingMode configured_threading_{ThreadingMode::SingleThreaded};
    SimdCapabilities capabilities_;
};

}
