#pragma once

#include "simulation/barnes_hut_tree.hpp"
#include "simulation/physics_engine.hpp"
#include "simulation/parallel_executor.hpp"
#include "simulation/scalar_collision_system.hpp"
#include "simulation/simd_collision_system.hpp"
#include "simulation/simd_capabilities.hpp"

#include <memory>
#include <vector>

namespace nbody {

class SimdApproximatedPhysicsEngine final : public IPhysicsEngine {
public:
    SimdApproximatedPhysicsEngine();

    PhysicsStepResult step(WorldState& world, const SimulationParameters& parameters) override;
    PhysicsEngineInfo info(Dimension dimension) const override;

private:
    void calculateAccelerations(const WorldState& world, const SimulationParameters& parameters,
                               std::vector<Vec3>& output) const;
    void configureThreading(const SolverConfiguration& configuration) const;
    mutable std::vector<Vec3> initial_accelerations_;
    mutable std::vector<Vec3> final_accelerations_;
    mutable std::vector<std::size_t> traversal_stack_;
    mutable std::unique_ptr<BarnesHutTree> tree_;
    mutable SimdCollisionWorkspace simd_collision_workspace_;
    mutable ScalarCollisionWorkspace scalar_collision_workspace_;
    mutable ParallelExecutor executor_;
    mutable std::vector<std::vector<std::size_t>> worker_traversal_stacks_;
    mutable std::size_t configured_worker_count_{};
    mutable ThreadingMode configured_threading_{ThreadingMode::SingleThreaded};
    SimdCapabilities capabilities_;
};

}
