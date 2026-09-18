#pragma once

#include "simulation/physics_engine.hpp"
#include "simulation/parallel_executor.hpp"
#include "simulation/scalar_collision_system.hpp"

#include <vector>

namespace nbody {

class ScalarFullPhysicsEngine final : public IPhysicsEngine {
public:
    PhysicsStepResult step(WorldState& world, const SimulationParameters& parameters) override;
    PhysicsEngineInfo info(Dimension dimension) const override;

private:
    void calculateAccelerations(const WorldState& world, const SimulationParameters& parameters,
                               std::vector<Vec3>& accelerations) const;
    void calculateAccelerationsMultithreaded(const WorldState& world,
                                             const SimulationParameters& parameters,
                                             std::vector<Vec3>& accelerations) const;
    void configureThreading(const SolverConfiguration& configuration) const;

    std::vector<Vec3> initial_accelerations_;
    std::vector<Vec3> final_accelerations_;
    mutable ScalarCollisionWorkspace collision_workspace_;
    mutable ParallelExecutor executor_;
    mutable std::vector<std::vector<Vec3>> worker_accelerations_;
    mutable std::size_t configured_worker_count_{};
    mutable ThreadingMode configured_threading_{ThreadingMode::SingleThreaded};
};

}
