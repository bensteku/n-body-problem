#include "physics_session.hpp"

namespace nbody {

PhysicsSession::PhysicsSession(SolverConfiguration configuration)
    : configuration_(configuration), engine_(createPhysicsEngine(configuration_)) {}

PhysicsStepResult PhysicsSession::step(WorldState& world,
                                       const SimulationParameters& parameters) {
    if (parameters.dimension != world.dimension()) {
        return {PhysicsStepStatus::Rejected, world.time(), world.bodyCount(),
                "simulation parameter dimension does not match world dimension"};
    }
    if (dimension_.has_value() && dimension_.value() != world.dimension()) {
        return {PhysicsStepStatus::Rejected, world.time(), world.bodyCount(),
                "simulation dimension cannot change during a session"};
    }
    dimension_ = world.dimension();
    SimulationParameters effective_parameters = parameters;
    effective_parameters.solver = configuration_;
    return engine_->step(world, effective_parameters);
}

FrameLease PhysicsSession::publishFrame(const WorldState& world) {
    return frame_publisher_.publish(world);
}

EngineSwitchResult PhysicsSession::switchEngine(const SolverConfiguration& configuration,
                                                const WorldState& world) {
    const SolverConfigurationValidation validation = validateSolverConfiguration(configuration);
    if (!validation.valid) return {EngineSwitchStatus::Rejected, validation.message};
    if (!world.isValid()) {
        return {EngineSwitchStatus::Rejected, "cannot switch engines from an invalid world state"};
    }
    if (dimension_.has_value() && dimension_.value() != world.dimension()) {
        return {EngineSwitchStatus::Rejected, "world dimension does not match session dimension"};
    }

    std::unique_ptr<IPhysicsEngine> candidate;
    try {
        candidate = createPhysicsEngine(configuration);
    } catch (...) {
        return {EngineSwitchStatus::Rejected, "candidate physics engine could not be created"};
    }
    if (candidate->info(world.dimension()).dimension != world.dimension()) {
        return {EngineSwitchStatus::Rejected, "candidate physics engine dimension is incompatible"};
    }

    configuration_ = configuration;
    engine_ = std::move(candidate);
    return {EngineSwitchStatus::Switched, "physics engine switched"};
}

}
