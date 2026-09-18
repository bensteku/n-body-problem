#pragma once

#include "physics_engine.hpp"

#include <memory>

namespace nbody {

std::unique_ptr<IPhysicsEngine> createPhysicsEngine(const SolverConfiguration& configuration);

}
