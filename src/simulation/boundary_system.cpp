#include "simulation/boundary_system.hpp"

#include <algorithm>

namespace nbody {

namespace {

void resolveAxis(double& position, double& velocity, double radius,
                 double minimum, double maximum, double restitution) {
    const double lower = minimum + radius;
    const double upper = maximum - radius;
    if (lower > upper) {
        position = (minimum + maximum) * 0.5;
        velocity = 0.0;
        return;
    }
    if (position < lower) {
        position = lower;
        if (velocity < 0.0) velocity = -velocity * restitution;
    } else if (position > upper) {
        position = upper;
        if (velocity > 0.0) velocity = -velocity * restitution;
    }
}

}

void BoundarySystem::resolve(WorldState& world, const BoundarySettings& settings) {
    if (!settings.enabled) return;

    const double restitution = std::clamp(settings.restitution, 0.0, 1.0);
    if (settings.response != BoundaryResponse::Reflective) return;

    for (BodyState& body : world.mutableBodies()) {
        if (body.is_static) continue;
        resolveAxis(body.position.x, body.velocity.x, body.radius,
                    settings.minimum.x, settings.maximum.x, restitution);
        resolveAxis(body.position.y, body.velocity.y, body.radius,
                    settings.minimum.y, settings.maximum.y, restitution);
        if (world.dimension() == Dimension::Three) {
            resolveAxis(body.position.z, body.velocity.z, body.radius,
                        settings.minimum.z, settings.maximum.z, restitution);
        } else {
            body.position.z = 0.0;
            body.velocity.z = 0.0;
        }
    }
}

}
