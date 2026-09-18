#pragma once

#include "simulation/physics_engine.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace nbody_test {

inline void require(bool condition, const char* expression, const char* file, int line) {
    if (condition) return;
    std::cerr << file << ':' << line << ": requirement failed: " << expression << '\n';
    throw std::runtime_error("test requirement failed");
}

struct NumericalParityTolerance {
    double absolute{1e-10};
    double relative{1e-10};
};

inline bool approximatelyEqual(double left, double right,
                               NumericalParityTolerance tolerance) {
    const double scale = std::max({1.0, std::abs(left), std::abs(right)});
    return std::abs(left - right) <= tolerance.absolute + tolerance.relative * scale;
}

inline bool approximatelyEqual(const nbody::Vec3& left, const nbody::Vec3& right,
                               nbody::Dimension dimension,
                               NumericalParityTolerance tolerance) {
    if (!approximatelyEqual(left.x, right.x, tolerance)
        || !approximatelyEqual(left.y, right.y, tolerance)) return false;
    return dimension == nbody::Dimension::Two
        || approximatelyEqual(left.z, right.z, tolerance);
}

inline bool worldsHaveNumericalParity(const nbody::WorldState& reference,
                                      const nbody::WorldState& candidate,
                                      NumericalParityTolerance tolerance) {
    if (reference.dimension() != candidate.dimension()
        || reference.bodyCount() != candidate.bodyCount()
        || !approximatelyEqual(reference.time(), candidate.time(), tolerance)) return false;
    for (std::size_t index = 0; index < reference.bodyCount(); ++index) {
        const nbody::ConstBodyView reference_body = reference.body(index);
        const nbody::ConstBodyView candidate_body = candidate.body(index);
        if (reference_body.id != candidate_body.id
            || reference_body.kind != candidate_body.kind
            || reference_body.is_static() != candidate_body.is_static()
            || !approximatelyEqual(reference_body.mass, candidate_body.mass, tolerance)
            || !approximatelyEqual(reference_body.radius, candidate_body.radius, tolerance)
            || !approximatelyEqual(reference_body.position, candidate_body.position,
                                   reference.dimension(), tolerance)
            || !approximatelyEqual(reference_body.velocity, candidate_body.velocity,
                                   reference.dimension(), tolerance)) return false;
    }
    if (reference.collisionEvents().size() != candidate.collisionEvents().size()) return false;
    for (std::size_t index = 0; index < reference.collisionEvents().size(); ++index) {
        const nbody::CollisionEvent& reference_event = reference.collisionEvents()[index];
        const nbody::CollisionEvent& candidate_event = candidate.collisionEvents()[index];
        if (reference_event.first_body != candidate_event.first_body
            || reference_event.second_body != candidate_event.second_body
            || reference_event.first.outcome != candidate_event.first.outcome
            || reference_event.second.outcome != candidate_event.second.outcome
            || reference_event.absorption_survivor != candidate_event.absorption_survivor
            || reference_event.absorption_removed != candidate_event.absorption_removed
            || reference_event.absorber_destroyed != candidate_event.absorber_destroyed
            || !approximatelyEqual(reference_event.normal_speed, candidate_event.normal_speed, tolerance)
            || !approximatelyEqual(reference_event.tangential_speed, candidate_event.tangential_speed, tolerance)
            || !approximatelyEqual(reference_event.impact_energy, candidate_event.impact_energy, tolerance)) return false;
    }
    const nbody::WorldDiagnostics reference_diagnostics = reference.diagnostics();
    const nbody::WorldDiagnostics candidate_diagnostics = candidate.diagnostics();
    return reference_diagnostics.finite == candidate_diagnostics.finite
        && approximatelyEqual(reference_diagnostics.total_mass, candidate_diagnostics.total_mass, tolerance)
        && approximatelyEqual(reference_diagnostics.center_of_mass, candidate_diagnostics.center_of_mass,
                              reference.dimension(), tolerance);
}

}

#define REQUIRE(condition) ::nbody_test::require((condition), #condition, __FILE__, __LINE__)
