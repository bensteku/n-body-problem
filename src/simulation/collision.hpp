#pragma once

#include "body_id.hpp"

#include <cstddef>
#include <vector>

namespace nbody {

enum class CollisionModel {
    Transparent,
    HardBody
};

enum class CollisionOutcome {
    Bounce,
    Damage,
    Fragment,
    Absorb,
    DestroyAbsorber
};

struct CollisionClassifierSettings {
    double damage_threshold_scale{1.0};
    double fragmentation_threshold_scale{1.0};
    double strength_scale{1.0};
    double binding_scale{1.0};
    double tangential_energy_scale{0.25};
    bool fragmentation_enabled{false};
    bool absorption_enabled{false};
    bool force_damage{false};
    bool force_fragmentation{false};
};

struct CollisionTransitionDiagnostics {
    CollisionModel active_model{CollisionModel::Transparent};
    CollisionModel requested_model{CollisionModel::Transparent};
    bool failed{false};
    std::size_t passes{};
    std::vector<BodyId> unresolved_bodies;
};

struct CollisionAssessment {
    BodyId body;
    CollisionOutcome outcome{CollisionOutcome::Bounce};
    double specific_impact_energy{};
    double damage_limit{};
    double fragmentation_limit{};
};

struct CollisionBroadPhaseSettings {
    bool spatial_tree_enabled{true};
    std::size_t leaf_capacity{16};
    std::size_t maximum_depth{12};
    double looseness{1.0};
};

struct CollisionEvent {
    BodyId first_body;
    BodyId second_body;
    double normal_speed{};
    double tangential_speed{};
    double impact_energy{};
    CollisionAssessment first;
    CollisionAssessment second;
    BodyId absorption_survivor;
    BodyId absorption_removed;
    bool absorber_destroyed{false};
};

struct CollisionSettings {
    CollisionModel model{CollisionModel::Transparent};
    double restitution{1.0};
    std::size_t maximum_consistency_passes{50};
    double absorber_mass_tolerance{1.0e-6};
    double absorber_size_tolerance{1.0e-6};
    double solid_absorption_mass_ratio{2.0};
    double solid_absorption_size_ratio{2.0};
    CollisionBroadPhaseSettings broad_phase;
    CollisionClassifierSettings classifier;
    std::size_t minimum_fragments{2};
    std::size_t maximum_fragments{5};
    std::size_t maximum_fragment_count{10000};
};

}
