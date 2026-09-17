#pragma once

#include "body_id.hpp"

#include <cstddef>

namespace nbody {

enum class CollisionModel {
    Transparent,
    HardBody,
    HeuristicFragmentation,
    HeuristicAbsorption
};

enum class CollisionOutcome {
    Bounce,
    Damage,
    Fragment
};

struct CollisionClassifierSettings {
    double damage_threshold_scale{1.0};
    double fragmentation_threshold_scale{1.0};
    double strength_scale{1.0};
    double binding_scale{1.0};
    double tangential_energy_scale{0.25};
    bool force_damage{false};
    bool force_fragmentation{false};
};

struct CollisionAssessment {
    BodyId body;
    CollisionOutcome outcome{CollisionOutcome::Bounce};
    double specific_impact_energy{};
    double damage_limit{};
    double fragmentation_limit{};
};

struct CollisionEvent {
    BodyId first_body;
    BodyId second_body;
    double normal_speed{};
    double tangential_speed{};
    double impact_energy{};
    CollisionAssessment first;
    CollisionAssessment second;
};

struct CollisionSettings {
    CollisionModel model{CollisionModel::Transparent};
    double restitution{1.0};
    CollisionClassifierSettings classifier;
    std::size_t minimum_fragments{2};
    std::size_t maximum_fragments{5};
    std::size_t maximum_fragment_count{10000};
};

}
