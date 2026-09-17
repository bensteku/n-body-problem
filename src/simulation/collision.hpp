#pragma once

namespace nbody {

enum class CollisionModel {
    Transparent,
    HardBody,
    HeuristicFragmentation,
    HeuristicAbsorption
};

struct CollisionSettings {
    CollisionModel model{CollisionModel::Transparent};
    double restitution{1.0};
};

}
