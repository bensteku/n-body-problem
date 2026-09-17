#pragma once

#include "body_state.hpp"
#include "simulation_parameters.hpp"

#include <cstddef>
#include <vector>

namespace nbody {

struct WorldDiagnostics {
    double total_mass{};
    Vec3 center_of_mass{};
    bool finite{true};
};

class WorldState {
public:
    explicit WorldState(Dimension dimension = Dimension::Two);

    Dimension dimension() const { return dimension_; }
    double time() const { return time_; }
    const std::vector<BodyState>& bodies() const { return bodies_; }

    BodyId addBody(BodyState body);
    void advance(double timestep);
    WorldDiagnostics diagnostics() const;
    bool isValid() const;

    static WorldState deterministic(std::size_t body_count, Dimension dimension, unsigned long long seed);

private:
    Dimension dimension_;
    double time_{};
    BodyId next_id_{1};
    std::vector<BodyState> bodies_;
};

}
