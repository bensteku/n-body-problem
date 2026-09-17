#include "simulation/world_state.hpp"
#include "simulation/solvers/scalar_full_solver.hpp"

#include <cassert>

int main() {
    using namespace nbody;

    WorldState first = WorldState::deterministic(8, Dimension::Two, 1234);
    WorldState second = WorldState::deterministic(8, Dimension::Two, 1234);
    assert(first.isValid());
    assert(second.isValid());
    assert(first.bodies().size() == second.bodies().size());

    for (std::size_t i = 0; i < first.bodies().size(); ++i) {
        assert(first.bodies()[i].id.value == i + 1);
        assert(first.bodies()[i].position == second.bodies()[i].position);
        assert(first.bodies()[i].velocity == second.bodies()[i].velocity);
    }

    first.advance(0.25);
    assert(first.time() == 0.25);
    assert(first.isValid());
    assert(first.diagnostics().finite);

    const Vec3 vector{3.0, 4.0, 12.0};
    assert(vector.lengthSquared2D() == 25.0);
    assert(vector.lengthSquared(Dimension::Two) == 25.0);
    assert(vector.lengthSquared(Dimension::Three) == 169.0);
    assert(vector.dot(Vec3{1.0, 1.0, 1.0}, Dimension::Two) == 7.0);
    assert(vector.dot(Vec3{1.0, 1.0, 1.0}, Dimension::Three) == 19.0);

    WorldState three_dimensional = WorldState::deterministic(2, Dimension::Three, 9);
    assert(three_dimensional.isValid());
    assert(three_dimensional.bodies()[0].position.z != 0.0 || three_dimensional.bodies()[0].velocity.z != 0.0);

    SimulationParameters parameters;
    parameters.gravitational_constant = 1.0;
    parameters.timestep = 0.01;
    parameters.softening_length = 0.0;

    WorldState two_body(Dimension::Two);
    two_body.addBody({{}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 1.0, 0.1, false});
    two_body.addBody({{}, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 1.0, 0.1, false});
    ScalarFullSolver solver;
    solver.step(two_body, parameters);
    assert(two_body.time() == parameters.timestep);
    assert(two_body.bodies()[0].velocity.x > 0.0);
    assert(two_body.bodies()[1].velocity.x < 0.0);
    assert(two_body.bodies()[0].position.z == 0.0);
    assert(two_body.isValid());

    WorldState static_central(Dimension::Three);
    static_central.addBody({{}, {0.0, 0.0, 0.0}, {}, 10.0, 0.1, true});
    static_central.addBody({{}, {2.0, 0.0, 1.0}, {}, 1.0, 0.1, false});
    parameters.dimension = Dimension::Three;
    solver.step(static_central, parameters);
    assert(static_central.bodies()[0].position == Vec3{});
    assert(static_central.bodies()[1].velocity.x < 0.0);
    assert(static_central.bodies()[1].velocity.z < 0.0);

    WorldState coincident(Dimension::Three);
    coincident.addBody({{}, {}, {}, 1.0, 0.1, false});
    coincident.addBody({{}, {}, {}, 1.0, 0.1, false});
    parameters.softening_length = 1e-3;
    solver.step(coincident, parameters);
    assert(coincident.diagnostics().finite);

    WorldState transparent(Dimension::Two);
    transparent.addBody({{}, {-0.25, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0, 1.0, false});
    transparent.addBody({{}, {0.25, 0.0, 0.0}, {-1.0, 0.0, 0.0}, 1.0, 1.0, false});
    parameters.dimension = Dimension::Two;
    parameters.gravitational_constant = 0.0;
    parameters.timestep = 0.5;
    parameters.softening_length = 0.0;
    parameters.collision.model = CollisionModel::Transparent;
    solver.step(transparent, parameters);
    assert(transparent.bodies()[0].position.x == 0.25);
    assert(transparent.bodies()[1].position.x == -0.25);
    assert(transparent.bodies()[0].velocity.x == 1.0);
    assert(transparent.bodies()[1].velocity.x == -1.0);

    WorldState hard_body(Dimension::Two);
    hard_body.addBody({{}, {-0.5, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0, 0.75, false});
    hard_body.addBody({{}, {0.5, 0.0, 0.0}, {-1.0, 0.0, 0.0}, 1.0, 0.75, false});
    parameters.collision.model = CollisionModel::HardBody;
    parameters.collision.restitution = 1.0;
    parameters.timestep = 0.0;
    solver.step(hard_body, parameters);
    assert(hard_body.bodies()[0].position.x == -0.75);
    assert(hard_body.bodies()[1].position.x == 0.75);
    assert(hard_body.bodies()[0].velocity.x == -1.0);
    assert(hard_body.bodies()[1].velocity.x == 1.0);

    WorldState static_collision(Dimension::Three);
    static_collision.addBody({{}, {0.0, 0.0, 0.0}, {}, 10.0, 1.0, true});
    static_collision.addBody({{}, {0.5, 0.0, 0.0}, {-1.0, 0.0, 0.0}, 1.0, 1.0, false});
    parameters.dimension = Dimension::Three;
    parameters.collision.restitution = 0.5;
    parameters.timestep = 0.0;
    solver.step(static_collision, parameters);
    assert(static_collision.bodies()[0].position == Vec3{});
    assert(static_collision.bodies()[1].position.x >= 2.0 - 1e-12);
    assert(static_collision.bodies()[1].velocity.x > 0.0);
    assert(static_collision.isValid());

    const MaterialProperties rocky = materialPreset(MaterialPreset::Rocky);
    const MaterialProperties metallic = materialPreset(MaterialPreset::Metallic);
    const MaterialProperties icy = materialPreset(MaterialPreset::Icy);
    const MaterialProperties gas = materialPreset(MaterialPreset::Gas);
    assert(rocky.preset == MaterialPreset::Rocky);
    assert(metallic.density > rocky.density);
    assert(icy.fragmentation_threshold < rocky.fragmentation_threshold);
    assert(gas.restitution < icy.restitution);

    MaterialProperties custom;
    applyMaterialPreset(custom, MaterialPreset::Metallic);
    assert(custom.preset == MaterialPreset::Metallic);
    assert(custom.tensile_strength == metallic.tensile_strength);
}
