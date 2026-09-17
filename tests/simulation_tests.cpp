#include "simulation/world_state.hpp"
#include "simulation/solvers/scalar_full_solver.hpp"
#include "simulation/solvers/scalar_barnes_hut_solver.hpp"
#include "simulation/barnes_hut_tree.hpp"
#include "simulation/spatial_tree.hpp"
#include "test_support.hpp"

#define assert(condition) REQUIRE(condition)
#include <cmath>
#include <algorithm>
#include <random>
#include <vector>

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

    WorldState quadtree_world(Dimension::Two);
    quadtree_world.addBody({{}, {-0.25, 0.0, 0.0}, {}, 1.0, 0.5, false});
    quadtree_world.addBody({{}, {0.25, 0.0, 0.0}, {}, 1.0, 0.5, false});
    quadtree_world.addBody({{}, {10.0, 10.0, 0.0}, {}, 1.0, 0.5, false});
    SpatialTree quadtree(Dimension::Two, 1);
    quadtree.rebuild(quadtree_world.bodyStorage());
    std::vector<std::pair<std::size_t, std::size_t>> spatial_pairs;
    quadtree.potentialContactPairs(quadtree_world.bodyStorage(), spatial_pairs);
    assert(spatial_pairs.size() == 1);

    WorldState octree_world(Dimension::Three);
    octree_world.addBody({{}, {0.0, 0.0, -2.0}, {}, 1.0, 0.5, false});
    octree_world.addBody({{}, {0.0, 0.0, 2.0}, {}, 1.0, 0.5, false});
    SpatialTree octree(Dimension::Three, 1);
    octree.rebuild(octree_world.bodyStorage());
    octree.potentialContactPairs(octree_world.bodyStorage(), spatial_pairs);
    assert(spatial_pairs.empty());

    // Compare both tree modes against the exact brute-force contact set.
    for (const Dimension test_dimension : {Dimension::Two, Dimension::Three}) {
        WorldState random_world(test_dimension);
        std::mt19937 generator(test_dimension == Dimension::Two ? 17U : 31U);
        std::uniform_real_distribution<double> position(-20.0, 20.0);
        std::uniform_real_distribution<double> radius(0.05, 0.8);
        for (std::size_t index = 0; index < 300; ++index) {
            random_world.addBody({{},
                {position(generator), position(generator), test_dimension == Dimension::Three ? position(generator) : 0.0},
                {}, 1.0, radius(generator), false});
        }

        std::vector<std::pair<std::size_t, std::size_t>> brute_force_pairs;
        for (std::size_t first = 0; first < random_world.bodyCount(); ++first) {
            const ConstBodyView first_body = random_world.body(first);
            for (std::size_t second = first + 1; second < random_world.bodyCount(); ++second) {
                const ConstBodyView second_body = random_world.body(second);
                const Vec3 displacement = second_body.position - first_body.position;
                const double combined_radius = first_body.radius + second_body.radius;
                if (displacement.lengthSquared(test_dimension) <= combined_radius * combined_radius) {
                    brute_force_pairs.emplace_back(first, second);
                }
            }
        }
        SpatialTree random_tree(test_dimension, 8, 10);
        random_tree.rebuild(random_world.bodyStorage());
        random_tree.potentialContactPairs(random_world.bodyStorage(), spatial_pairs);
        std::sort(brute_force_pairs.begin(), brute_force_pairs.end());
        std::sort(spatial_pairs.begin(), spatial_pairs.end());
        assert(spatial_pairs == brute_force_pairs);
    }

    WorldState gravity_world(Dimension::Three);
    std::mt19937 gravity_generator(77U);
    std::uniform_real_distribution<double> gravity_position(-10.0, 10.0);
    for (std::size_t index = 0; index < 80; ++index) {
        gravity_world.addBody({{},
            {gravity_position(gravity_generator), gravity_position(gravity_generator), gravity_position(gravity_generator)},
            {}, 0.5 + static_cast<double>(index % 7) * 0.2, 0.1, false});
    }
    BarnesHutTree exact_tree(Dimension::Three, 4, 12);
    exact_tree.rebuild(gravity_world.bodyStorage());
    for (std::size_t target = 0; target < gravity_world.bodyCount(); ++target) {
        const Vec3 tree_acceleration = exact_tree.accelerationOn(target, gravity_world.bodyStorage(), 1.0, 0.0, 0.0);
        Vec3 brute_acceleration{};
        const ConstBodyView target_body = gravity_world.body(target);
        for (std::size_t source = 0; source < gravity_world.bodyCount(); ++source) {
            if (source == target) continue;
            const ConstBodyView source_body = gravity_world.body(source);
            const Vec3 displacement = source_body.position - target_body.position;
            const double distance_squared = displacement.lengthSquared(Dimension::Three);
            const double distance = std::sqrt(distance_squared);
            brute_acceleration += displacement * (source_body.mass / (distance_squared * distance));
        }
        assert((tree_acceleration - brute_acceleration).length(Dimension::Three) < 1e-8);
    }

    WorldState full_world = WorldState::deterministic(100, Dimension::Two, 91);
    WorldState barnes_world = WorldState::deterministic(100, Dimension::Two, 91);
    SimulationParameters full_parameters;
    full_parameters.dimension = Dimension::Two;
    full_parameters.gravitational_constant = 0.1;
    full_parameters.timestep = 1e-4;
    full_parameters.collision.model = CollisionModel::Transparent;
    SimulationParameters barnes_parameters = full_parameters;
    barnes_parameters.solver.force_model = ForceModel::BarnesHut;
    barnes_parameters.solver.barnes_hut.opening_angle = 0.5;
    ScalarFullSolver full_solver;
    ScalarBarnesHutSolver barnes_solver;
    full_solver.step(full_world, full_parameters);
    barnes_solver.step(barnes_world, barnes_parameters);
    for (std::size_t index = 0; index < full_world.bodyCount(); ++index) {
        assert((full_world.body(index).velocity - barnes_world.body(index).velocity).length(Dimension::Two) < 1e-5);
    }

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

    bool dimension_mismatch_rejected = false;
    parameters.dimension = Dimension::Three;
    try {
        solver.step(two_body, parameters);
    } catch (const std::invalid_argument&) {
        dimension_mismatch_rejected = true;
    }
    assert(dimension_mismatch_rejected);
    parameters.dimension = Dimension::Three;

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
    assert(hard_body.collisionEvents().size() == 1);
    assert(hard_body.collisionEvents()[0].first.outcome == CollisionOutcome::Bounce);
    assert(hard_body.collisionEvents()[0].second.outcome == CollisionOutcome::Bounce);

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
    assert(static_collision.collisionEvents().size() == 1);
    assert(static_collision.collisionEvents()[0].first.outcome == CollisionOutcome::Bounce);
    assert(static_collision.collisionEvents()[0].second.outcome == CollisionOutcome::Bounce);

    WorldState fragmenting(Dimension::Two);
    fragmenting.addBody({{}, {-0.5, 0.0, 0.0}, {1.0, 0.0, 0.0}, 8.0, 1.0, false});
    fragmenting.addBody({{}, {0.5, 0.0, 0.0}, {-1.0, 0.0, 0.0}, 4.0, 1.0, false});
    parameters.dimension = Dimension::Two;
    parameters.timestep = 0.0;
    parameters.collision.minimum_fragments = 3;
    parameters.collision.maximum_fragments = 3;
    parameters.collision.maximum_fragment_count = 32;
    parameters.collision.classifier.force_fragmentation = true;
    solver.step(fragmenting, parameters);
    assert(fragmenting.bodies().size() == 6);
    double fragment_mass = 0.0;
    for (const BodyState& body : fragmenting.bodies()) fragment_mass += body.mass;
    assert(std::abs(fragment_mass - 12.0) < 1e-12);
    assert(fragmenting.isValid());

    WorldState damaged(Dimension::Two);
    damaged.addBody({{}, {-0.5, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0, 0.75, false});
    damaged.addBody({{}, {0.5, 0.0, 0.0}, {-1.0, 0.0, 0.0}, 1.0, 0.75, false});
    parameters.collision.classifier.force_fragmentation = false;
    parameters.collision.classifier.force_damage = true;
    parameters.collision.minimum_fragments = 2;
    parameters.collision.maximum_fragments = 2;
    parameters.timestep = 0.0;
    solver.step(damaged, parameters);
    assert(damaged.bodies().size() == 2);
    assert(damaged.bodies()[0].accumulated_damage > 0.0);
    assert(damaged.bodies()[1].accumulated_damage > 0.0);
    assert(damaged.isValid());

    WorldState bounded(Dimension::Two);
    bounded.addBody({{}, {-0.75, 0.0, 0.0}, {-2.0, 0.0, 0.0}, 1.0, 0.5, false});
    parameters.dimension = Dimension::Two;
    parameters.collision.model = CollisionModel::Transparent;
    parameters.boundary.enabled = true;
    parameters.boundary.minimum = {-1.0, -1.0, -1.0};
    parameters.boundary.maximum = {1.0, 1.0, 1.0};
    parameters.boundary.restitution = 0.5;
    parameters.timestep = 0.0;
    solver.step(bounded, parameters);
    assert(bounded.bodies()[0].position.x == -0.5);
    assert(bounded.bodies()[0].velocity.x == 1.0);
    assert(bounded.bodies()[0].position.z == 0.0);

    WorldState unbounded(Dimension::Two);
    unbounded.addBody({{}, {-2.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, 1.0, 0.5, false});
    parameters.boundary.enabled = false;
    solver.step(unbounded, parameters);
    assert(unbounded.bodies()[0].position.x == -2.0);
    assert(unbounded.bodies()[0].velocity.x == -1.0);

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
