#include "simulation/world_state.hpp"
#include "simulation/solvers/scalar_full_physics_engine.hpp"
#include "simulation/solvers/scalar_approximated_physics_engine.hpp"
#include "simulation/solvers/simd_full_physics_engine.hpp"
#include "simulation/solvers/simd_approximated_physics_engine.hpp"
#include "simulation/barnes_hut_tree.hpp"
#include "simulation/spatial_tree.hpp"
#include "simulation/scalar_collision_system.hpp"
#include "simulation/simd_collision_system.hpp"
#include "simulation/physics_engine_factory.hpp"
#include "simulation/simulation_frame.hpp"
#include "simulation/physics_session.hpp"
#include "test_support.hpp"

#define assert(condition) REQUIRE(condition)
#include <cmath>
#include <algorithm>
#include <random>
#include <vector>

int main() {
    using namespace nbody;

    WorldState frame_world(Dimension::Three);
    const BodyId frame_body_id = frame_world.addBody({{}, {1.0, 2.0, 3.0},
        {0.5, 0.0, -0.5}, 4.0, 0.25, false});
    FramePublisher frame_publisher(2);
    FrameLease first_frame = frame_publisher.publish(frame_world);
    assert(static_cast<bool>(first_frame));
    assert(first_frame->dimension == Dimension::Three);
    assert(first_frame->body_count == 1);
    assert(first_frame->bodies.size() == 1);
    assert(first_frame->bodies[0].id == frame_body_id);
    assert(first_frame->bodies[0].position.x == 1.0);
    frame_world.mutableBody(0).position.x = 9.0;
    FrameLease second_frame = frame_publisher.publish(frame_world);
    assert(second_frame->bodies[0].position.x == 9.0);
    assert(first_frame->bodies[0].position.x == 1.0);
    assert(first_frame->body_count_changed);
    assert(second_frame->body_count_changed == false);

    PhysicsSession session;
    WorldState session_world = WorldState::deterministic(8, Dimension::Two, 19);
    SimulationParameters session_parameters;
    session_parameters.dimension = Dimension::Two;
    assert(session.step(session_world, session_parameters).advanced());
    SolverConfiguration session_simd;
    session_simd.backend = ComputeBackend::SIMD;
    const EngineSwitchResult switch_result = session.switchEngine(session_simd, session_world);
    assert(switch_result.switched());
    assert(session.configuration().backend == ComputeBackend::SIMD);
    assert(session_world.isValid());
    const FrameLease session_frame = session.publishFrame(session_world);
    assert(session_frame->body_count == session_world.bodyCount());
    SolverConfiguration invalid_switch;
    invalid_switch.force_model = ForceModel::BarnesHut;
    assert(!session.switchEngine(invalid_switch, session_world).switched());
    assert(session.configuration().backend == ComputeBackend::SIMD);
    SimulationParameters wrong_dimension = session_parameters;
    wrong_dimension.dimension = Dimension::Three;
    const double session_time = session_world.time();
    const PhysicsStepResult rejected_step = session.step(session_world, wrong_dimension);
    assert(rejected_step.rejected());
    assert(session_world.time() == session_time);

    WorldState scalar_single_world = WorldState::deterministic(48, Dimension::Two, 23);
    WorldState scalar_mt_world = WorldState::deterministic(48, Dimension::Two, 23);
    SimulationParameters scalar_thread_parameters;
    scalar_thread_parameters.dimension = Dimension::Two;
    scalar_thread_parameters.gravitational_constant = 0.1;
    scalar_thread_parameters.timestep = 1e-4;
    scalar_thread_parameters.collision.model = CollisionModel::Transparent;
    ScalarFullPhysicsEngine scalar_single_engine;
    ScalarFullPhysicsEngine scalar_mt_engine;
    scalar_thread_parameters.solver.threading = ThreadingMode::MultiThreaded;
    scalar_thread_parameters.solver.worker_count = 2;
    for (int step = 0; step < 3; ++step) {
        assert(scalar_single_engine.step(scalar_single_world, scalar_thread_parameters).advanced());
        assert(scalar_mt_engine.step(scalar_mt_world, scalar_thread_parameters).advanced());
    }
    assert(nbody_test::worldsHaveNumericalParity(scalar_single_world, scalar_mt_world,
                                                  {1e-10, 1e-10}));
    scalar_thread_parameters.solver.threading = ThreadingMode::SingleThreaded;
    assert(scalar_mt_engine.step(scalar_mt_world, scalar_thread_parameters).advanced());
    scalar_thread_parameters.solver.threading = ThreadingMode::MultiThreaded;
    assert(scalar_mt_engine.step(scalar_mt_world, scalar_thread_parameters).advanced());
    assert(scalar_mt_world.isValid());

    WorldState bh_single_world = WorldState::deterministic(96, Dimension::Two, 29);
    WorldState bh_mt_world = WorldState::deterministic(96, Dimension::Two, 29);
    SimulationParameters bh_thread_parameters;
    bh_thread_parameters.dimension = Dimension::Two;
    bh_thread_parameters.gravitational_constant = 0.1;
    bh_thread_parameters.timestep = 1e-4;
    bh_thread_parameters.collision.model = CollisionModel::Transparent;
    bh_thread_parameters.solver.force_model = ForceModel::BarnesHut;
    bh_thread_parameters.solver.kind = SolverKind::Approximated;
    bh_thread_parameters.solver.threading = ThreadingMode::SingleThreaded;
    ScalarApproximatedPhysicsEngine bh_single_engine;
    ScalarApproximatedPhysicsEngine bh_mt_engine;
    for (int step = 0; step < 3; ++step) {
        assert(bh_single_engine.step(bh_single_world, bh_thread_parameters).advanced());
    }
    bh_thread_parameters.solver.threading = ThreadingMode::MultiThreaded;
    bh_thread_parameters.solver.worker_count = 2;
    for (int step = 0; step < 3; ++step) {
        assert(bh_mt_engine.step(bh_mt_world, bh_thread_parameters).advanced());
    }
    assert(nbody_test::worldsHaveNumericalParity(bh_single_world, bh_mt_world,
                                                  {1e-10, 1e-10}));
    bh_thread_parameters.solver.threading = ThreadingMode::SingleThreaded;
    assert(bh_mt_engine.step(bh_mt_world, bh_thread_parameters).advanced());
    bh_thread_parameters.solver.threading = ThreadingMode::MultiThreaded;
    assert(bh_mt_engine.step(bh_mt_world, bh_thread_parameters).advanced());
    assert(bh_mt_world.isValid());

    WorldState simd_full_single_world = WorldState::deterministic(48, Dimension::Two, 31);
    WorldState simd_full_mt_world = WorldState::deterministic(48, Dimension::Two, 31);
    SimulationParameters simd_full_thread_parameters;
    simd_full_thread_parameters.dimension = Dimension::Two;
    simd_full_thread_parameters.gravitational_constant = 0.1;
    simd_full_thread_parameters.timestep = 1e-4;
    simd_full_thread_parameters.collision.model = CollisionModel::Transparent;
    simd_full_thread_parameters.solver.threading = ThreadingMode::SingleThreaded;
    SimdFullPhysicsEngine simd_full_single_engine;
    SimdFullPhysicsEngine simd_full_mt_engine;
    for (int step = 0; step < 3; ++step) {
        assert(simd_full_single_engine.step(simd_full_single_world,
                                            simd_full_thread_parameters).advanced());
    }
    simd_full_thread_parameters.solver.threading = ThreadingMode::MultiThreaded;
    simd_full_thread_parameters.solver.worker_count = 2;
    for (int step = 0; step < 3; ++step) {
        assert(simd_full_mt_engine.step(simd_full_mt_world,
                                        simd_full_thread_parameters).advanced());
    }
    assert(nbody_test::worldsHaveNumericalParity(simd_full_single_world, simd_full_mt_world,
                                                  {1e-10, 1e-10}));
    simd_full_thread_parameters.solver.threading = ThreadingMode::SingleThreaded;
    assert(simd_full_mt_engine.step(simd_full_mt_world, simd_full_thread_parameters).advanced());
    simd_full_thread_parameters.solver.threading = ThreadingMode::MultiThreaded;
    assert(simd_full_mt_engine.step(simd_full_mt_world, simd_full_thread_parameters).advanced());

    WorldState simd_bh_single_world = WorldState::deterministic(96, Dimension::Two, 37);
    WorldState simd_bh_mt_world = WorldState::deterministic(96, Dimension::Two, 37);
    SimulationParameters simd_bh_thread_parameters;
    simd_bh_thread_parameters.dimension = Dimension::Two;
    simd_bh_thread_parameters.gravitational_constant = 0.1;
    simd_bh_thread_parameters.timestep = 1e-4;
    simd_bh_thread_parameters.collision.model = CollisionModel::Transparent;
    simd_bh_thread_parameters.solver.force_model = ForceModel::BarnesHut;
    simd_bh_thread_parameters.solver.kind = SolverKind::Approximated;
    simd_bh_thread_parameters.solver.threading = ThreadingMode::SingleThreaded;
    SimdApproximatedPhysicsEngine simd_bh_single_engine;
    SimdApproximatedPhysicsEngine simd_bh_mt_engine;
    for (int step = 0; step < 3; ++step) {
        assert(simd_bh_single_engine.step(simd_bh_single_world,
                                          simd_bh_thread_parameters).advanced());
    }
    simd_bh_thread_parameters.solver.threading = ThreadingMode::MultiThreaded;
    simd_bh_thread_parameters.solver.worker_count = 2;
    for (int step = 0; step < 3; ++step) {
        assert(simd_bh_mt_engine.step(simd_bh_mt_world,
                                      simd_bh_thread_parameters).advanced());
    }
    assert(nbody_test::worldsHaveNumericalParity(simd_bh_single_world, simd_bh_mt_world,
                                                  {1e-10, 1e-10}));
    simd_bh_thread_parameters.solver.threading = ThreadingMode::SingleThreaded;
    assert(simd_bh_mt_engine.step(simd_bh_mt_world, simd_bh_thread_parameters).advanced());
    simd_bh_thread_parameters.solver.threading = ThreadingMode::MultiThreaded;
    assert(simd_bh_mt_engine.step(simd_bh_mt_world, simd_bh_thread_parameters).advanced());

    const auto makeParityScenario = [](Dimension dimension, std::size_t body_count,
                                       unsigned long long seed) {
        WorldState world = WorldState::deterministic(body_count, dimension, seed);
        for (std::size_t index = 0; index < body_count; ++index) {
            MutableBodyView body = world.mutableBody(index);
            const double column = static_cast<double>(index % 16);
            const double row = static_cast<double>(index / 16);
            body.position.x = column * 0.8 - 6.0;
            body.position.y = row * 0.8 - 6.0;
            body.position.z = dimension == Dimension::Three
                ? static_cast<double>(index % 5) * 0.6 - 1.2 : 0.0;
            body.velocity.x = 0.01 * (static_cast<double>(index % 7) - 3.0);
            body.velocity.y = -0.01 * (static_cast<double>(index % 5) - 2.0);
            body.velocity.z = dimension == Dimension::Three
                ? 0.01 * (static_cast<double>(index % 3) - 1.0) : 0.0;
            body.static_flag = index % 17 == 0 ? 1 : 0;
        }
        return world;
    };

    const auto compareEngineParity = [&](IPhysicsEngine& reference_engine,
                                          IPhysicsEngine& candidate_engine,
                                          Dimension dimension, SolverKind kind,
                                          std::size_t body_count, int steps,
                                          unsigned long long seed,
                                          nbody_test::NumericalParityTolerance tolerance) {
        WorldState reference = makeParityScenario(dimension, body_count, seed);
        WorldState candidate = makeParityScenario(dimension, body_count, seed);
        SimulationParameters parameters;
        parameters.dimension = dimension;
        parameters.gravitational_constant = 0.1;
        parameters.timestep = 1e-4;
        parameters.softening_length = 1e-3;
        parameters.collision.model = CollisionModel::Transparent;
        parameters.solver.kind = kind;
        parameters.solver.force_model = kind == SolverKind::Approximated
            ? ForceModel::BarnesHut : ForceModel::Full;
        parameters.solver.barnes_hut.opening_angle = 0.5;
        for (int step = 0; step < steps; ++step) {
            assert(reference_engine.step(reference, parameters).advanced());
            assert(candidate_engine.step(candidate, parameters).advanced());
            assert(nbody_test::worldsHaveNumericalParity(reference, candidate, tolerance));
        }
    };

    SolverConfiguration scalar_full_configuration;
    const std::unique_ptr<IPhysicsEngine> scalar_full_engine =
        createPhysicsEngine(scalar_full_configuration);
    assert(scalar_full_engine->info(Dimension::Two).backend == ComputeBackend::Scalar);
    assert(scalar_full_engine->info(Dimension::Two).kind == SolverKind::Full);

    SolverConfiguration scalar_approximate_configuration;
    scalar_approximate_configuration.kind = SolverKind::Approximated;
    scalar_approximate_configuration.force_model = ForceModel::BarnesHut;
    const std::unique_ptr<IPhysicsEngine> scalar_approximate_engine =
        createPhysicsEngine(scalar_approximate_configuration);
    assert(scalar_approximate_engine->info(Dimension::Two).backend == ComputeBackend::Scalar);
    assert(scalar_approximate_engine->info(Dimension::Two).kind == SolverKind::Approximated);

    SolverConfiguration simd_full_configuration;
    simd_full_configuration.backend = ComputeBackend::SIMD;
    const std::unique_ptr<IPhysicsEngine> simd_full_engine =
        createPhysicsEngine(simd_full_configuration);
    assert(simd_full_engine->info(Dimension::Two).backend == ComputeBackend::SIMD);

    SolverConfiguration gpu_configuration;
    gpu_configuration.backend = ComputeBackend::GPU;
    const std::unique_ptr<IPhysicsEngine> gpu_engine = createPhysicsEngine(gpu_configuration);
    assert(gpu_engine->info(Dimension::Two).backend == ComputeBackend::Scalar);

    const SolverConfigurationValidation valid_configuration =
        validateSolverConfiguration(scalar_full_configuration);
    assert(valid_configuration.valid);
    SolverConfiguration invalid_configuration;
    invalid_configuration.force_model = ForceModel::BarnesHut;
    assert(!validateSolverConfiguration(invalid_configuration).valid);

    WorldState engine_contract_world = WorldState::deterministic(4, Dimension::Two, 7);
    SimulationParameters engine_contract_parameters;
    engine_contract_parameters.dimension = Dimension::Two;
    const PhysicsStepResult engine_step =
        scalar_full_engine->step(engine_contract_world, engine_contract_parameters);
    assert(engine_step.advanced());
    assert(engine_step.body_count == engine_contract_world.bodyCount());
    assert(engine_step.simulation_time == engine_contract_world.time());

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
    ScalarFullPhysicsEngine full_solver;
    ScalarApproximatedPhysicsEngine barnes_solver;
    full_solver.step(full_world, full_parameters);
    barnes_solver.step(barnes_world, barnes_parameters);
    for (std::size_t index = 0; index < full_world.bodyCount(); ++index) {
        assert((full_world.body(index).velocity - barnes_world.body(index).velocity).length(Dimension::Two) < 1e-5);
    }

    WorldState scalar_simd_reference = WorldState::deterministic(120, Dimension::Three, 131);
    WorldState simd_world = WorldState::deterministic(120, Dimension::Three, 131);
    SimulationParameters simd_parameters;
    simd_parameters.dimension = Dimension::Three;
    simd_parameters.gravitational_constant = 0.1;
    simd_parameters.timestep = 1e-4;
    simd_parameters.collision.model = CollisionModel::Transparent;
    ScalarFullPhysicsEngine scalar_reference_solver;
    SimdFullPhysicsEngine simd_solver;
    scalar_reference_solver.step(scalar_simd_reference, simd_parameters);
    simd_solver.step(simd_world, simd_parameters);
    for (std::size_t index = 0; index < scalar_simd_reference.bodyCount(); ++index) {
        assert((scalar_simd_reference.body(index).position - simd_world.body(index).position).length(Dimension::Three) < 1e-10);
        assert((scalar_simd_reference.body(index).velocity - simd_world.body(index).velocity).length(Dimension::Three) < 1e-10);
    }

    WorldState scalar_barnes_reference = WorldState::deterministic(120, Dimension::Three, 151);
    WorldState simd_barnes_world = WorldState::deterministic(120, Dimension::Three, 151);
    SimulationParameters barnes_simd_parameters = simd_parameters;
    barnes_simd_parameters.solver.barnes_hut.opening_angle = 0.5;
    ScalarApproximatedPhysicsEngine scalar_barnes_solver;
    SimdApproximatedPhysicsEngine simd_barnes_solver;
    scalar_barnes_solver.step(scalar_barnes_reference, barnes_simd_parameters);
    simd_barnes_solver.step(simd_barnes_world, barnes_simd_parameters);
    for (std::size_t index = 0; index < scalar_barnes_reference.bodyCount(); ++index) {
        assert((scalar_barnes_reference.body(index).position - simd_barnes_world.body(index).position).length(Dimension::Three) < 1e-9);
        assert((scalar_barnes_reference.body(index).velocity - simd_barnes_world.body(index).velocity).length(Dimension::Three) < 1e-9);
    }

    ScalarFullPhysicsEngine parity_scalar_full_2d;
    SimdFullPhysicsEngine parity_simd_full_2d;
    compareEngineParity(parity_scalar_full_2d, parity_simd_full_2d, Dimension::Two,
                        SolverKind::Full, 64, 8, 401, {1e-9, 1e-9});

    ScalarFullPhysicsEngine parity_scalar_full_3d;
    SimdFullPhysicsEngine parity_simd_full_3d;
    compareEngineParity(parity_scalar_full_3d, parity_simd_full_3d, Dimension::Three,
                        SolverKind::Full, 64, 8, 403, {1e-9, 1e-9});

    ScalarApproximatedPhysicsEngine parity_scalar_barnes_2d;
    SimdApproximatedPhysicsEngine parity_simd_barnes_2d;
    compareEngineParity(parity_scalar_barnes_2d, parity_simd_barnes_2d, Dimension::Two,
                        SolverKind::Approximated, 256, 6, 407, {1e-9, 1e-9});

    ScalarApproximatedPhysicsEngine parity_scalar_barnes_3d;
    SimdApproximatedPhysicsEngine parity_simd_barnes_3d;
    compareEngineParity(parity_scalar_barnes_3d, parity_simd_barnes_3d, Dimension::Three,
                        SolverKind::Approximated, 128, 6, 409, {1e-9, 1e-9});

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
    ScalarFullPhysicsEngine solver;
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

    WorldState collision_mode_switch(Dimension::Two);
    collision_mode_switch.addBody({{}, {-0.25, 0.0, 0.0}, {}, 1.0, 0.5, false});
    collision_mode_switch.addBody({{}, {0.25, 0.0, 0.0}, {}, 1.0, 0.5, false});
    CollisionSettings hard_body_settings;
    hard_body_settings.model = CollisionModel::HardBody;
    hard_body_settings.restitution = 0.0;
    hard_body_settings.maximum_consistency_passes = 50;
    ScalarCollisionSystem::resolveContacts(collision_mode_switch, hard_body_settings, 0.0);
    assert(collision_mode_switch.activeCollisionModel() == CollisionModel::HardBody);
    assert(!collision_mode_switch.collisionTransitionDiagnostics().failed);

    assert((collision_mode_switch.body(1).position - collision_mode_switch.body(0).position).length(Dimension::Two)
        >= collision_mode_switch.body(0).radius + collision_mode_switch.body(1).radius - 1e-12);
    CollisionSettings transparent_settings;
    transparent_settings.model = CollisionModel::Transparent;
    ScalarCollisionSystem::resolveContacts(collision_mode_switch, transparent_settings, 0.0);
    assert(collision_mode_switch.activeCollisionModel() == CollisionModel::Transparent);
    assert(!collision_mode_switch.collisionTransitionDiagnostics().failed);

    bool transparent_fragmentation_rejected = false;
    transparent_settings.classifier.fragmentation_enabled = true;
    try {
        ScalarCollisionSystem::resolveContacts(collision_mode_switch, transparent_settings, 0.0);
    } catch (const std::invalid_argument&) {
        transparent_fragmentation_rejected = true;
    }
    assert(transparent_fragmentation_rejected);

    bool transparent_absorption_rejected = false;
    transparent_settings.classifier.fragmentation_enabled = false;
    transparent_settings.classifier.absorption_enabled = true;
    try {
        ScalarCollisionSystem::resolveContacts(collision_mode_switch, transparent_settings, 0.0);
    } catch (const std::invalid_argument&) {
        transparent_absorption_rejected = true;
    }
    assert(transparent_absorption_rejected);

    CollisionSettings hard_body_absorption = hard_body_settings;
    hard_body_absorption.classifier.absorption_enabled = true;
    ScalarCollisionSystem::resolveContacts(collision_mode_switch, hard_body_absorption, 0.0);
    assert(collision_mode_switch.activeCollisionModel() == CollisionModel::HardBody);

    WorldState gas_absorber(Dimension::Two);
    gas_absorber.addBody({{}, {-0.25, 0.0, 0.0}, {1.0, 0.0, 0.0}, 10.0, 2.0, false});
    gas_absorber.addBody({{}, {0.25, 0.0, 0.0}, {-1.0, 0.0, 0.0}, 1.0, 0.5, false});
    gas_absorber.mutableBody(0).kind = BodyKind::Gas;
    ScalarCollisionSystem::resolveContacts(gas_absorber, hard_body_absorption, 0.0);
    ScalarCollisionSystem::applyDeferredOutcomes(gas_absorber, hard_body_absorption);
    assert(gas_absorber.bodyCount() == 1);
    assert(gas_absorber.body(0).id.value == 1);
    assert(gas_absorber.body(0).kind == BodyKind::Gas);
    assert(gas_absorber.body(0).mass == 11.0);
    assert(std::abs(gas_absorber.body(0).velocity.x - 9.0 / 11.0) < 1e-12);
    assert(gas_absorber.body(0).accumulated_damage == 0.0);

    WorldState static_absorber(Dimension::Two);
    static_absorber.addBody({{}, {}, {}, 10.0, 2.0, true});
    static_absorber.addBody({{}, {}, {}, 1.0, 0.5, true});
    static_absorber.mutableBody(0).kind = BodyKind::Gas;
    static_absorber.setActiveCollisionModel(CollisionModel::HardBody);
    ScalarCollisionSystem::resolveContacts(static_absorber, hard_body_absorption, 0.0);
    ScalarCollisionSystem::applyDeferredOutcomes(static_absorber, hard_body_absorption);
    assert(static_absorber.bodyCount() == 1);
    assert(static_absorber.body(0).kind == BodyKind::Gas);
    assert(static_absorber.body(0).is_static());
    assert(static_absorber.body(0).mass == 11.0);

    WorldState dominant_solid(Dimension::Two);
    dominant_solid.addBody({{}, {-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0, 1.0, false});
    dominant_solid.addBody({{}, {1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, 3.0, 3.0, false});
    dominant_solid.mutableBody(0).kind = BodyKind::Gas;
    ScalarCollisionSystem::resolveContacts(dominant_solid, hard_body_absorption, 0.0);
    ScalarCollisionSystem::applyDeferredOutcomes(dominant_solid, hard_body_absorption);
    assert(dominant_solid.bodyCount() == 1);
    assert(dominant_solid.body(0).id.value == 2);
    assert(dominant_solid.body(0).kind == BodyKind::Ordinary);
    assert(dominant_solid.body(0).mass == 3.0);
    assert(dominant_solid.body(0).radius == 3.0);
    assert(std::abs(dominant_solid.body(0).velocity.x + 0.5) < 1e-12);

    WorldState black_hole(Dimension::Two);
    black_hole.addBody({{}, {-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0, 1.0, false});
    black_hole.addBody({{}, {1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, 100.0, 100.0, false});
    black_hole.mutableBody(0).kind = BodyKind::BlackHole;
    ScalarCollisionSystem::resolveContacts(black_hole, hard_body_absorption, 0.0);
    ScalarCollisionSystem::applyDeferredOutcomes(black_hole, hard_body_absorption);
    assert(black_hole.bodyCount() == 1);
    assert(black_hole.body(0).id.value == 1);
    assert(black_hole.body(0).kind == BodyKind::BlackHole);
    assert(black_hole.body(0).mass == 101.0);

    WorldState failed_collision_mode_switch(Dimension::Two);
    failed_collision_mode_switch.addBody({{}, {}, {}, 1.0, 1.0, true});
    failed_collision_mode_switch.addBody({{}, {}, {}, 1.0, 1.0, true});
    CollisionSettings transparent_failure_settings;
    transparent_failure_settings.model = CollisionModel::HardBody;
    transparent_failure_settings.maximum_consistency_passes = 50;
    ScalarCollisionSystem::resolveContacts(failed_collision_mode_switch, transparent_failure_settings, 0.0);
    const CollisionTransitionDiagnostics& transition =
        failed_collision_mode_switch.collisionTransitionDiagnostics();
    assert(failed_collision_mode_switch.activeCollisionModel() == CollisionModel::Transparent);
    assert(transition.failed);
    assert(transition.passes == 50);
    assert(transition.unresolved_bodies.size() == 2);

    WorldState scalar_collision_backend(Dimension::Two);
    WorldState simd_collision_backend(Dimension::Two);
    const BodyState collision_body_a{{}, {-0.75, 0.0, 0.0}, {1.0, 0.0, 0.0}, 2.0, 1.0, false};
    const BodyState collision_body_b{{}, {0.75, 0.0, 0.0}, {-1.0, 0.0, 0.0}, 1.0, 1.0, false};
    const BodyState collision_body_c{{}, {8.0, 0.0, 0.0}, {}, 1.0, 0.5, false};
    scalar_collision_backend.addBody(collision_body_a);
    scalar_collision_backend.addBody(collision_body_b);
    scalar_collision_backend.addBody(collision_body_c);
    simd_collision_backend.addBody(collision_body_a);
    simd_collision_backend.addBody(collision_body_b);
    simd_collision_backend.addBody(collision_body_c);
    scalar_collision_backend.setActiveCollisionModel(CollisionModel::HardBody);
    simd_collision_backend.setActiveCollisionModel(CollisionModel::HardBody);
    ScalarCollisionSystem::resolveContacts(scalar_collision_backend, hard_body_settings, 0.0);
    SimdCollisionSystem::resolveContacts(simd_collision_backend, hard_body_settings, 0.0);
    assert(scalar_collision_backend.collisionEvents().size() == simd_collision_backend.collisionEvents().size());
    for (std::size_t index = 0; index < scalar_collision_backend.bodyCount(); ++index) {
        assert(scalar_collision_backend.body(index).position == simd_collision_backend.body(index).position);
        assert(scalar_collision_backend.body(index).velocity == simd_collision_backend.body(index).velocity);
    }

    WorldState multi_pass_collision_mode_switch(Dimension::Two);
    for (std::size_t index = 0; index < 3; ++index) {
        multi_pass_collision_mode_switch.addBody({{}, {static_cast<double>(index), 0.0, 0.0}, {}, 1.0, 1.0, false});
    }
    ScalarCollisionSystem::resolveContacts(multi_pass_collision_mode_switch, hard_body_settings, 0.0);
    const CollisionTransitionDiagnostics& multi_pass_transition =
        multi_pass_collision_mode_switch.collisionTransitionDiagnostics();
    assert(multi_pass_collision_mode_switch.activeCollisionModel() == CollisionModel::HardBody);
    assert(!multi_pass_transition.failed);
    assert(multi_pass_transition.passes >= 2);
    for (std::size_t first = 0; first < multi_pass_collision_mode_switch.bodyCount(); ++first) {
        for (std::size_t second = first + 1; second < multi_pass_collision_mode_switch.bodyCount(); ++second) {
            const Vec3 displacement = multi_pass_collision_mode_switch.body(second).position
                - multi_pass_collision_mode_switch.body(first).position;
            const double combined_radius = multi_pass_collision_mode_switch.body(first).radius
                + multi_pass_collision_mode_switch.body(second).radius;
            assert(displacement.lengthSquared(Dimension::Two)
                >= combined_radius * combined_radius - 1e-10);
        }
    }

    WorldState dense_dynamic_collision_mode_switch(Dimension::Two);
    for (std::size_t index = 0; index < 6; ++index) {
        dense_dynamic_collision_mode_switch.addBody({{}, {static_cast<double>(index), 0.0, 0.0}, {}, 1.0, 1.0, false});
    }
    const std::vector<Vec3> dense_dynamic_original_positions = {
        dense_dynamic_collision_mode_switch.body(0).position,
        dense_dynamic_collision_mode_switch.body(1).position,
        dense_dynamic_collision_mode_switch.body(2).position,
        dense_dynamic_collision_mode_switch.body(3).position,
        dense_dynamic_collision_mode_switch.body(4).position,
        dense_dynamic_collision_mode_switch.body(5).position};
    ScalarCollisionSystem::resolveContacts(dense_dynamic_collision_mode_switch, hard_body_settings, 0.0);
    const CollisionTransitionDiagnostics& dense_dynamic_transition =
        dense_dynamic_collision_mode_switch.collisionTransitionDiagnostics();
    assert(dense_dynamic_collision_mode_switch.activeCollisionModel() == CollisionModel::Transparent);
    assert(dense_dynamic_transition.failed);
    assert(dense_dynamic_transition.passes == 50);
    assert(!dense_dynamic_transition.unresolved_bodies.empty());
    for (std::size_t index = 0; index < dense_dynamic_original_positions.size(); ++index) {
        assert(dense_dynamic_collision_mode_switch.body(index).position == dense_dynamic_original_positions[index]);
    }

    WorldState dense_failed_collision_mode_switch(Dimension::Two);
    for (std::size_t index = 0; index < 4; ++index) {
        dense_failed_collision_mode_switch.addBody({{}, {0.0, 0.0, 0.0}, {}, 1.0, 1.0, true});
    }
    for (std::size_t index = 0; index < 4; ++index) {
        dense_failed_collision_mode_switch.addBody({{}, {0.25 * static_cast<double>(index), 0.0, 0.0}, {}, 1.0, 1.0, false});
    }
    const std::vector<Vec3> dense_failed_original_positions = {
        dense_failed_collision_mode_switch.body(0).position,
        dense_failed_collision_mode_switch.body(1).position,
        dense_failed_collision_mode_switch.body(2).position,
        dense_failed_collision_mode_switch.body(3).position,
        dense_failed_collision_mode_switch.body(4).position,
        dense_failed_collision_mode_switch.body(5).position,
        dense_failed_collision_mode_switch.body(6).position,
        dense_failed_collision_mode_switch.body(7).position};
    ScalarCollisionSystem::resolveContacts(dense_failed_collision_mode_switch, hard_body_settings, 0.0);
    const CollisionTransitionDiagnostics& dense_transition =
        dense_failed_collision_mode_switch.collisionTransitionDiagnostics();
    assert(dense_failed_collision_mode_switch.activeCollisionModel() == CollisionModel::Transparent);
    assert(dense_transition.failed);
    assert(dense_transition.passes == 50);
    assert(dense_transition.unresolved_bodies.size() >= 4);
    for (std::size_t index = 0; index < dense_failed_original_positions.size(); ++index) {
        assert(dense_failed_collision_mode_switch.body(index).position == dense_failed_original_positions[index]);
    }

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
    for (std::size_t first = 0; first < fragmenting.bodyCount(); ++first) {
        assert(fragmenting.body(first).id.isValid());
        for (std::size_t second = first + 1; second < fragmenting.bodyCount(); ++second) {
            assert(!(fragmenting.body(first).id == fragmenting.body(second).id));
        }
    }

    WorldState fragmenting_3d(Dimension::Three);
    fragmenting_3d.addBody({{}, {-0.5, 0.0, 0.0}, {1.0, 2.0, 3.0}, 8.0, 1.0, false});
    fragmenting_3d.addBody({{}, {0.5, 0.0, 0.0}, {-1.0, -1.0, -2.0}, 4.0, 1.0, false});
    SimulationParameters fragment_parameters;
    fragment_parameters.dimension = Dimension::Three;
    fragment_parameters.gravitational_constant = 0.0;
    fragment_parameters.timestep = 0.0;
    fragment_parameters.collision.model = CollisionModel::HardBody;
    fragment_parameters.collision.minimum_fragments = 4;
    fragment_parameters.collision.maximum_fragments = 4;
    fragment_parameters.collision.maximum_fragment_count = 32;
    fragment_parameters.collision.classifier.force_fragmentation = true;
    const Vec3 initial_center_of_mass =
        (fragmenting_3d.body(0).position * fragmenting_3d.body(0).mass
         + fragmenting_3d.body(1).position * fragmenting_3d.body(1).mass) * (1.0 / 12.0);
    const Vec3 initial_momentum = fragmenting_3d.body(0).velocity * fragmenting_3d.body(0).mass
        + fragmenting_3d.body(1).velocity * fragmenting_3d.body(1).mass;
    ScalarFullPhysicsEngine fragment_solver;
    fragment_solver.step(fragmenting_3d, fragment_parameters);
    assert(fragmenting_3d.bodyCount() == 8);
    double fragmenting_3d_mass = 0.0;
    Vec3 final_weighted_position{};
    Vec3 final_momentum{};
    bool has_z_spread = false;
    for (std::size_t index = 0; index < fragmenting_3d.bodyCount(); ++index) {
        const ConstBodyView body = fragmenting_3d.body(index);
        fragmenting_3d_mass += body.mass;
        final_weighted_position += body.position * body.mass;
        final_momentum += body.velocity * body.mass;
        has_z_spread = has_z_spread || std::abs(body.position.z) > 1e-12;
    }
    assert(std::abs(fragmenting_3d_mass - 12.0) < 1e-12);
    assert((final_weighted_position * (1.0 / fragmenting_3d_mass) - initial_center_of_mass)
        .length(Dimension::Three) < 1e-10);
    assert((final_momentum - initial_momentum).length(Dimension::Three) < 1e-10);
    assert(has_z_spread);
    assert(fragmenting_3d.isValid());

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
