#include "simulation/world_state.hpp"

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
}
