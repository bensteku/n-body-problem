#include "simulation/world_state.hpp"

#include <iostream>
#include <string_view>

namespace {

nbody::Dimension parseDimension(int argc, char** argv) {
    if (argc > 1 && (std::string_view(argv[1]) == "3" || std::string_view(argv[1]) == "3D")) {
        return nbody::Dimension::Three;
    }
    return nbody::Dimension::Two;
}

}

int main(int argc, char** argv) {
    constexpr unsigned long long seed = 42;
    const nbody::Dimension dimension = parseDimension(argc, argv);
    nbody::WorldState world = nbody::WorldState::deterministic(3, dimension, seed);
    world.advance(0.01);
    const nbody::WorldDiagnostics diagnostics = world.diagnostics();

    std::cout << "nbody greenfield scaffold\n"
              << "dimension=" << (dimension == nbody::Dimension::Two ? "2D" : "3D") << "\n"
              << "bodies=" << world.bodies().size() << "\n"
              << "time=" << world.time() << "\n"
              << "total_mass=" << diagnostics.total_mass << "\n"
              << "finite=" << (diagnostics.finite ? "true" : "false") << '\n';
    return world.isValid() ? 0 : 1;
}
