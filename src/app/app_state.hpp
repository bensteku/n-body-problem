#pragma once

#include "app/presentation_state.hpp"
#include "app/simulation_session.hpp"

namespace nbody::app {

enum class AppMode {
    MainMenu,
    Simulation,
    Benchmark
};

struct ApplicationState {
    AppMode mode{AppMode::MainMenu};
    SimulationSession session;
    PresentationState presentation;
};

}
