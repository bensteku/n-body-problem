#pragma once

#include "simulation/body_id.hpp"

#include <array>
#include <optional>
#include <vector>

namespace nbody::app {

struct CameraState {
    double x{};
    double y{};
    double z{};
    double zoom{1.0};
};

struct CameraFocus {
    enum class Mode { Free, FollowBody, FollowOrigin };

    Mode mode{Mode::Free};
    std::optional<BodyId> body;
};

struct SelectionState {
    std::optional<BodyId> primary;
    std::vector<BodyId> secondary;
};

struct InspectorState {
    enum class Presentation { StaticPanel, WorldOverlay };

    std::optional<BodyId> body;
    Presentation presentation{Presentation::StaticPanel};
    bool pinned{};
};

enum class PanelEdge { Top, Bottom, Left, Right };

struct FoldablePanelState {
    PanelEdge edge{PanelEdge::Top};
    bool expanded{};
    bool pinned{};
    float animation_progress{};
    float hover_time{};
};

struct PanelLayoutState {
    std::array<FoldablePanelState, 4> panels{{
        {PanelEdge::Top, true, false, 1.0f, 0.0f},
        {PanelEdge::Bottom, true, false, 1.0f, 0.0f},
        {PanelEdge::Left, false, false, 0.0f, 0.0f},
        {PanelEdge::Right, false, false, 0.0f, 0.0f}
    }};
    float reveal_dwell_seconds{1.0f};
};

enum class TimeDisplayUnit { Seconds, Days, Months, Years };
enum class DistanceDisplayUnit { Kilometers, AstronomicalUnits, LightYears };

struct UnitPreferences {
    TimeDisplayUnit time{TimeDisplayUnit::Seconds};
    bool automatic_distance{true};
    DistanceDisplayUnit distance{DistanceDisplayUnit::Kilometers};
};

struct GridSettings {
    bool visible{};
};

struct TrajectorySettings {
    bool visible{};
    double duration_seconds{60.0};
};

struct PresentationState {
    CameraState camera;
    CameraFocus focus;
    SelectionState selection;
    InspectorState inspector;
    PanelLayoutState panels;
    UnitPreferences units;
    GridSettings grid;
    TrajectorySettings trajectories;
};

}
