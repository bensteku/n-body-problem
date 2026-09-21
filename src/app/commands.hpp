#pragma once

#include "app/app_state.hpp"

#include <optional>
#include <filesystem>
#include <string>
#include <string_view>
#include <variant>

namespace nbody::app {

struct StartSimulation {};
struct FinishStartupEdit {};
struct EnterEditMode {};
struct LeaveEditMode {};
struct PauseSimulation {};
struct ResumeSimulation {};
struct CaptureInitialState {};
struct RestoreInitialState {};
struct ResetToInitialEdit {};
struct SwitchDimension {
    Dimension dimension;
};
struct SaveSnapshot {
    std::filesystem::path path;
    std::string name;
};
struct LoadSnapshot {
    std::filesystem::path path;
};

struct SelectBody {
    std::optional<BodyId> body;
};

struct FocusBody {
    std::optional<BodyId> body;
};

struct CreateBody {
    BodyState body;
};
struct ClearBodies {};

struct SwitchSolver {
    SolverConfiguration configuration;
};

struct SetTimeDisplayUnit {
    TimeDisplayUnit unit;
};

struct SetAutomaticDistanceUnits {
    bool enabled;
};

struct SetDistanceDisplayUnit {
    DistanceDisplayUnit unit;
};
struct SetGridVisible {
    bool visible;
};

using Command = std::variant<
    StartSimulation,
    FinishStartupEdit,
    EnterEditMode,
    LeaveEditMode,
    PauseSimulation,
    ResumeSimulation,
    CaptureInitialState,
    RestoreInitialState,
    ResetToInitialEdit,
    SwitchDimension,
    SaveSnapshot,
    LoadSnapshot,
    SelectBody,
    FocusBody,
    CreateBody,
    ClearBodies,
    SwitchSolver,
    SetTimeDisplayUnit,
    SetAutomaticDistanceUnits,
    SetDistanceDisplayUnit,
    SetGridVisible>;

enum class CommandStatus { Applied, Rejected };

struct CommandResult {
    CommandStatus status{CommandStatus::Rejected};
    std::string message;
    std::optional<BodyId> affected_body;

    bool applied() const { return status == CommandStatus::Applied; }
};

class CommandDispatcher {
public:
    explicit CommandDispatcher(ApplicationState& application) : application_(application) {}

    CommandResult dispatch(const Command& command);

private:
    bool containsBody(BodyId body) const;
    CommandResult reject(std::string_view message) const;

    ApplicationState& application_;
};

}
