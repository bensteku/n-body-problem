#include "app/commands.hpp"

#include "io/snapshot.hpp"

#include <type_traits>
#include <utility>

namespace nbody::app {

CommandResult CommandDispatcher::dispatch(const Command& command) {
    return std::visit([this](const auto& value) -> CommandResult {
        using CommandType = std::decay_t<decltype(value)>;

        if constexpr (std::is_same_v<CommandType, StartSimulation>) {
            if (application_.mode != AppMode::MainMenu) {
                return reject("simulation is already active");
            }
            application_.mode = AppMode::Simulation;
            application_.session.setMode(SimulationMode::StartupEdit);
            return {CommandStatus::Applied, "entered Startup Edit Mode", {}};
        } else if constexpr (std::is_same_v<CommandType, FinishStartupEdit>) {
            if (application_.mode != AppMode::Simulation
                || application_.session.mode() != SimulationMode::StartupEdit) {
                return reject("not in Startup Edit Mode");
            }
            application_.session.captureInitialState();
            application_.session.setMode(SimulationMode::Paused);
            return {CommandStatus::Applied, "Startup Edit Mode finished", {}};
        } else if constexpr (std::is_same_v<CommandType, EnterEditMode>) {
            if (application_.mode != AppMode::Simulation) {
                return reject("simulation workspace is not active");
            }
            const SimulationMode mode = application_.session.mode();
            if (mode != SimulationMode::Running && mode != SimulationMode::Paused) {
                return reject("simulation is not in a mode that can enter Edit Mode");
            }
            application_.session.setMode(SimulationMode::Edit);
            return {CommandStatus::Applied, "entered Edit Mode", {}};
        } else if constexpr (std::is_same_v<CommandType, LeaveEditMode>) {
            if (application_.session.mode() != SimulationMode::Edit) {
                return reject("simulation is not in Edit Mode");
            }
            application_.session.setMode(SimulationMode::Paused);
            return {CommandStatus::Applied, "left Edit Mode", {}};
        } else if constexpr (std::is_same_v<CommandType, PauseSimulation>) {
            if (application_.session.mode() != SimulationMode::Running) {
                return reject("simulation is not running");
            }
            application_.session.setMode(SimulationMode::Paused);
            return {CommandStatus::Applied, "simulation paused", {}};
        } else if constexpr (std::is_same_v<CommandType, ResumeSimulation>) {
            if (application_.session.mode() != SimulationMode::Paused) {
                return reject("simulation is not paused");
            }
            application_.session.setMode(SimulationMode::Running);
            return {CommandStatus::Applied, "simulation resumed", {}};
        } else if constexpr (std::is_same_v<CommandType, CaptureInitialState>) {
            application_.session.captureInitialState();
            return {CommandStatus::Applied, "initial state captured", {}};
        } else if constexpr (std::is_same_v<CommandType, RestoreInitialState>) {
            if (!application_.session.restoreInitialState()) {
                return reject("no initial state has been captured");
            }
            application_.presentation.selection = {};
            application_.presentation.focus = {};
            application_.presentation.inspector = {};
            return {CommandStatus::Applied, "initial state restored", {}};
        } else if constexpr (std::is_same_v<CommandType, ResetToInitialEdit>) {
            if (application_.mode != AppMode::Simulation) {
                return reject("simulation workspace is not active");
            }
            const Dimension dimension = application_.session.state().parameters.dimension;
            SimulationState state;
            state.world = WorldState(dimension);
            state.parameters.dimension = dimension;
            application_.session = SimulationSession(std::move(state));
            application_.session.setMode(SimulationMode::StartupEdit);
            application_.presentation = {};
            return {CommandStatus::Applied, "reset to Initial Edit Mode", {}};
        } else if constexpr (std::is_same_v<CommandType, SwitchDimension>) {
            if (application_.mode != AppMode::Simulation
                || application_.session.mode() != SimulationMode::StartupEdit) {
                return reject("dimension can only be switched in Initial Edit Mode");
            }
            SimulationState state;
            state.world = WorldState(value.dimension);
            state.parameters.dimension = value.dimension;
            application_.session = SimulationSession(std::move(state));
            application_.session.setMode(SimulationMode::StartupEdit);
            application_.presentation = {};
            return {CommandStatus::Applied, "dimension switched and setup reset", {}};
        } else if constexpr (std::is_same_v<CommandType, SaveSnapshot>) {
            io::SimulationSnapshot snapshot;
            snapshot.state = application_.session.state();
            snapshot.metadata.name = value.name;
            const io::SnapshotResult result = io::SnapshotSerializer::save(value.path, snapshot);
            if (!result.succeeded()) return reject(result.message);
            return {CommandStatus::Applied, "snapshot saved", {}};
        } else if constexpr (std::is_same_v<CommandType, LoadSnapshot>) {
            if (application_.mode != AppMode::MainMenu) {
                return reject("snapshots can only be loaded from the main menu");
            }
            const io::SnapshotResult result = io::SnapshotSerializer::load(value.path);
            if (!result.succeeded()) return reject(result.message);
            application_.session = SimulationSession(result.snapshot.state);
            application_.session.setMode(SimulationMode::StartupEdit);
            application_.mode = AppMode::Simulation;
            application_.presentation = {};
            return {CommandStatus::Applied, "snapshot loaded into Startup Edit Mode", {}};
        } else if constexpr (std::is_same_v<CommandType, SelectBody>) {
            if (value.body && !containsBody(*value.body)) {
                return reject("cannot select an unknown body");
            }
            application_.presentation.selection.primary = value.body;
            application_.presentation.inspector.body = value.body;
            if (!value.body) application_.presentation.inspector = {};
            return {CommandStatus::Applied, "selection updated", value.body};
        } else if constexpr (std::is_same_v<CommandType, FocusBody>) {
            if (value.body && !containsBody(*value.body)) {
                return reject("cannot focus an unknown body");
            }
            if (value.body) {
                application_.presentation.focus.mode = CameraFocus::Mode::FollowBody;
                application_.presentation.focus.body = value.body;
            } else {
                application_.presentation.focus = {};
            }
            return {CommandStatus::Applied, "camera focus updated", value.body};
        } else if constexpr (std::is_same_v<CommandType, CreateBody>) {
            if (application_.mode != AppMode::Simulation
                || application_.session.mode() != SimulationMode::Edit
                && application_.session.mode() != SimulationMode::StartupEdit) {
                return reject("bodies can only be created in Edit Mode");
            }
            const BodyId body = application_.session.state().world.addBody(value.body);
            return {CommandStatus::Applied, "body created", body};
        } else if constexpr (std::is_same_v<CommandType, ClearBodies>) {
            if (application_.mode != AppMode::Simulation
                || (application_.session.mode() != SimulationMode::Edit
                    && application_.session.mode() != SimulationMode::StartupEdit)) {
                return reject("bodies can only be cleared in Edit Mode");
            }
            const Dimension dimension = application_.session.state().parameters.dimension;
            application_.session.state().world = WorldState(dimension);
            application_.presentation.selection = {};
            application_.presentation.focus = {};
            application_.presentation.inspector = {};
            return {CommandStatus::Applied, "bodies cleared", {}};
        } else if constexpr (std::is_same_v<CommandType, SwitchSolver>) {
            if (application_.mode != AppMode::Simulation
                || application_.session.mode() == SimulationMode::Running) {
                return reject("solver changes require a non-running simulation");
            }
            const EngineSwitchResult result = application_.session.physics().switchEngine(
                value.configuration, application_.session.state().world);
            if (!result.switched()) return reject(result.message);
            application_.session.state().parameters.solver = value.configuration;
            return {CommandStatus::Applied, "solver switched", {}};
        } else if constexpr (std::is_same_v<CommandType, SetTimeDisplayUnit>) {
            application_.presentation.units.time = value.unit;
            return {CommandStatus::Applied, "time display unit updated", {}};
        } else if constexpr (std::is_same_v<CommandType, SetAutomaticDistanceUnits>) {
            application_.presentation.units.automatic_distance = value.enabled;
            return {CommandStatus::Applied, "automatic distance units updated", {}};
        } else if constexpr (std::is_same_v<CommandType, SetDistanceDisplayUnit>) {
            application_.presentation.units.distance = value.unit;
            application_.presentation.units.automatic_distance = false;
            return {CommandStatus::Applied, "distance display unit updated", {}};
        } else if constexpr (std::is_same_v<CommandType, SetGridVisible>) {
            application_.presentation.grid.visible = value.visible;
            return {CommandStatus::Applied, "grid visibility updated", {}};
        }
    }, command);
}

bool CommandDispatcher::containsBody(BodyId body) const {
    const WorldState& world = application_.session.state().world;
    for (std::size_t index = 0; index < world.bodyCount(); ++index) {
        if (world.body(index).id == body) return true;
    }
    return false;
}

CommandResult CommandDispatcher::reject(std::string_view message) const {
    return {CommandStatus::Rejected, std::string(message), {}};
}

}
