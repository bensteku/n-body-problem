#include "frontend/vulkan_frontend.hpp"

#include "app/app_state.hpp"
#include "app/commands.hpp"
#include "rendering/render_scene.hpp"
#include "rendering/render_camera.hpp"
#include "rendering/camera_policy.hpp"
#include "rendering/render_diagnostics.hpp"
#include "rendering/render_graph.hpp"
#include "rendering/render_graph_executor.hpp"
#include "rendering/render_picking.hpp"
#include "rendering/render_style.hpp"
#include "rendering/material_registry.hpp"
#include "rendering/upload_arena.hpp"
#include "rendering/vulkan_renderer.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <span>
#include <vector>

namespace nbody::frontend {

namespace {

std::string compactNumber(double value, int significant_digits = 6) {
    char format[16]{};
    std::snprintf(format, sizeof(format), "%%.%dg", significant_digits);
    char text[64]{};
    std::snprintf(text, sizeof(text), format, value);
    return text;
}

void fixedTextField(const char* id, float width, const std::string& text) {
    const float height = ImGui::GetTextLineHeightWithSpacing();
    ImGui::BeginChild(id, {width, height}, false,
                      ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar
                      | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::TextUnformatted(text.c_str());
    ImGui::EndChild();
}

} // namespace

namespace {

class VulkanFrontendState {
public:
    VulkanFrontendState() : commands_(application_) {}

    void draw(GLFWwindow* window, rendering::VulkanRenderer& renderer) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - last_frame_).count();
        last_frame_ = now;
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        if (application_.mode == nbody::app::AppMode::MainMenu) {
            drawTitle();
        } else if (application_.mode == nbody::app::AppMode::Simulation) {
            drawSimulation(window, std::min(elapsed, 0.1));
            drawRenderDiagnostics(renderer);
        }
        ImGui::Render();
        renderer.render(window, ImGui::GetDrawData(), buildRenderScene());
    }

    void onScroll(double offset) { pending_scroll_ += offset; }

    static void scrollCallback(GLFWwindow* window, double x_offset, double y_offset) {
        ImGui_ImplGlfw_ScrollCallback(window, x_offset, y_offset);
        if (auto* state = static_cast<VulkanFrontendState*>(glfwGetWindowUserPointer(window))) {
            state->onScroll(y_offset);
        }
    }

private:
    nbody::app::ApplicationState application_;
    nbody::app::CommandDispatcher commands_;
    std::optional<FramePublication> publication_;
    ::nbody::rendering::TrajectoryHistory trajectory_history_;
    std::chrono::steady_clock::time_point last_frame_{std::chrono::steady_clock::now()};
    double accumulator_{};
    double time_scale_value_{1.0};
    double simulation_timestep_{0.001};
    double simulation_steps_per_second_{};
    int time_scale_unit_{};
    int generator_{};
    int generated_count_{8};
    int spiral_arms_{2};
    int random_seed_{42};
    float body_x_{};
    float body_y_{};
    float body_velocity_x_{};
    float body_velocity_y_{};
    float body_mass_{1.0f};
    float body_radius_{0.25f};
    float minimum_mass_{0.5f};
    float maximum_mass_{2.0f};
    float minimum_radius_{0.1f};
    float maximum_radius_{0.5f};
    float distribution_size_{5.0f};
    float gaussian_sigma_{2.0f};
    float inner_radius_{2.0f};
    float outer_radius_{8.0f};
    float spiral_turns_{2.0f};
    float tangential_velocity_{1.0f};
    bool add_body_{};
    bool static_body_{};
    bool scaled_solar_system_{};
    float camera_x_{};
    float camera_y_{};
    float view_scale_{20.0f};
    double pending_scroll_{};
    double last_cursor_x_{};
    double last_cursor_y_{};
    bool have_cursor_position_{};
    bool left_mouse_down_{};
    bool camera_dragged_{};
    ImVec2 viewport_min_{};
    ImVec2 viewport_max_{};
    bool reset_confirmation_{};
    std::optional<Dimension> dimension_confirmation_;
    bool show_render_diagnostics_{};

    WorldState& world() { return application_.session.state().world; }
    const WorldState& world() const { return application_.session.state().world; }
    SimulationParameters& parameters() { return application_.session.state().parameters; }
    const SimulationParameters& parameters() const { return application_.session.state().parameters; }

    bool simulationRunning() const {
        return application_.session.mode() == nbody::app::SimulationMode::Running;
    }

    ::nbody::rendering::RenderScene buildRenderScene() const {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        const float render_width = std::max(1.0f, viewport->WorkSize.x);
        const float render_height = std::max(1.0f, viewport->WorkSize.y);
        ::nbody::rendering::RenderCamera camera;
        camera.position = {camera_x_, camera_y_, 0.0};
        camera.zoom = view_scale_;
        ::nbody::rendering::RenderSceneSettings settings;
        settings.grid.visible = application_.presentation.grid.visible;
        settings.grid.spacing = static_cast<double>(gridStep(std::min(render_width, render_height)));
        settings.show_trajectories = application_.presentation.trajectories.visible;
        settings.show_selection_outline = true;
        return ::nbody::rendering::makeRenderScene(
            publication_.value_or(FramePublication{}), camera, settings, {},
            render_width, render_height,
            application_.presentation.selection.primary,
            application_.presentation.focus.body,
            trajectory_history_.view());
    }

    void drawTitle() {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->GetCenter(), ImGuiCond_Always, {0.5f, 0.5f});
        ImGui::SetNextWindowSize({440.0f, 240.0f}, ImGuiCond_Always);
        ImGui::Begin("N-Body Simulator", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove);
        ImGui::Dummy({1.0f, 20.0f});
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - ImGui::CalcTextSize("N-BODY SIMULATOR").x) * 0.5f);
        ImGui::TextUnformatted("N-BODY SIMULATOR");
        ImGui::Dummy({1.0f, 60.0f});
        if (ImGui::Button("Enter simulation", {-1.0f, 36.0f}) || ImGui::IsKeyPressed(ImGuiKey_Enter)) {
            commands_.dispatch(nbody::app::StartSimulation{});
        }
        ImGui::TextDisabled("Press Enter to begin");
        ImGui::End();
    }

    void drawRenderDiagnostics(const rendering::VulkanRenderer& renderer) {
        if (!show_render_diagnostics_) return;
        const auto& diagnostics = renderer.diagnostics();
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos({viewport->WorkPos.x + 16.0f, viewport->WorkPos.y + 56.0f},
                                ImGuiCond_Always);
        ImGui::SetNextWindowSize({300.0f, 260.0f}, ImGuiCond_Always);
        ImGui::Begin("Renderer diagnostics", &show_render_diagnostics_,
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoResize);
        ImGui::Text("frames %llu  |  %.1f FPS",
                    static_cast<unsigned long long>(diagnostics.frame_count),
                    diagnostics.frames_per_second);
        ImGui::Text("frame %.3f ms  |  prepare %.3f ms",
                    diagnostics.frame_milliseconds, diagnostics.cpu_prepare_milliseconds);
        ImGui::Text("record %.3f ms  |  bodies %zu",
                    diagnostics.command_record_milliseconds, diagnostics.submitted_bodies);
        ImGui::Text("upload %zu bytes  |  passes %zu",
                    diagnostics.uploaded_bytes, diagnostics.compiled_passes);
        ImGui::Text("barriers %zu  |  external GPU %s",
                    diagnostics.resource_barriers,
                    diagnostics.used_external_gpu_frame ? "yes" : "no");
        ImGui::SeparatorText("CPU command encoding by pass");
        constexpr const char* pass_names[] = {
            "depth", "opaque", "transparent", "grid/debug",
            "trajectories", "selection", "UI"};
        for (std::size_t index = 0; index < diagnostics.pass_cpu_milliseconds.size(); ++index) {
            ImGui::Text("%-12s %.3f ms", pass_names[index],
                        diagnostics.pass_cpu_milliseconds[index]);
        }
        ImGui::Separator();
        ImGui::Text("validation: %s  messages %llu  warnings %llu  errors %llu",
                    diagnostics.validation_enabled ? "enabled" : "off",
                    static_cast<unsigned long long>(diagnostics.validation_messages),
                    static_cast<unsigned long long>(diagnostics.validation_warnings),
                    static_cast<unsigned long long>(diagnostics.validation_errors));
        ImGui::End();
    }

    void drawSimulation(GLFWwindow* window, double elapsed) {
        updatePanelAnimation(elapsed);
        updateViewportBounds();
        drawTopBar();
        drawTopBarIndicator();
        drawSidePanels();
        drawBottomBar();
        drawBottomBarIndicator();
        updateCameraInput(window);

        simulation_timestep_ = std::clamp(simulation_timestep_, 1.0e-5, 86400.0);
        parameters().timestep = simulation_timestep_;
        parameters().gravitational_constant = scaled_solar_system_ ? 9.33076e-11 : 0.1;
        // Solar-system distances use a physical kilometre scale. Keep the
        // regularization far below satellite orbital distances; it is not a
        // substitute for collision handling or timestep selection.
        constexpr double solar_system_softening_km = 100.0;
        parameters().softening_length = scaled_solar_system_
            ? solar_system_softening_km / worldKilometersPerUnit()
            : 0.05;
        parameters().collision.model = CollisionModel::Transparent;
        const double requested_rate = timeScaleSecondsPerRealSecond();
        if (simulationRunning()) {
            accumulator_ += elapsed * requested_rate;
        } else {
            accumulator_ = 0.0;
        }
        std::size_t steps_this_frame = 0;
        while (simulationRunning() && accumulator_ >= parameters().timestep && steps_this_frame < 256) {
            application_.session.step();
            accumulator_ -= parameters().timestep;
            ++steps_this_frame;
        }
        trajectory_history_.setDurationSeconds(std::max(
            application_.presentation.trajectories.duration_seconds,
            simulation_timestep_ * 512.0));
        trajectory_history_.setSamplingIntervalSeconds(std::max(
            0.05, simulation_timestep_));
        publication_ = application_.session.publishFrame();
        trajectory_history_.observe(*publication_);
        updateCameraLock();
        if (elapsed > 0.000001) {
            const double measured = static_cast<double>(steps_this_frame) / elapsed;
            simulation_steps_per_second_ = simulation_steps_per_second_ == 0.0
                ? measured
                : simulation_steps_per_second_ * 0.9 + measured * 0.1;
        }
        if (add_body_) drawAddBody();
        drawWorld(window);
    }

    void updatePanelAnimation(double elapsed) {
        for (auto& panel : application_.presentation.panels.panels) {
            const float target = panel.expanded ? 1.0f : 0.0f;
            const float step = static_cast<float>(std::min(elapsed, 0.5) / 0.5);
            if (panel.animation_progress < target) {
                panel.animation_progress = std::min(target, panel.animation_progress + step);
            } else {
                panel.animation_progress = std::max(target, panel.animation_progress - step);
            }
        }
    }

    nbody::app::FoldablePanelState& panel(nbody::app::PanelEdge edge) {
        for (auto& candidate : application_.presentation.panels.panels) {
            if (candidate.edge == edge) return candidate;
        }
        return application_.presentation.panels.panels[0];
    }

    void drawTopBar() {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        constexpr float height = 42.0f;
        const float progress = panel(nbody::app::PanelEdge::Top).animation_progress;
        ImGui::SetNextWindowPos({viewport->WorkPos.x, viewport->WorkPos.y - height * (1.0f - progress)},
                                ImGuiCond_Always);
        ImGui::SetNextWindowSize({viewport->WorkSize.x, height}, ImGuiCond_Always);
        ImGui::Begin("Top bar", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        const auto mode = application_.session.mode();
        if (mode == nbody::app::SimulationMode::StartupEdit) {
            if (ImGui::Button("Start simulation")) {
                commands_.dispatch(nbody::app::FinishStartupEdit{});
                commands_.dispatch(nbody::app::ResumeSimulation{});
            }
            ImGui::SameLine();
            const int current_dimension = parameters().dimension == Dimension::Three ? 1 : 0;
            int selected_dimension = current_dimension;
            const char* dimensions[] = {"2D", "3D"};
            ImGui::SetNextItemWidth(70.0f);
            if (ImGui::Combo("Dimension", &selected_dimension, dimensions, 2)
                && selected_dimension != current_dimension) {
                dimension_confirmation_ = selected_dimension == 1 ? Dimension::Three : Dimension::Two;
                reset_confirmation_ = true;
            }
        } else if (ImGui::Button(simulationRunning() ? "Pause" : "Start")) {
            if (simulationRunning()) commands_.dispatch(nbody::app::PauseSimulation{});
            else commands_.dispatch(nbody::app::ResumeSimulation{});
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100.0f);
        ImGui::InputDouble("##time-scale", &time_scale_value_, 0.1, 1.0, "%.3f");
        ImGui::SameLine();
        const char* rate_units[] = {"seconds / real second", "days / real second",
                                    "months / real second", "years / real second"};
        ImGui::SetNextItemWidth(155.0f);
        ImGui::Combo("##time-scale-unit", &time_scale_unit_, rate_units, 4);
        ImGui::SameLine();
        fixedTextField("##sim-time-value", 132.0f,
                       "sim time " + compactNumber(displayedSimulationTime(), 6));
        ImGui::SameLine();
        int display_unit = static_cast<int>(application_.presentation.units.time);
        const char* display_units[] = {"seconds", "days", "months", "years"};
        ImGui::SetNextItemWidth(85.0f);
        if (ImGui::Combo("##sim-time-unit", &display_unit, display_units, 4)) {
            commands_.dispatch(nbody::app::SetTimeDisplayUnit{
                static_cast<nbody::app::TimeDisplayUnit>(display_unit)});
        }
        ImGui::SameLine();
        fixedTextField("##body-count", 92.0f,
                       "| bodies " + std::to_string(world().bodyCount()));
        ImGui::SameLine();
        if (ImGui::Button(application_.presentation.grid.visible ? "Hide grid" : "Show grid")) {
            commands_.dispatch(nbody::app::SetGridVisible{!application_.presentation.grid.visible});
        }
        const bool edit_mode = mode == nbody::app::SimulationMode::StartupEdit
            || mode == nbody::app::SimulationMode::Edit;
        if (edit_mode) {
            ImGui::SameLine();
            if (ImGui::Button("Add body")) add_body_ = true;
            ImGui::SameLine();
            if (ImGui::Button("Solar system")) createDebugSolarSystem();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            dimension_confirmation_.reset();
            reset_confirmation_ = true;
        }
        ImGui::SameLine(ImGui::GetWindowWidth() - 28.0f);
        if (ImGui::SmallButton("^")) panel(nbody::app::PanelEdge::Top).expanded = false;
        ImGui::End();

        if (reset_confirmation_) {
            ImGui::OpenPopup("Confirm reset");
            reset_confirmation_ = false;
        }
        if (ImGui::BeginPopupModal("Confirm reset", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextUnformatted(dimension_confirmation_
                ? "Switching dimension will discard the entire current setup."
                : "This will discard the current setup and return to Initial Edit Mode.");
            if (ImGui::Button("Reset", {100.0f, 0.0f})) {
                if (dimension_confirmation_) {
                    commands_.dispatch(nbody::app::SwitchDimension{*dimension_confirmation_});
                    dimension_confirmation_.reset();
                } else {
                    commands_.dispatch(nbody::app::ResetToInitialEdit{});
                }
                resetTransientFrontendState();
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", {100.0f, 0.0f})) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
    }

    void resetTransientFrontendState() {
        accumulator_ = 0.0;
        scaled_solar_system_ = false;
        simulation_timestep_ = 0.001;
        time_scale_value_ = 1.0;
        time_scale_unit_ = 0;
        camera_x_ = 0.0f;
        camera_y_ = 0.0f;
        view_scale_ = 20.0f;
        add_body_ = false;
        pending_scroll_ = 0.0;
        publication_.reset();
    }

    void drawSidePanels() {
        drawPanelIndicator(nbody::app::PanelEdge::Left);
        drawPanelIndicator(nbody::app::PanelEdge::Right);
        drawLeftPanel();
        drawRightPanel();
    }

    void drawBottomBar() {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        constexpr float height = 48.0f;
        const float progress = panel(nbody::app::PanelEdge::Bottom).animation_progress;
        const float render_span = std::min(viewport->WorkSize.x, viewport->WorkSize.y);
        const int active_backend = static_cast<int>(parameters().solver.backend);
        ImGui::SetNextWindowPos({viewport->WorkPos.x,
                                 viewport->WorkPos.y + viewport->WorkSize.y - height * progress},
                                ImGuiCond_Always);
        ImGui::SetNextWindowSize({viewport->WorkSize.x, height}, ImGuiCond_Always);
        ImGui::Begin("Bottom bar", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        ImGui::SetNextItemWidth(115.0f);
        ImGui::InputDouble("##timestep", &simulation_timestep_, 0.001, 0.01, "%.6g");
        ImGui::SameLine();
        fixedTextField("##simulation-rate", 218.0f,
                       "step s | sim " + compactNumber(simulation_steps_per_second_, 5)
                       + " fps | " + (simulationRunning() ? "running" : "paused"));
        ImGui::SameLine();
        fixedTextField("##backend-label", 62.0f, "| backend");
        ImGui::SameLine();
        int selected_backend = active_backend;
        const char* backend_names[] = {"Scalar", "SIMD"};
        ImGui::SetNextItemWidth(85.0f);
        ImGui::BeginDisabled(simulationRunning());
        if (ImGui::Combo("##backend", &selected_backend, backend_names, 2)
            && selected_backend != active_backend) {
            SolverConfiguration configuration = parameters().solver;
            configuration.backend = static_cast<ComputeBackend>(selected_backend);
            commands_.dispatch(nbody::app::SwitchSolver{configuration});
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        fixedTextField("##gpu-status", 112.0f, "GPU unavailable");
        ImGui::SameLine();
        if (ImGui::SmallButton(show_render_diagnostics_ ? "Hide render stats" : "Render stats")) {
            show_render_diagnostics_ = !show_render_diagnostics_;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton(application_.presentation.trajectories.visible
                ? "Hide orbits" : "Show orbits")) {
            application_.presentation.trajectories.visible =
                !application_.presentation.trajectories.visible;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(92.0f);
        ImGui::InputDouble("##trajectory-duration",
                           &application_.presentation.trajectories.duration_seconds,
                           1.0, 10.0, "%.3g s");
        ImGui::SameLine();
        const GridDistance distance = gridDistance(render_span);
        fixedTextField("##camera-status", 190.0f,
                       "| zoom " + compactNumber(view_scale_, 5)
                       + " | grid " + compactNumber(distance.value, 5)
                       + " " + distance.unit);
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset view")) {
            camera_x_ = 0.0f;
            camera_y_ = 0.0f;
            view_scale_ = 20.0f;
        }
        ImGui::SameLine(ImGui::GetWindowWidth() - 28.0f);
        if (ImGui::SmallButton("v")) panel(nbody::app::PanelEdge::Bottom).expanded = false;
        ImGui::End();
    }

    void drawTopBarIndicator() {
        auto& state = panel(nbody::app::PanelEdge::Top);
        if (state.expanded) return;
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos({viewport->WorkPos.x + viewport->WorkSize.x * 0.5f - 18.0f,
                                 viewport->WorkPos.y + 3.0f}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({36.0f, 24.0f}, ImGuiCond_Always);
        ImGui::Begin("Top bar indicator", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        if (ImGui::IsWindowHovered()) state.hover_time += ImGui::GetIO().DeltaTime;
        else state.hover_time = 0.0f;
        if (state.hover_time >= application_.presentation.panels.reveal_dwell_seconds
            || ImGui::Button("v")) {
            state.expanded = true;
            state.hover_time = 0.0f;
        }
        ImGui::End();
    }

    void drawBottomBarIndicator() {
        auto& state = panel(nbody::app::PanelEdge::Bottom);
        if (state.expanded) return;
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos({viewport->WorkPos.x + viewport->WorkSize.x * 0.5f - 18.0f,
                                 viewport->WorkPos.y + viewport->WorkSize.y - 27.0f}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({36.0f, 24.0f}, ImGuiCond_Always);
        ImGui::Begin("Bottom bar indicator", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        if (ImGui::IsWindowHovered()) state.hover_time += ImGui::GetIO().DeltaTime;
        else state.hover_time = 0.0f;
        if (state.hover_time >= application_.presentation.panels.reveal_dwell_seconds
            || ImGui::Button("^")) {
            state.expanded = true;
            state.hover_time = 0.0f;
        }
        ImGui::End();
    }

    void drawPanelIndicator(nbody::app::PanelEdge edge) {
        auto& state = panel(edge);
        if (state.expanded) return;
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        const bool left = edge == nbody::app::PanelEdge::Left;
        const ImVec2 position = left
            ? ImVec2{viewport->WorkPos.x + 4.0f, viewport->WorkPos.y + viewport->WorkSize.y * 0.5f - 20.0f}
            : ImVec2{viewport->WorkPos.x + viewport->WorkSize.x - 34.0f,
                     viewport->WorkPos.y + viewport->WorkSize.y * 0.5f - 20.0f};
        ImGui::SetNextWindowPos(position, ImGuiCond_Always);
        ImGui::SetNextWindowSize({30.0f, 40.0f}, ImGuiCond_Always);
        ImGui::Begin(left ? "Left panel indicator" : "Right panel indicator", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        if (ImGui::IsWindowHovered()) state.hover_time += ImGui::GetIO().DeltaTime;
        else state.hover_time = 0.0f;
        if (state.hover_time >= application_.presentation.panels.reveal_dwell_seconds
            || ImGui::Button(left ? ">" : "<")) {
            state.expanded = true;
            state.hover_time = 0.0f;
        }
        ImGui::End();
    }

    void drawLeftPanel() {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        constexpr float width = 280.0f;
        const float progress = panel(nbody::app::PanelEdge::Left).animation_progress;
        const float top = 42.0f * panel(nbody::app::PanelEdge::Top).animation_progress;
        const float bottom = 48.0f * panel(nbody::app::PanelEdge::Bottom).animation_progress;
        ImGui::SetNextWindowPos({viewport->WorkPos.x - width * (1.0f - progress),
                                 viewport->WorkPos.y + top}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({width, std::max(1.0f, viewport->WorkSize.y - top - bottom)}, ImGuiCond_Always);
        ImGui::Begin("Bodies", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
            | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings);
        if (ImGui::Button("Fold")) panel(nbody::app::PanelEdge::Left).expanded = false;
        ImGui::SameLine();
        if (application_.presentation.focus.body) {
            fixedTextField("##focused-body", 178.0f,
                           "Locked to #" + std::to_string(application_.presentation.focus.body->value));
        }
        else fixedTextField("##focused-body", 178.0f, "Select a body to inspect");
        ImGui::Separator();
        if (publication_ && publication_->published()) {
            for (const RenderBody& body : publication_->cpu_snapshot->bodies) {
                ImGui::PushID(static_cast<int>(body.id.value));
                const bool selected = application_.presentation.selection.primary == body.id;
                if (ImGui::Selectable("##body", selected, 0, {245.0f, 24.0f})) {
                    commands_.dispatch(nbody::app::SelectBody{body.id});
                }
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    commands_.dispatch(nbody::app::FocusBody{body.id});
                }
                ImGui::SameLine();
                fixedTextField("##body-summary", 86.0f,
                               "#" + std::to_string(body.id.value) + "  "
                               + compactNumber(body.mass, 4));
                ImGui::PopID();
            }
        }
        ImGui::End();
    }

    void drawRightPanel() {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        constexpr float width = 300.0f;
        const float progress = panel(nbody::app::PanelEdge::Right).animation_progress;
        const float top = 42.0f * panel(nbody::app::PanelEdge::Top).animation_progress;
        const float bottom = 48.0f * panel(nbody::app::PanelEdge::Bottom).animation_progress;
        ImGui::SetNextWindowPos({viewport->WorkPos.x + viewport->WorkSize.x - width * progress,
                                 viewport->WorkPos.y + top}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({width, std::max(1.0f, viewport->WorkSize.y - top - bottom)}, ImGuiCond_Always);
        ImGui::Begin("Inspector", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
            | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings);
        if (ImGui::Button("Fold")) panel(nbody::app::PanelEdge::Right).expanded = false;
        ImGui::Separator();
        const auto selected = application_.presentation.selection.primary;
        if (selected && world().bodyCount() > 0) {
            for (std::size_t index = 0; index < world().bodyCount(); ++index) {
                if (world().body(index).id == *selected) {
                    const ConstBodyView body = world().body(index);
                    fixedTextField("##inspector-body", 260.0f,
                                   "Body #" + std::to_string(body.id.value));
                    fixedTextField("##inspector-mass", 260.0f,
                                   "Mass " + compactNumber(body.mass));
                    fixedTextField("##inspector-radius", 260.0f,
                                   "Radius " + compactNumber(body.radius));
                    fixedTextField("##inspector-position", 260.0f,
                                   "Position " + compactNumber(body.position.x, 5) + ", "
                                   + compactNumber(body.position.y, 5) + ", "
                                   + compactNumber(body.position.z, 5));
                    break;
                }
            }
        } else {
            ImGui::TextDisabled("No body selected");
        }
        ImGui::End();
    }

    void updateViewportBounds() {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        const float left = panel(nbody::app::PanelEdge::Left).animation_progress * 280.0f;
        const float right = panel(nbody::app::PanelEdge::Right).animation_progress * 300.0f;
        const float top = 42.0f * panel(nbody::app::PanelEdge::Top).animation_progress;
        const float bottom = 48.0f * panel(nbody::app::PanelEdge::Bottom).animation_progress;
        viewport_min_ = {viewport->WorkPos.x + left, viewport->WorkPos.y + top};
        viewport_max_ = {viewport->WorkPos.x + viewport->WorkSize.x - right,
                         viewport->WorkPos.y + viewport->WorkSize.y - bottom};
    }

    void updateCameraLock() {
        const auto focused_body = application_.presentation.focus.body;
        if (!focused_body || !publication_ || !publication_->published()) return;
        for (const RenderBody& body : publication_->cpu_snapshot->bodies) {
            if (body.id == *focused_body) {
                camera_x_ = static_cast<float>(body.position.x);
                camera_y_ = static_cast<float>(body.position.y);
                return;
            }
        }
        commands_.dispatch(nbody::app::FocusBody{std::nullopt});
    }

    double timeScaleSecondsPerRealSecond() const {
        const double value = std::max(0.0, time_scale_value_);
        return value * secondsPerDisplayTimeUnit(time_scale_unit_);
    }

    static double secondsPerDisplayTimeUnit(int unit) {
        switch (unit) {
        case 1: return 86400.0;
        case 2: return 30.0 * 86400.0;
        case 3: return 365.25 * 86400.0;
        default: return 1.0;
        }
    }

    double displayedSimulationTime() const {
        return world().time() / secondsPerDisplayTimeUnit(
            static_cast<int>(application_.presentation.units.time));
    }

    void drawAddBody() {
        ImGui::OpenPopup("Add body");
        if (!ImGui::BeginPopupModal("Add body", &add_body_, ImGuiWindowFlags_AlwaysAutoResize)) return;
        const char* generators[] = {"Single body", "Uniform", "Gaussian", "Spiral", "Circle"};
        ImGui::Combo("Pattern", &generator_, generators, 5);
        if (generator_ == 0) {
            ImGui::InputFloat("X", &body_x_);
            ImGui::InputFloat("Y", &body_y_);
            ImGui::InputFloat("Velocity X", &body_velocity_x_);
            ImGui::InputFloat("Velocity Y", &body_velocity_y_);
            ImGui::InputFloat("Mass", &body_mass_);
            ImGui::InputFloat("Radius", &body_radius_);
        } else {
            ImGui::InputInt("Count", &generated_count_);
            ImGui::InputFloat("Center X", &body_x_);
            ImGui::InputFloat("Center Y", &body_y_);
            ImGui::InputFloat("Min mass", &minimum_mass_);
            ImGui::InputFloat("Max mass", &maximum_mass_);
            ImGui::InputFloat("Min radius", &minimum_radius_);
            ImGui::InputFloat("Max radius", &maximum_radius_);
            ImGui::InputFloat("Velocity X", &body_velocity_x_);
            ImGui::InputFloat("Velocity Y", &body_velocity_y_);
            if (generator_ == 1) ImGui::InputFloat("Uniform half-size", &distribution_size_);
            if (generator_ == 2) ImGui::InputFloat("Gaussian sigma", &gaussian_sigma_);
            if (generator_ == 3) {
                ImGui::InputInt("Arms", &spiral_arms_);
                ImGui::InputFloat("Inner radius", &inner_radius_);
                ImGui::InputFloat("Outer radius", &outer_radius_);
                ImGui::InputFloat("Turns", &spiral_turns_);
                ImGui::InputFloat("Tangential velocity", &tangential_velocity_);
            }
            if (generator_ == 4) {
                ImGui::InputFloat("Inner radius", &inner_radius_);
                ImGui::InputFloat("Outer radius", &outer_radius_);
                ImGui::InputFloat("Tangential velocity", &tangential_velocity_);
            }
            ImGui::InputInt("Random seed", &random_seed_);
        }
        ImGui::Checkbox("Static body/bodies", &static_body_);
        if (ImGui::Button("Create")) {
            generateBodies();
            accumulator_ = 0.0;
            add_body_ = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            add_body_ = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    void generateBodies() {
        const int count = generator_ == 0 ? 1 : std::max(1, generated_count_);
        std::mt19937 generator(static_cast<std::mt19937::result_type>(random_seed_));
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        std::uniform_real_distribution<float> mass(std::min(minimum_mass_, maximum_mass_),
                                                   std::max(minimum_mass_, maximum_mass_));
        std::uniform_real_distribution<float> radius(std::min(minimum_radius_, maximum_radius_),
                                                     std::max(minimum_radius_, maximum_radius_));
        constexpr float pi = 3.14159265358979323846f;
        for (int index = 0; index < count; ++index) {
            float x = body_x_;
            float y = body_y_;
            float angle = 0.0f;
            if (generator_ == 1) {
                x += (unit(generator) * 2.0f - 1.0f) * distribution_size_;
                y += (unit(generator) * 2.0f - 1.0f) * distribution_size_;
            } else if (generator_ == 2) {
                std::normal_distribution<float> gaussian(0.0f, std::max(0.001f, gaussian_sigma_));
                x += gaussian(generator);
                y += gaussian(generator);
            } else if (generator_ == 3 || generator_ == 4) {
                const float fraction = count == 1 ? 0.0f : static_cast<float>(index) / (count - 1);
                angle = generator_ == 3
                    ? 2.0f * pi * static_cast<float>(index % std::max(1, spiral_arms_))
                        / std::max(1, spiral_arms_) + 2.0f * pi * spiral_turns_ * fraction
                    : 2.0f * pi * fraction;
                const float radial = inner_radius_ + (outer_radius_ - inner_radius_) * fraction;
                x += radial * std::cos(angle);
                y += radial * std::sin(angle);
            }
            BodyState body;
            body.position = {x, y, 0.0};
            body.velocity = {body_velocity_x_, body_velocity_y_, 0.0};
            if (generator_ == 3 || generator_ == 4) {
                // Build the tangential component from the actual radial
                // offset so it remains perpendicular to the spiral centre,
                // independent of how the point was generated.
                const float radial_x = x - body_x_;
                const float radial_y = y - body_y_;
                const float radial_length = std::hypot(radial_x, radial_y);
                if (radial_length > 0.0f) {
                    body.velocity.x += -radial_y / radial_length * tangential_velocity_;
                    body.velocity.y += radial_x / radial_length * tangential_velocity_;
                }
            }
            body.mass = generator_ == 0 ? std::max(0.001f, body_mass_) : std::max(0.001f, mass(generator));
            body.radius = generator_ == 0 ? std::max(0.001f, body_radius_) : std::max(0.001f, radius(generator));
            body.is_static = static_body_;
            commands_.dispatch(nbody::app::CreateBody{body});
        }
    }

    void updateCameraInput(GLFWwindow* window) {
        const ImVec2 work_min = viewport_min_;
        const ImVec2 work_max = viewport_max_;
        const ImGuiViewport* full_viewport = ImGui::GetMainViewport();
        const float viewport_width = std::max(1.0f, full_viewport->WorkSize.x);
        const float viewport_height = std::max(1.0f, full_viewport->WorkSize.y);
        const float viewport_span = std::min(viewport_width, viewport_height);
        const float maximum_view_scale = maximumViewScale(viewport_span);
        double cursor_x = 0.0;
        double cursor_y = 0.0;
        glfwGetCursorPos(window, &cursor_x, &cursor_y);
        const bool over_world = cursor_x >= work_min.x && cursor_x <= work_max.x
            && cursor_y >= work_min.y + 28.0 && cursor_y <= work_max.y;
        const bool ui_using_mouse = ImGui::IsAnyItemActive() || add_body_;
        const bool left_mouse_down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        if (left_mouse_down && !left_mouse_down_) camera_dragged_ = false;
        if (over_world && !ui_using_mouse && pending_scroll_ != 0.0) {
            const float old_scale = view_scale_;
            view_scale_ = std::clamp(view_scale_ * std::pow(1.15f, static_cast<float>(pending_scroll_)),
                                     0.25f, maximum_view_scale);
            const ImVec2 mouse{static_cast<float>(cursor_x), static_cast<float>(cursor_y)};
            const float center_x = full_viewport->WorkPos.x + viewport_width * 0.5f;
            const float center_y = full_viewport->WorkPos.y + viewport_height * 0.5f;
            const float world_x = camera_x_ + (mouse.x - center_x) / old_scale;
            const float world_y = camera_y_ - (mouse.y - center_y) / old_scale;
            camera_x_ = world_x - (mouse.x - center_x) / view_scale_;
            camera_y_ = world_y + (mouse.y - center_y) / view_scale_;
        }
        pending_scroll_ = 0.0;
        if (over_world && !ui_using_mouse
            && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            if (!have_cursor_position_) {
                last_cursor_x_ = cursor_x;
                last_cursor_y_ = cursor_y;
                have_cursor_position_ = true;
                camera_dragged_ = false;
            }
            const double delta_x = cursor_x - last_cursor_x_;
            const double delta_y = cursor_y - last_cursor_y_;
            if (delta_x != 0.0 || delta_y != 0.0) {
                commands_.dispatch(nbody::app::FocusBody{std::nullopt});
                camera_dragged_ = true;
                camera_x_ -= static_cast<float>(delta_x) / view_scale_;
                camera_y_ += static_cast<float>(delta_y) / view_scale_;
            }
        }
        if (!left_mouse_down && left_mouse_down_ && over_world && !camera_dragged_ && !ui_using_mouse) {
            const auto scene = buildRenderScene();
            const auto full_viewport = ImGui::GetMainViewport();
            const ::nbody::rendering::PickingQuery query{
                parameters().dimension,
                static_cast<float>(cursor_x - full_viewport->WorkPos.x),
                static_cast<float>(cursor_y - full_viewport->WorkPos.y),
                {scene.viewport_width, scene.viewport_height},
                6.0f};
            const auto result = ::nbody::rendering::pickBody(scene, query);
            commands_.dispatch(nbody::app::SelectBody{
                result.hit ? std::optional<BodyId>{result.hit->body} : std::nullopt});
        }
        left_mouse_down_ = left_mouse_down;
        last_cursor_x_ = cursor_x;
        last_cursor_y_ = cursor_y;
        have_cursor_position_ = true;
    }

    double worldKilometersPerUnit() const {
        if (!scaled_solar_system_) return 1.0;
        constexpr double kilometersPerAu = 149597870.7;
        return 30.070 * kilometersPerAu / 400.0;
    }

    float maximumViewScale(float viewport_width) const {
        return ::nbody::rendering::zoomPolicy(
            scaled_solar_system_, worldKilometersPerUnit(), viewport_width).maximum;
    }

    float gridStep(float viewport_span) const {
        // The Moon reference zoom is defined as four cells across the
        // reference span. Keeping this proportional also makes the indicator
        // describe the actual rendered grid rather than a separate scale.
        return viewport_span / (4.0f * view_scale_);
    }

    struct GridDistance {
        double value;
        const char* unit;
    };

    GridDistance gridDistance(float viewport_width) const {
        constexpr double kilometersPerAu = 149597870.7;
        constexpr double kilometersPerLightYear = 9.460730472e12;
        const double kilometers = static_cast<double>(gridStep(viewport_width)) * worldKilometersPerUnit();
        if (kilometers < 0.01 * kilometersPerAu) return {kilometers, "km"};
        if (kilometers < kilometersPerLightYear) return {kilometers / kilometersPerAu, "AU"};
        return {kilometers / kilometersPerLightYear, "Ly"};
    }

    void createDebugSolarSystem() {
        struct PlanetDefinition {
            const char* name;
            double orbit_au;
            double mass_ratio;
            double radius_km;
        };
        constexpr std::array<PlanetDefinition, 8> planets{{
            {"Mercury", 0.387, 1.660e-7, 2439.7},
            {"Venus", 0.723, 2.447e-6, 6051.8},
            {"Earth", 1.000, 3.003e-6, 6371.0},
            {"Mars", 1.524, 3.227e-7, 3389.5},
            {"Jupiter", 5.203, 9.545e-4, 69911.0},
            {"Saturn", 9.537, 2.857e-4, 58232.0},
            {"Uranus", 19.191, 4.366e-5, 25362.0},
            {"Neptune", 30.070, 5.151e-5, 24622.0}
        }};
        constexpr double neptune_orbit = 400.0;
        constexpr double scale = neptune_orbit / 30.070;
        constexpr double kilometers_per_au = 149597870.7;
        constexpr double radius_scale = neptune_orbit / (30.070 * kilometers_per_au);
        // Calibrated so the Earth orbit is approximately one Julian year in
        // simulation seconds after Neptune is mapped to radius 400.
        constexpr double gravitational_constant = 9.33076e-11;
        constexpr double sun_mass = 1.0;
        constexpr double pi = 3.14159265358979323846;
        std::mt19937 generator(20260918);
        std::uniform_real_distribution<double> phase(0.0, 2.0 * pi);
        Vec3 earth_position{};
        Vec3 earth_velocity{};
        Vec3 jupiter_position{};
        Vec3 jupiter_velocity{};
        constexpr double earth_mass = 3.003e-6;
        constexpr double jupiter_mass = 9.545e-4;

        commands_.dispatch(nbody::app::ClearBodies{});
        BodyState sun;
        sun.mass = sun_mass;
        sun.radius = 696340.0 * radius_scale;
        sun.is_static = true;
        sun.kind = BodyKind::Star;
        commands_.dispatch(nbody::app::CreateBody{sun});

        for (const PlanetDefinition& planet : planets) {
            const double orbit = planet.orbit_au * scale;
            const double angle = phase(generator);
            const double speed = std::sqrt(gravitational_constant * sun_mass / orbit);
            BodyState body;
            body.position = {orbit * std::cos(angle), orbit * std::sin(angle), 0.0};
            body.velocity = {-speed * std::sin(angle), speed * std::cos(angle), 0.0};
            body.mass = planet.mass_ratio;
            body.radius = planet.radius_km * radius_scale;
            body.kind = BodyKind::Ordinary;
            commands_.dispatch(nbody::app::CreateBody{body});
            if (std::string_view(planet.name) == "Earth") {
                earth_position = body.position;
                earth_velocity = body.velocity;
            } else if (std::string_view(planet.name) == "Jupiter") {
                jupiter_position = body.position;
                jupiter_velocity = body.velocity;
            }
        }

        const auto addMoon = [&](const Vec3& parent_position, const Vec3& parent_velocity,
                                 double parent_mass, double distance_km, double radius_km,
                                 double mass_ratio, double orbit_phase) {
            const double distance_world = distance_km * radius_scale;
            const double orbital_speed = std::sqrt(gravitational_constant * parent_mass / distance_world);
            const Vec3 offset{distance_world * std::cos(orbit_phase),
                              distance_world * std::sin(orbit_phase), 0.0};
            const Vec3 tangent{-std::sin(orbit_phase) * orbital_speed,
                               std::cos(orbit_phase) * orbital_speed, 0.0};
            BodyState moon;
            moon.position = parent_position + offset;
            moon.velocity = parent_velocity + tangent;
            moon.mass = mass_ratio;
            moon.radius = radius_km * radius_scale;
            moon.kind = BodyKind::Ordinary;
            commands_.dispatch(nbody::app::CreateBody{moon});
        };

        // Values are physical present-day means, expressed in the demo's
        // solar-mass and kilometre-to-world-unit scales.
        addMoon(earth_position, earth_velocity, earth_mass,
                384400.0, 1737.4, 3.694e-8, phase(generator));
        addMoon(jupiter_position, jupiter_velocity, jupiter_mass,
                421700.0, 1821.6, 4.49e-8, phase(generator)); // Io
        addMoon(jupiter_position, jupiter_velocity, jupiter_mass,
                671034.0, 1560.8, 2.41e-8, phase(generator)); // Europa
        addMoon(jupiter_position, jupiter_velocity, jupiter_mass,
                1070412.0, 2634.1, 7.57e-8, phase(generator)); // Ganymede
        addMoon(jupiter_position, jupiter_velocity, jupiter_mass,
                1882709.0, 2410.3, 5.70e-8, phase(generator)); // Callisto
        accumulator_ = 0.0;
        scaled_solar_system_ = true;
        simulation_timestep_ = 3600.0;
        time_scale_value_ = 1.0;
        time_scale_unit_ = 1;
        camera_x_ = 0.0f;
        camera_y_ = 0.0f;
        view_scale_ = 1.5f;
    }

    void drawWorld(GLFWwindow* window) {
        // The bottom bar owns camera status and controls. Keep this hook for
        // the future world-space overlay layer.
        (void)window;
    }
};

} // namespace

} // namespace nbody::frontend

namespace nbody::frontend {

int VulkanFrontend::run() {
    if (!glfwInit()) throw std::runtime_error("GLFW initialization failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "N-Body Simulator", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("GLFW window creation failed");
    }
    try {
        rendering::VulkanRenderer renderer;
        renderer.initialize(window);
        VulkanFrontendState application;
        glfwSetWindowUserPointer(window, &application);
        glfwSetScrollCallback(window, VulkanFrontendState::scrollCallback);
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            application.draw(window, renderer);
            // Swapchain recreation reinitializes ImGui's GLFW backend, which
            // installs its own scroll callback. Restore the frontend callback
            // after each frame so zoom input remains owned by the frontend.
            glfwSetScrollCallback(window, VulkanFrontendState::scrollCallback);
        }
    } catch (...) {
        glfwDestroyWindow(window);
        glfwTerminate();
        throw;
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

}
