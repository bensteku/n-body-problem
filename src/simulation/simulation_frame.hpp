#pragma once

#include "world_state.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace nbody {

// Renderer-facing data deliberately has no references into WorldState. A frame
// therefore remains valid while the simulation advances or compacts its body
// storage.
struct RenderBody {
    BodyId id;
    Vec3 position;
    Vec3 velocity;
    double mass{1.0};
    double radius{1.0};
    bool is_static{false};
    BodyKind kind{BodyKind::Ordinary};
};

struct SimulationFrame {
    Dimension dimension{Dimension::Two};
    double simulation_time{};
    std::size_t body_count{};
    std::size_t previous_body_count{};
    bool body_count_changed{false};
    WorldDiagnostics diagnostics;
    CollisionTransitionDiagnostics collision_transition;
    std::vector<RenderBody> bodies;
    std::vector<CollisionEvent> collision_events;
};

class FrameLease {
public:
    FrameLease() = default;

    const SimulationFrame* operator->() const { return frame_.get(); }
    const SimulationFrame& operator*() const { return *frame_; }
    explicit operator bool() const { return static_cast<bool>(frame_); }

private:
    friend class FramePublisher;
    explicit FrameLease(std::shared_ptr<const SimulationFrame> frame)
        : frame_(std::move(frame)) {}

    std::shared_ptr<const SimulationFrame> frame_;
};

// CPU publication uses a small pool so the renderer can retain a frame while
// simulation advances. If all slots are in flight, a temporary slot is added
// rather than overwriting data still owned by the renderer.
class FramePublisher {
public:
    explicit FramePublisher(std::size_t slot_count = 3);

    FrameLease publish(const WorldState& world);

private:
    std::vector<std::shared_ptr<SimulationFrame>> slots_;
    std::size_t previous_body_count_{};
};

}
