#pragma once

#include "world_state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
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
    friend class CpuFramePublisher;
    explicit FrameLease(std::shared_ptr<const SimulationFrame> frame)
        : frame_(std::move(frame)) {}

    std::shared_ptr<const SimulationFrame> frame_;
};

enum class FrameTransport {
    CpuSnapshot,
    RendererBuffer
};

struct FramePublicationRequest {
    FrameTransport transport{FrameTransport::CpuSnapshot};
};

enum class FramePublicationStatus {
    Published,
    UnsupportedTransport
};

// The publication object is the stable boundary consumed by renderers and
// frontends. CPU engines currently populate cpu_snapshot; a Vulkan publisher
// can later populate the same object with a renderer buffer without changing
// PhysicsSession or its callers.
struct FramePublication {
    FramePublicationStatus status{FramePublicationStatus::UnsupportedTransport};
    FrameTransport transport{FrameTransport::CpuSnapshot};
    std::uint64_t sequence{};
    FrameLease cpu_snapshot;
    std::string_view message{};

    bool published() const { return status == FramePublicationStatus::Published; }
};

class IFramePublisher {
public:
    virtual ~IFramePublisher() = default;
    virtual FramePublication publish(const WorldState& world,
                                     const FramePublicationRequest& request) = 0;
};

// CPU publication uses a small pool so the renderer can retain a frame while
// simulation advances. If all slots are in flight, a temporary slot is added
// rather than overwriting data still owned by the renderer.
class CpuFramePublisher final : public IFramePublisher {
public:
    explicit CpuFramePublisher(std::size_t slot_count = 3);

    FramePublication publish(const WorldState& world,
                             const FramePublicationRequest& request) override;

private:
    std::vector<std::shared_ptr<SimulationFrame>> slots_;
    std::size_t previous_body_count_{};
    std::uint64_t next_sequence_{1};
};

// Source compatibility for the low-level CPU publisher. New code should use
// IFramePublisher or CpuFramePublisher through FramePublication.
using FramePublisher = CpuFramePublisher;

}
