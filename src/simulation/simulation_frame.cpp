#include "simulation_frame.hpp"

#include <algorithm>

namespace nbody {

CpuFramePublisher::CpuFramePublisher(std::size_t slot_count) {
    slots_.reserve(std::max<std::size_t>(slot_count, 1));
    for (std::size_t index = 0; index < std::max<std::size_t>(slot_count, 1); ++index) {
        slots_.push_back(std::make_shared<SimulationFrame>());
    }
}

FramePublication CpuFramePublisher::publish(const WorldState& world,
                                            const FramePublicationRequest& request) {
    if (request.transport != FrameTransport::CpuSnapshot) {
        FramePublication publication;
        publication.status = FramePublicationStatus::UnsupportedTransport;
        publication.transport = request.transport;
        publication.message = "CPU frame publisher cannot provide a renderer buffer";
        return publication;
    }
    std::shared_ptr<SimulationFrame> writable;
    for (const auto& slot : slots_) {
        if (slot.use_count() == 1) {
            writable = slot;
            break;
        }
    }
    if (!writable) {
        writable = std::make_shared<SimulationFrame>();
        slots_.push_back(writable);
    }

    writable->dimension = world.dimension();
    writable->simulation_time = world.time();
    writable->body_count = world.bodyCount();
    writable->previous_body_count = previous_body_count_;
    writable->body_count_changed = previous_body_count_ != world.bodyCount();
    writable->diagnostics = world.diagnostics();
    writable->collision_transition = world.collisionTransitionDiagnostics();

    auto& bodies = writable->bodies;
    bodies.clear();
    bodies.reserve(world.bodyCount());
    for (std::size_t index = 0; index < world.bodyCount(); ++index) {
        const ConstBodyView body = world.body(index);
        bodies.push_back({body.id, body.position, body.velocity, body.mass, body.radius,
                          body.is_static(), body.kind});
    }

    writable->collision_events = world.collisionEvents();
    previous_body_count_ = world.bodyCount();
    FramePublication publication;
    publication.status = FramePublicationStatus::Published;
    publication.transport = request.transport;
    publication.sequence = next_sequence_++;
    publication.description = {
        publication.sequence,
        writable->dimension,
        writable->simulation_time,
        writable->body_count,
        writable->previous_body_count,
        writable->body_count_changed};
    publication.storage.kind = rendering::FrameStorageKind::CpuSnapshot;
    publication.readiness = {rendering::ReadinessKind::Immediate, 0};
    publication.cpu_snapshot = FrameLease(
        std::const_pointer_cast<const SimulationFrame>(writable));
    publication.message = "CPU snapshot published";
    return publication;
}

}
