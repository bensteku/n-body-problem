#pragma once

#include "rendering/frame_contract.hpp"

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace nbody::rendering {

struct RenderResourceId {
    std::uint32_t value{};

    friend constexpr bool operator==(RenderResourceId, RenderResourceId) = default;
};

enum class RenderResourceKind {
    ColorTarget,
    DepthTarget,
    FrameStorage,
    MaterialStorage,
    UniformStorage,
    ExternalResource
};

struct RenderResourceDescription {
    RenderResourceId id;
    RenderResourceKind kind{RenderResourceKind::ExternalResource};
    bool transient{};
    bool imported{};
};

enum class RenderAccess {
    Read,
    Write,
    ReadWrite
};

struct RenderResourceUse {
    RenderResourceId resource;
    RenderAccess access{RenderAccess::Read};
};

enum class RenderPassKind {
    FrameImport,
    DepthPrepass,
    OpaqueBodies,
    TransparentBodies,
    GridAndDebug,
    Trajectories,
    SelectionOutlines,
    UserInterface
};

struct RenderPassDescription {
    std::string_view name;
    RenderPassKind kind{RenderPassKind::OpaqueBodies};
    std::vector<RenderResourceUse> resources;
    bool enabled{true};
};

// This graph describes intent and dependencies only. Vulkan and WebGPU
// backends compile it into their own command encoders, barriers, and passes.
class RenderGraph {
public:
    // Canonical 2D pass order. Backends may fuse or specialize these passes,
    // but the logical ordering remains shared with the future 3D renderer.
    static RenderGraph make2D(bool has_frame, bool show_trajectories,
                              bool show_debug, bool show_ui,
                              bool show_selection = true) {
        RenderGraph graph;
        const RenderResourceId color = graph.addResource({{}, RenderResourceKind::ColorTarget, false, true});
        const RenderResourceId frame = graph.addResource({{}, RenderResourceKind::FrameStorage, false, true});
        const RenderResourceId depth = graph.addResource({{}, RenderResourceKind::DepthTarget, true, false});
        if (has_frame) {
            graph.addPass({"frame import", RenderPassKind::FrameImport,
                           {{frame, RenderAccess::Read}}, true});
        }
        graph.addPass({"depth prepass", RenderPassKind::DepthPrepass,
                       {{depth, RenderAccess::Write}}, show_debug});
        if (has_frame) {
            graph.addPass({"opaque bodies", RenderPassKind::OpaqueBodies,
                           {{frame, RenderAccess::Read}, {color, RenderAccess::Write}}, true});
        }
        graph.addPass({"transparent bodies", RenderPassKind::TransparentBodies,
                       {{frame, RenderAccess::Read}, {color, RenderAccess::ReadWrite}}, true});
        graph.addPass({"grid and debug", RenderPassKind::GridAndDebug,
                       {{color, RenderAccess::ReadWrite}}, show_debug});
        graph.addPass({"trajectories", RenderPassKind::Trajectories,
                       {{color, RenderAccess::ReadWrite}}, show_trajectories});
        graph.addPass({"selection outlines", RenderPassKind::SelectionOutlines,
                       {{color, RenderAccess::ReadWrite}}, show_selection});
        graph.addPass({"user interface", RenderPassKind::UserInterface,
                       {{color, RenderAccess::ReadWrite}}, show_ui});
        return graph;
    }

    RenderResourceId addResource(RenderResourceDescription description) {
        description.id = {static_cast<std::uint32_t>(resources_.size())};
        resources_.push_back(description);
        return description.id;
    }

    std::size_t addPass(RenderPassDescription description) {
        passes_.push_back(std::move(description));
        return passes_.size() - 1;
    }

    const std::vector<RenderResourceDescription>& resources() const { return resources_; }
    const std::vector<RenderPassDescription>& passes() const { return passes_; }

    bool validate() const {
        if (passes_.empty()) return false;
        for (const RenderPassDescription& pass : passes_) {
            if (!pass.enabled || pass.resources.empty()) continue;
            for (const RenderResourceUse& use : pass.resources) {
                if (use.resource.value >= resources_.size()) return false;
            }
        }
        return true;
    }

private:
    std::vector<RenderResourceDescription> resources_;
    std::vector<RenderPassDescription> passes_;
};

}
