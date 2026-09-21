#pragma once

#include "rendering/render_graph.hpp"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace nbody::rendering {

struct CompiledRenderPass {
    std::size_t source_index{};
    RenderPassKind kind{RenderPassKind::OpaqueBodies};
};

struct CompiledResourceBarrier {
    RenderResourceId resource;
    RenderAccess previous{RenderAccess::Read};
    RenderAccess next{RenderAccess::Read};
    std::size_t before_pass{};
    std::size_t after_pass{};
};

// Backend-neutral compilation stage. Native backends consume this compact
// ordered plan to create barriers and command encoders without rebuilding
// dependency decisions themselves.
class RenderGraphExecutor {
public:
    bool compile(const RenderGraph& graph) {
        compiled_.clear();
        barriers_.clear();
        if (!graph.validate()) return false;
        compiled_.reserve(graph.passes().size());
        std::vector<std::optional<std::pair<RenderAccess, std::size_t>>> last_use(
            graph.resources().size());
        for (std::size_t index = 0; index < graph.passes().size(); ++index) {
            const RenderPassDescription& pass = graph.passes()[index];
            if (!pass.enabled) continue;
            compiled_.push_back({index, pass.kind});
            for (const RenderResourceUse& use : pass.resources) {
                if (use.resource.value >= last_use.size()) return false;
                if (last_use[use.resource.value]) {
                    const auto [previous, previous_pass] = *last_use[use.resource.value];
                    if (previous != use.access || previous == RenderAccess::Write
                        || use.access == RenderAccess::Write) {
                        barriers_.push_back({use.resource, previous, use.access,
                                              previous_pass, index});
                    }
                }
                last_use[use.resource.value] = std::pair{use.access, index};
            }
        }
        return !compiled_.empty();
    }

    const std::vector<CompiledRenderPass>& passes() const { return compiled_; }
    const std::vector<CompiledResourceBarrier>& barriers() const { return barriers_; }

    bool hasPass(RenderPassKind kind) const {
        for (const CompiledRenderPass& pass : compiled_) {
            if (pass.kind == kind) return true;
        }
        return false;
    }

private:
    std::vector<CompiledRenderPass> compiled_;
    std::vector<CompiledResourceBarrier> barriers_;
};

}
