#pragma once

#include "rendering/render_scene.hpp"

#include <span>
#include <vector>

namespace nbody::rendering {

// The registry owns visual materials, keeping appearance policy out of
// simulation bodies and making the same scene model usable by Vulkan/WebGPU.
class MaterialRegistry {
public:
    MaterialId add(RenderMaterial material) {
        materials_.push_back(material);
        return material.id;
    }

    const RenderMaterial* find(MaterialId id) const {
        for (const RenderMaterial& material : materials_) {
            if (material.id == id) return &material;
        }
        return nullptr;
    }

    std::span<const RenderMaterial> view() const { return materials_; }

private:
    std::vector<RenderMaterial> materials_;
};

}
