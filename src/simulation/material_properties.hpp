#pragma once

namespace nbody {

enum class MaterialPreset {
    Custom,
    Rocky,
    Metallic,
    Icy,
    Gas
};

// SI units are used for material values: density kg/m^3, strengths Pa,
// and energy thresholds J/kg. Values are heuristic and intentionally exposed.
struct MaterialProperties {
    MaterialPreset preset{MaterialPreset::Custom};
    double density{1000.0};
    double compressive_strength{1.0e7};
    double tensile_strength{1.0e6};
    double brittleness{0.5};
    double energy_absorption{0.5};
    double restitution{1.0};
    double damage_threshold{1.0e4};
    double fragmentation_threshold{1.0e5};
};

MaterialProperties materialPreset(MaterialPreset preset);
void applyMaterialPreset(MaterialProperties& material, MaterialPreset preset);

}
