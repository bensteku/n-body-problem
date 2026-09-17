#include "simulation/material_properties.hpp"

namespace nbody {

MaterialProperties materialPreset(MaterialPreset preset) {
    switch (preset) {
    case MaterialPreset::Rocky:
        return {preset, 3000.0, 1.0e8, 1.0e7, 0.75, 0.25, 0.45, 1.0e4, 1.0e5};
    case MaterialPreset::Metallic:
        return {preset, 7800.0, 2.5e8, 2.5e8, 0.20, 0.70, 0.60, 2.0e5, 2.0e6};
    case MaterialPreset::Icy:
        return {preset, 1000.0, 1.0e7, 1.0e6, 0.80, 0.15, 0.35, 1.0e3, 1.0e4};
    case MaterialPreset::Gas:
        return {preset, 1.0, 1.0e3, 0.0, 0.05, 0.90, 0.05, 1.0, 10.0};
    case MaterialPreset::Custom:
        return {};
    }
    return {};
}

void applyMaterialPreset(MaterialProperties& material, MaterialPreset preset) {
    material = materialPreset(preset);
}

}
