#pragma once

#include <cstddef>
#include <string_view>

namespace nbody {

enum class ComputeBackend { Scalar, SIMD, GPU };
enum class ForceModel { Full, BarnesHut };
enum class SolverKind { Full, Approximated };
enum class ThreadingMode { SingleThreaded, MultiThreaded };

struct BarnesHutSettings {
    // Smaller values are more accurate and more expensive.
    double opening_angle{0.5};
    std::size_t leaf_capacity{8};
    std::size_t maximum_depth{16};
};

struct SolverConfiguration {
    ComputeBackend backend{ComputeBackend::Scalar};
    ForceModel force_model{ForceModel::Full};
    SolverKind kind{SolverKind::Full};
    ThreadingMode threading{ThreadingMode::SingleThreaded};
    std::size_t worker_count{};
    BarnesHutSettings barnes_hut;
};

struct SolverConfigurationValidation {
    bool valid{true};
    std::string_view message{};
};

inline SolverConfigurationValidation validateSolverConfiguration(
    const SolverConfiguration& configuration) {
    if (configuration.force_model == ForceModel::BarnesHut
        && configuration.kind != SolverKind::Approximated) {
        return {false, "Barnes-Hut force calculation requires an Approximated solver kind"};
    }
    if (configuration.barnes_hut.opening_angle < 0.0
        || configuration.barnes_hut.leaf_capacity == 0
        || configuration.barnes_hut.maximum_depth == 0) {
        return {false, "Barnes-Hut settings must have a non-negative opening angle and non-zero limits"};
    }
    return {};
}

}
