#pragma once

#include <cstddef>

namespace nbody {

enum class ComputeBackend { Scalar, SIMD, GPU };
enum class ForceModel { Full, BarnesHut };
enum class SolverKind { Full, Approximated };

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
    BarnesHutSettings barnes_hut;
};

}
