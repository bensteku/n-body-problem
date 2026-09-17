#pragma once

#include <cstddef>

namespace nbody {

enum class ComputeBackend { Scalar, SIMD, GPU };
enum class ForceModel { Full, BarnesHut };

struct BarnesHutSettings {
    // Smaller values are more accurate and more expensive.
    double opening_angle{0.5};
    std::size_t leaf_capacity{8};
    std::size_t maximum_depth{16};
};

struct SolverConfiguration {
    ComputeBackend backend{ComputeBackend::Scalar};
    ForceModel force_model{ForceModel::Full};
    BarnesHutSettings barnes_hut;
};

}
