#pragma once

namespace nbody {

enum class ComputeBackend { Scalar, SIMD, CUDA };
enum class ForceModel { Full, BarnesHut };

struct SolverConfiguration {
    ComputeBackend backend{ComputeBackend::Scalar};
    ForceModel force_model{ForceModel::Full};
};

}
