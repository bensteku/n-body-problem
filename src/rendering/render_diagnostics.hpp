#pragma once

#include <cstddef>
#include <cstdint>
#include <array>

namespace nbody::rendering {

struct RenderDiagnostics {
    std::uint64_t frame_count{};
    std::size_t submitted_bodies{};
    std::size_t uploaded_bytes{};
    double cpu_prepare_milliseconds{};
    double command_record_milliseconds{};
    double frame_milliseconds{};
    double frames_per_second{};
    std::size_t compiled_passes{};
    std::size_t resource_barriers{};
    std::array<double, 7> pass_cpu_milliseconds{};
    std::uint64_t validation_messages{};
    std::uint64_t validation_warnings{};
    std::uint64_t validation_errors{};
    bool validation_enabled{};
    bool used_external_gpu_frame{};
};

}
