#pragma once

#include "app/simulation_session.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace nbody::io {

inline constexpr std::uint32_t current_snapshot_format_version = 1;

struct SnapshotMetadata {
    std::string name;
};

struct SimulationSnapshot {
    std::uint32_t format_version{current_snapshot_format_version};
    app::SimulationState state;
    SnapshotMetadata metadata;
};

enum class SnapshotStatus { Succeeded, Failed };

struct SnapshotResult {
    SnapshotStatus status{SnapshotStatus::Failed};
    std::string message;
    SimulationSnapshot snapshot;

    bool succeeded() const { return status == SnapshotStatus::Succeeded; }
};

class SnapshotSerializer {
public:
    static SnapshotResult load(const std::filesystem::path& path);
    static SnapshotResult save(const std::filesystem::path& path,
                               const SimulationSnapshot& snapshot);
};

}
