#pragma once

#include "body_state.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nbody {

struct ConstBodyView {
    const BodyId& id;
    const Vec3& position;
    const Vec3& velocity;
    const double& mass;
    const double& radius;
    const std::uint8_t& static_flag;
    const MaterialProperties& material;
    const double& accumulated_damage;
    const BodyKind& kind;

    BodyState snapshot() const;
    bool is_static() const { return static_flag != 0; }
};

struct MutableBodyView {
    BodyId& id;
    Vec3& position;
    Vec3& velocity;
    double& mass;
    double& radius;
    std::uint8_t& static_flag;
    MaterialProperties& material;
    double& accumulated_damage;
    BodyKind& kind;

    BodyState snapshot() const;
    bool is_static() const { return static_flag != 0; }
};

class BodyStorage {
public:
    std::size_t size() const { return ids_.size(); }
    void reserve(std::size_t count);
    void clear();
    void append(const BodyState& body);
    void synchronizePositionComponents() const;

    ConstBodyView view(std::size_t index) const;
    MutableBodyView mutableView(std::size_t index);
    std::vector<BodyState> snapshot() const;

    const std::vector<Vec3>& positions() const { return positions_; }
    const std::vector<double>& positionX() const { return position_x_; }
    const std::vector<double>& positionY() const { return position_y_; }
    const std::vector<double>& positionZ() const { return position_z_; }
    const std::vector<Vec3>& velocities() const { return velocities_; }
    const std::vector<double>& masses() const { return masses_; }

private:
    // Hot simulation data: contiguous and isolated from material/editor state.
    std::vector<Vec3> positions_;
    // Component views for vectorized force kernels.
    mutable std::vector<double> position_x_;
    mutable std::vector<double> position_y_;
    mutable std::vector<double> position_z_;
    std::vector<Vec3> velocities_;
    std::vector<double> masses_;
    std::vector<double> radii_;
    std::vector<std::uint8_t> static_flags_;

    // Cold/warm state: not loaded by the gravity kernel unless required.
    std::vector<BodyId> ids_;
    std::vector<MaterialProperties> materials_;
    std::vector<double> accumulated_damage_;
    std::vector<BodyKind> kinds_;
};

}
