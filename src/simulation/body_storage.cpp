#include "simulation/body_storage.hpp"

#include <stdexcept>

namespace nbody {

BodyState ConstBodyView::snapshot() const {
    return {id, position, velocity, mass, radius, is_static(), material, accumulated_damage, kind};
}

BodyState MutableBodyView::snapshot() const {
    return {id, position, velocity, mass, radius, is_static(), material, accumulated_damage, kind};
}

void BodyStorage::reserve(std::size_t count) {
    positions_.reserve(count);
    position_x_.reserve(count);
    position_y_.reserve(count);
    position_z_.reserve(count);
    velocities_.reserve(count);
    masses_.reserve(count);
    radii_.reserve(count);
    static_flags_.reserve(count);
    ids_.reserve(count);
    materials_.reserve(count);
    accumulated_damage_.reserve(count);
    kinds_.reserve(count);
}

void BodyStorage::clear() {
    positions_.clear();
    position_x_.clear();
    position_y_.clear();
    position_z_.clear();
    velocities_.clear();
    masses_.clear();
    radii_.clear();
    static_flags_.clear();
    ids_.clear();
    materials_.clear();
    accumulated_damage_.clear();
    kinds_.clear();
}

void BodyStorage::append(const BodyState& body) {
    positions_.push_back(body.position);
    position_x_.push_back(body.position.x);
    position_y_.push_back(body.position.y);
    position_z_.push_back(body.position.z);
    velocities_.push_back(body.velocity);
    masses_.push_back(body.mass);
    radii_.push_back(body.radius);
    static_flags_.push_back(body.is_static ? 1 : 0);
    ids_.push_back(body.id);
    materials_.push_back(body.material);
    accumulated_damage_.push_back(body.accumulated_damage);
    kinds_.push_back(body.kind);
}

void BodyStorage::integratePositions(std::span<const Vec3> accelerations,
                                     double timestep, Dimension dimension) {
    if (accelerations.size() != size()) {
        throw std::invalid_argument("position acceleration count does not match body count");
    }
    const double half_timestep_squared = 0.5 * timestep * timestep;
    for (std::size_t index = 0; index < size(); ++index) {
        if (static_flags_[index] != 0) continue;
        positions_[index] += velocities_[index] * timestep
            + accelerations[index] * half_timestep_squared;
        if (dimension == Dimension::Two) positions_[index].z = 0.0;
    }
}

void BodyStorage::integrateVelocities(std::span<const Vec3> initial_accelerations,
                                      std::span<const Vec3> final_accelerations,
                                      double timestep, Dimension dimension) {
    if (initial_accelerations.size() != size() || final_accelerations.size() != size()) {
        throw std::invalid_argument("velocity acceleration count does not match body count");
    }
    const double half_timestep = 0.5 * timestep;
    for (std::size_t index = 0; index < size(); ++index) {
        if (static_flags_[index] != 0) continue;
        velocities_[index] += (initial_accelerations[index] + final_accelerations[index])
            * half_timestep;
        if (dimension == Dimension::Two) velocities_[index].z = 0.0;
    }
}

void BodyStorage::advance(double timestep, Dimension dimension) {
    for (std::size_t index = 0; index < size(); ++index) {
        if (static_flags_[index] != 0) continue;
        positions_[index] += velocities_[index] * timestep;
        if (dimension == Dimension::Two) {
            positions_[index].z = 0.0;
            velocities_[index].z = 0.0;
        }
    }
}

void BodyStorage::synchronizePositionComponents() const {
    if (position_x_.size() != positions_.size()) {
        position_x_.resize(positions_.size());
        position_y_.resize(positions_.size());
        position_z_.resize(positions_.size());
    }
    for (std::size_t index = 0; index < positions_.size(); ++index) {
        position_x_[index] = positions_[index].x;
        position_y_[index] = positions_[index].y;
        position_z_[index] = positions_[index].z;
    }
}

ConstBodyView BodyStorage::view(std::size_t index) const {
    return {ids_[index], positions_[index], velocities_[index], masses_[index], radii_[index],
            static_flags_[index], materials_[index], accumulated_damage_[index], kinds_[index]};
}

MutableBodyView BodyStorage::mutableView(std::size_t index) {
    return {ids_[index], positions_[index], velocities_[index], masses_[index], radii_[index],
            static_flags_[index], materials_[index], accumulated_damage_[index], kinds_[index]};
}

std::vector<BodyState> BodyStorage::snapshot() const {
    std::vector<BodyState> result;
    result.reserve(size());
    for (std::size_t index = 0; index < size(); ++index) result.push_back(view(index).snapshot());
    return result;
}

}
