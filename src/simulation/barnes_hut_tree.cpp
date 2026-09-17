#include "simulation/barnes_hut_tree.hpp"

#include <algorithm>
#include <cmath>

namespace nbody {

BarnesHutTree::BarnesHutTree(Dimension dimension, std::size_t leaf_capacity,
                             std::size_t maximum_depth)
    : dimension_(dimension), leaf_capacity_(std::max<std::size_t>(1, leaf_capacity)),
      maximum_depth_(maximum_depth) {}

void BarnesHutTree::rebuild(const BodyStorage& bodies) {
    nodes_.clear();
    if (bodies.size() == 0) return;

    SpatialBounds bounds;
    bounds.minimum = bodies.view(0).position;
    bounds.maximum = bounds.minimum;
    for (std::size_t index = 1; index < bodies.size(); ++index) {
        const Vec3 position = bodies.view(index).position;
        bounds.minimum.x = std::min(bounds.minimum.x, position.x);
        bounds.minimum.y = std::min(bounds.minimum.y, position.y);
        bounds.minimum.z = std::min(bounds.minimum.z, position.z);
        bounds.maximum.x = std::max(bounds.maximum.x, position.x);
        bounds.maximum.y = std::max(bounds.maximum.y, position.y);
        bounds.maximum.z = std::max(bounds.maximum.z, position.z);
    }

    constexpr double padding = 1e-9;
    if (bounds.maximum.x - bounds.minimum.x < padding) bounds.maximum.x = bounds.minimum.x + padding;
    if (bounds.maximum.y - bounds.minimum.y < padding) bounds.maximum.y = bounds.minimum.y + padding;
    if (dimension_ == Dimension::Three && bounds.maximum.z - bounds.minimum.z < padding) {
        bounds.maximum.z = bounds.minimum.z + padding;
    }
    nodes_.push_back({bounds, {}, 0, {}, 0.0, {}});
    for (std::size_t index = 0; index < bodies.size(); ++index) insert(0, index, bodies, 0);
    aggregate(0, bodies);
}

void BarnesHutTree::insert(std::size_t node_index, std::size_t body_index,
                            const BodyStorage& bodies, std::size_t depth) {
    if (nodes_[node_index].child_count == 0) {
        nodes_[node_index].bodies.push_back(body_index);
        if (nodes_[node_index].bodies.size() > leaf_capacity_ && depth < maximum_depth_) {
            subdivide(node_index, bodies, depth);
        }
        return;
    }

    const Vec3 position = bodies.view(body_index).position;
    for (std::size_t child_index = 0; child_index < nodes_[node_index].child_count; ++child_index) {
        const std::size_t child = nodes_[node_index].children[child_index];
        if (containsBody(nodes_[child].bounds, position)) {
            insert(child, body_index, bodies, depth + 1);
            return;
        }
    }
    nodes_[node_index].bodies.push_back(body_index);
}

void BarnesHutTree::subdivide(std::size_t node_index, const BodyStorage& bodies, std::size_t depth) {
    const SpatialBounds bounds = nodes_[node_index].bounds;
    const std::size_t child_count = dimension_ == Dimension::Two ? 4 : 8;
    nodes_[node_index].child_count = static_cast<std::uint8_t>(child_count);
    for (std::size_t child = 0; child < child_count; ++child) {
        nodes_.push_back({childBounds(bounds, child), {}, 0, {}, 0.0, {}});
        nodes_[node_index].children[child] = nodes_.size() - 1;
    }
    const std::vector<std::size_t> previous = std::move(nodes_[node_index].bodies);
    nodes_[node_index].bodies.clear();
    for (const std::size_t body : previous) insert(node_index, body, bodies, depth);
}

void BarnesHutTree::aggregate(std::size_t node_index, const BodyStorage& bodies) {
    Node& node = nodes_[node_index];
    node.mass = 0.0;
    node.center_of_mass = {};
    for (const std::size_t body_index : node.bodies) {
        const ConstBodyView body = bodies.view(body_index);
        node.mass += body.mass;
        node.center_of_mass += body.position * body.mass;
    }
    for (std::size_t child_index = 0; child_index < node.child_count; ++child_index) {
        const std::size_t child = node.children[child_index];
        aggregate(child, bodies);
        node.mass += nodes_[child].mass;
        node.center_of_mass += nodes_[child].center_of_mass * nodes_[child].mass;
    }
    if (node.mass > 0.0) node.center_of_mass = node.center_of_mass * (1.0 / node.mass);
}

bool BarnesHutTree::containsPoint(const SpatialBounds& bounds, const Vec3& position) const {
    return position.x >= bounds.minimum.x && position.x <= bounds.maximum.x
        && position.y >= bounds.minimum.y && position.y <= bounds.maximum.y
        && (dimension_ == Dimension::Two || (position.z >= bounds.minimum.z && position.z <= bounds.maximum.z));
}

bool BarnesHutTree::containsBody(const SpatialBounds& bounds, const Vec3& position) const {
    return containsPoint(bounds, position);
}

SpatialBounds BarnesHutTree::childBounds(const SpatialBounds& bounds, std::size_t child) const {
    const Vec3 midpoint = (bounds.minimum + bounds.maximum) * 0.5;
    SpatialBounds result = bounds;
    if ((child & 1U) != 0) result.minimum.x = midpoint.x;
    else result.maximum.x = midpoint.x;
    if ((child & 2U) != 0) result.minimum.y = midpoint.y;
    else result.maximum.y = midpoint.y;
    if (dimension_ == Dimension::Three) {
        if ((child & 4U) != 0) result.minimum.z = midpoint.z;
        else result.maximum.z = midpoint.z;
    }
    return result;
}

Vec3 BarnesHutTree::accelerationFromNode(std::size_t node_index, std::size_t target,
                                         const BodyStorage& bodies, double gravitational_constant,
                                         double softening_squared, double opening_angle) const {
    const Node& node = nodes_[node_index];
    if (node.mass <= 0.0) return {};
    const ConstBodyView target_body = bodies.view(target);
    const double distance_squared = (node.center_of_mass - target_body.position).lengthSquared(dimension_)
        + softening_squared;
    const double distance = std::sqrt(distance_squared);
    const double extent = std::max({node.bounds.maximum.x - node.bounds.minimum.x,
                                    node.bounds.maximum.y - node.bounds.minimum.y,
                                    dimension_ == Dimension::Three
                                        ? node.bounds.maximum.z - node.bounds.minimum.z : 0.0});
    const bool target_inside = containsPoint(node.bounds, target_body.position);
    if (node.child_count != 0 && !target_inside && distance > 0.0 && extent / distance < opening_angle) {
        Vec3 displacement = node.center_of_mass - target_body.position;
        if (distance_squared == 0.0) return {};
        const double inverse_distance_cubed = 1.0 / (distance_squared * distance);
        return displacement * (gravitational_constant * node.mass * inverse_distance_cubed);
    }

    Vec3 acceleration{};
    for (const std::size_t body_index : node.bodies) {
        if (body_index == target) continue;
        const ConstBodyView body = bodies.view(body_index);
        const Vec3 displacement = body.position - target_body.position;
        const double body_distance_squared = displacement.lengthSquared(dimension_) + softening_squared;
        if (body_distance_squared == 0.0) continue;
        const double body_distance = std::sqrt(body_distance_squared);
        acceleration += displacement * (gravitational_constant * body.mass
            / (body_distance_squared * body_distance));
    }
    for (std::size_t child_index = 0; child_index < node.child_count; ++child_index) {
        const std::size_t child = node.children[child_index];
        acceleration += accelerationFromNode(child, target, bodies, gravitational_constant,
                                              softening_squared, opening_angle);
    }
    return acceleration;
}

Vec3 BarnesHutTree::accelerationOn(std::size_t target, const BodyStorage& bodies,
                                   double gravitational_constant, double softening_length,
                                   double opening_angle) const {
    if (nodes_.empty()) return {};
    return accelerationFromNode(0, target, bodies, gravitational_constant,
                                softening_length * softening_length,
                                std::max(0.0, opening_angle));
}

}
