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
    packed_indices_.clear();
    packed_x_.clear();
    packed_y_.clear();
    packed_z_.clear();
    packed_masses_.clear();
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
    nodes_.push_back({bounds, {}, 0, {}, 0, 0.0, {}});
    for (std::size_t index = 0; index < bodies.size(); ++index) insert(0, index, bodies, 0);
    aggregate(0, bodies);
    packed_indices_.reserve(bodies.size());
    packed_x_.reserve(bodies.size());
    packed_y_.reserve(bodies.size());
    packed_z_.reserve(bodies.size());
    packed_masses_.reserve(bodies.size());
    packBodies(0, bodies);
}

BarnesHutTree::NodeView BarnesHutTree::node(std::size_t index) const {
    const Node& value = nodes_[index];
    return {value.bounds, value.children, value.child_count,
            std::span<const std::size_t>(value.bodies.data(), value.bodies.size()),
            std::span<const std::size_t>(packed_indices_.data() + value.packed_offset, value.bodies.size()),
            std::span<const double>(packed_x_.data() + value.packed_offset, value.bodies.size()),
            std::span<const double>(packed_y_.data() + value.packed_offset, value.bodies.size()),
            std::span<const double>(packed_z_.data() + value.packed_offset, value.bodies.size()),
            std::span<const double>(packed_masses_.data() + value.packed_offset, value.bodies.size()),
            value.mass, value.center_of_mass};
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
        nodes_.push_back({childBounds(bounds, child), {}, 0, {}, 0, 0.0, {}});
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

void BarnesHutTree::packBodies(std::size_t node_index, const BodyStorage& bodies) {
    Node& node = nodes_[node_index];
    node.packed_offset = packed_indices_.size();
    for (const std::size_t body_index : node.bodies) {
        const ConstBodyView body = bodies.view(body_index);
        packed_indices_.push_back(body_index);
        packed_x_.push_back(body.position.x);
        packed_y_.push_back(body.position.y);
        packed_z_.push_back(body.position.z);
        packed_masses_.push_back(body.mass);
    }
    for (std::size_t child_index = 0; child_index < node.child_count; ++child_index) {
        packBodies(node.children[child_index], bodies);
    }
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

Vec3 BarnesHutTree::accelerationOn(std::size_t target, const BodyStorage& bodies,
                                   double gravitational_constant, double softening_length,
                                   double opening_angle) const {
    std::vector<std::size_t> traversal_stack;
    traversal_stack.reserve(nodes_.size());
    return accelerationOn(target, bodies, gravitational_constant, softening_length,
                          opening_angle, traversal_stack);
}

Vec3 BarnesHutTree::accelerationOn(std::size_t target, const BodyStorage& bodies,
                                   double gravitational_constant, double softening_length,
                                   double opening_angle,
                                   std::vector<std::size_t>& traversal_stack) const {
    if (nodes_.empty()) return {};
    const Vec3 target_position = bodies.view(target).position;
    const double softening_squared = softening_length * softening_length;
    const double clamped_opening_angle = std::max(0.0, opening_angle);
    if (traversal_stack.capacity() < nodes_.size()) traversal_stack.reserve(nodes_.size());
    traversal_stack.clear();
    traversal_stack.push_back(0);

    Vec3 acceleration{};
    while (!traversal_stack.empty()) {
        const std::size_t node_index = traversal_stack.back();
        traversal_stack.pop_back();
        const Node& node = nodes_[node_index];
        if (node.mass <= 0.0) continue;

        const double distance_squared = (node.center_of_mass - target_position).lengthSquared(dimension_)
            + softening_squared;
        const double distance = std::sqrt(distance_squared);
        const double extent = std::max({node.bounds.maximum.x - node.bounds.minimum.x,
                                        node.bounds.maximum.y - node.bounds.minimum.y,
                                        dimension_ == Dimension::Three
                                            ? node.bounds.maximum.z - node.bounds.minimum.z : 0.0});
        const bool target_inside = containsPoint(node.bounds, target_position);
        if (node.child_count != 0 && !target_inside && distance > 0.0
            && extent / distance < clamped_opening_angle) {
            const Vec3 displacement = node.center_of_mass - target_position;
            const double inverse_distance_cubed = 1.0 / (distance_squared * distance);
            acceleration += displacement * (gravitational_constant * node.mass * inverse_distance_cubed);
            continue;
        }

        for (const std::size_t body_index : node.bodies) {
            if (body_index == target) continue;
            const ConstBodyView body = bodies.view(body_index);
            const Vec3 displacement = body.position - target_position;
            const double body_distance_squared = displacement.lengthSquared(dimension_) + softening_squared;
            if (body_distance_squared == 0.0) continue;
            const double body_distance = std::sqrt(body_distance_squared);
            acceleration += displacement * (gravitational_constant * body.mass
                / (body_distance_squared * body_distance));
        }
        for (std::size_t child_index = node.child_count; child_index > 0; --child_index) {
            traversal_stack.push_back(node.children[child_index - 1]);
        }
    }
    return acceleration;
}

}
