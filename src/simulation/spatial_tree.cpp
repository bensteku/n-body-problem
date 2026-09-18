#include "simulation/spatial_tree.hpp"

#include <algorithm>
#include <cmath>

namespace nbody {

namespace {

double positiveRadius(double radius) {
    return std::max(0.0, radius);
}

}

SpatialTree::SpatialTree(Dimension dimension, std::size_t leaf_capacity,
                         std::size_t maximum_depth, double looseness)
    : dimension_(dimension), leaf_capacity_(std::max<std::size_t>(1, leaf_capacity)),
      maximum_depth_(maximum_depth), looseness_(std::max(1.0, looseness)) {}

void SpatialTree::rebuild(const BodyStorage& bodies) {
    nodes_.clear();
    if (bodies.size() == 0) return;

    SpatialBounds bounds;
    const ConstBodyView first = bodies.view(0);
    const double first_radius = positiveRadius(first.radius);
    bounds.minimum = first.position - Vec3{first_radius, first_radius, first_radius};
    bounds.maximum = first.position + Vec3{first_radius, first_radius, first_radius};
    for (std::size_t index = 1; index < bodies.size(); ++index) {
        const ConstBodyView body = bodies.view(index);
        const double radius = positiveRadius(body.radius);
        bounds.minimum.x = std::min(bounds.minimum.x, body.position.x - radius);
        bounds.minimum.y = std::min(bounds.minimum.y, body.position.y - radius);
        bounds.minimum.z = std::min(bounds.minimum.z, body.position.z - radius);
        bounds.maximum.x = std::max(bounds.maximum.x, body.position.x + radius);
        bounds.maximum.y = std::max(bounds.maximum.y, body.position.y + radius);
        bounds.maximum.z = std::max(bounds.maximum.z, body.position.z + radius);
    }

    // Keep the root non-degenerate so midpoint subdivision remains well-defined.
    constexpr double minimum_extent = 1e-9;
    if (bounds.maximum.x - bounds.minimum.x < minimum_extent) bounds.maximum.x = bounds.minimum.x + minimum_extent;
    if (bounds.maximum.y - bounds.minimum.y < minimum_extent) bounds.maximum.y = bounds.minimum.y + minimum_extent;
    if (dimension_ == Dimension::Three && bounds.maximum.z - bounds.minimum.z < minimum_extent) {
        bounds.maximum.z = bounds.minimum.z + minimum_extent;
    }

    nodes_.push_back({bounds, {}, {}});
    for (std::size_t index = 0; index < bodies.size(); ++index) insert(0, index, bodies, 0);
}

void SpatialTree::insert(std::size_t node_index, std::size_t body_index,
                         const BodyStorage& bodies, std::size_t depth) {
    Node& node = nodes_[node_index];
    const ConstBodyView body = bodies.view(body_index);
    if (node.children.empty()) {
        node.bodies.push_back(body_index);
        if (node.bodies.size() > leaf_capacity_ && depth < maximum_depth_) subdivide(node_index, bodies, depth);
        return;
    }

    const std::size_t child_count = node.children.size();
    for (std::size_t child = 0; child < child_count; ++child) {
        const Node& child_node = nodes_[node.children[child]];
        if (containsSphere(child_node.bounds, body.position, positiveRadius(body.radius))) {
            insert(node.children[child], body_index, bodies, depth + 1);
            return;
        }
    }
    node.bodies.push_back(body_index);
}

void SpatialTree::subdivide(std::size_t node_index, const BodyStorage& bodies, std::size_t depth) {
    const SpatialBounds bounds = nodes_[node_index].bounds;
    const std::size_t child_count = dimension_ == Dimension::Two ? 4 : 8;
    nodes_[node_index].children.reserve(child_count);
    for (std::size_t child = 0; child < child_count; ++child) {
        nodes_.push_back({childBounds(bounds, child), {}, {}});
        nodes_[node_index].children.push_back(nodes_.size() - 1);
    }

    const std::vector<std::size_t> previous = std::move(nodes_[node_index].bodies);
    nodes_[node_index].bodies.clear();
    for (const std::size_t body_index : previous) insert(node_index, body_index, bodies, depth);
}

bool SpatialTree::containsSphere(const SpatialBounds& bounds, const Vec3& position, double radius) const {
    if (position.x - radius < bounds.minimum.x || position.x + radius > bounds.maximum.x
        || position.y - radius < bounds.minimum.y || position.y + radius > bounds.maximum.y) return false;
    return dimension_ == Dimension::Two
        || (position.z - radius >= bounds.minimum.z && position.z + radius <= bounds.maximum.z);
}

bool SpatialTree::intersectsSphere(const SpatialBounds& bounds, const Vec3& position, double radius) const {
    const Vec3 center = (bounds.minimum + bounds.maximum) * 0.5;
    SpatialBounds query_bounds = bounds;
    const Vec3 half_extent = (bounds.maximum - bounds.minimum) * (0.5 * looseness_);
    query_bounds.minimum = center - half_extent;
    query_bounds.maximum = center + half_extent;
    const double closest_x = std::clamp(position.x, query_bounds.minimum.x, query_bounds.maximum.x);
    const double closest_y = std::clamp(position.y, query_bounds.minimum.y, query_bounds.maximum.y);
    const double closest_z = dimension_ == Dimension::Two
        ? position.z : std::clamp(position.z, query_bounds.minimum.z, query_bounds.maximum.z);
    const Vec3 delta{position.x - closest_x, position.y - closest_y, position.z - closest_z};
    return delta.lengthSquared(dimension_) <= radius * radius;
}

SpatialBounds SpatialTree::childBounds(const SpatialBounds& bounds, std::size_t child) const {
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

void SpatialTree::query(std::size_t node_index, std::size_t body_index,
                        const BodyStorage& bodies, std::vector<std::size_t>& output) const {
    const ConstBodyView body = bodies.view(body_index);
    const Node& node = nodes_[node_index];
    if (!intersectsSphere(node.bounds, body.position, positiveRadius(body.radius))) return;
    output.insert(output.end(), node.bodies.begin(), node.bodies.end());
    for (const std::size_t child : node.children) query(child, body_index, bodies, output);
}

void SpatialTree::potentialContactPairs(const BodyStorage& bodies,
                                        std::vector<std::pair<std::size_t, std::size_t>>& output) const {
    output.clear();
    if (nodes_.empty()) return;
    for (std::size_t first = 0; first < bodies.size(); ++first) {
        query_candidates_.clear();
        query(0, first, bodies, query_candidates_);
        for (const std::size_t second : query_candidates_) {
            if (second <= first) continue;
            const ConstBodyView first_body = bodies.view(first);
            const ConstBodyView second_body = bodies.view(second);
            const Vec3 displacement = second_body.position - first_body.position;
            const double radius = positiveRadius(first_body.radius) + positiveRadius(second_body.radius);
            if (displacement.lengthSquared(dimension_) <= radius * radius) output.emplace_back(first, second);
        }
    }
}

void SpatialTree::potentialBodyPairs(const BodyStorage& bodies,
                                     std::vector<std::pair<std::size_t, std::size_t>>& output) const {
    output.clear();
    if (nodes_.empty()) return;
    for (std::size_t first = 0; first < bodies.size(); ++first) {
        query_candidates_.clear();
        query(0, first, bodies, query_candidates_);
        for (const std::size_t second : query_candidates_) {
            if (second > first) output.emplace_back(first, second);
        }
    }
}

}
