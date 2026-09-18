#pragma once

#include "body_storage.hpp"
#include "dimension.hpp"
#include "spatial_bounds.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nbody {

class BarnesHutTree {
public:
    BarnesHutTree(Dimension dimension, std::size_t leaf_capacity, std::size_t maximum_depth);

    void rebuild(const BodyStorage& bodies);
    Dimension dimension() const { return dimension_; }
    std::size_t leafCapacity() const { return leaf_capacity_; }
    std::size_t maximumDepth() const { return maximum_depth_; }
    std::size_t nodeCount() const { return nodes_.size(); }
    struct NodeView {
        const SpatialBounds& bounds;
        const std::array<std::size_t, 8>& children;
        std::uint8_t child_count;
        std::span<const std::size_t> bodies;
        std::span<const std::size_t> packed_indices;
        std::span<const double> packed_x;
        std::span<const double> packed_y;
        std::span<const double> packed_z;
        std::span<const double> packed_masses;
        double mass;
        const Vec3& center_of_mass;
    };
    struct PackedNodeView {
        const SpatialBounds& bounds;
        const std::array<std::size_t, 8>& children;
        std::uint8_t child_count;
        const std::size_t* packed_indices;
        const double* packed_x;
        const double* packed_y;
        const double* packed_z;
        const double* packed_masses;
        std::size_t packed_count;
        double mass;
        const Vec3& center_of_mass;
    };
    NodeView node(std::size_t index) const;
    PackedNodeView packedNode(std::size_t index) const;
    Vec3 accelerationOn(std::size_t target, const BodyStorage& bodies,
                        double gravitational_constant, double softening_length,
                        double opening_angle) const;
    Vec3 accelerationOn(std::size_t target, const BodyStorage& bodies,
                        double gravitational_constant, double softening_length,
                        double opening_angle, std::vector<std::size_t>& traversal_stack) const;

private:
    struct Node {
        SpatialBounds bounds;
        std::array<std::size_t, 8> children{};
        std::uint8_t child_count{};
        std::vector<std::size_t> bodies;
        std::size_t packed_offset{};
        double mass{};
        Vec3 center_of_mass{};
        std::size_t packed_count{};
        const std::size_t* packed_indices{};
        const double* packed_x{};
        const double* packed_y{};
        const double* packed_z{};
        const double* packed_masses{};
    };

    Dimension dimension_;
    std::size_t leaf_capacity_;
    std::size_t maximum_depth_;
    std::vector<Node> nodes_;
    std::vector<std::size_t> packed_indices_;
    std::vector<double> packed_x_;
    std::vector<double> packed_y_;
    std::vector<double> packed_z_;
    std::vector<double> packed_masses_;

    void insert(std::size_t node_index, std::size_t body_index,
                const BodyStorage& bodies, std::size_t depth);
    void subdivide(std::size_t node_index, const BodyStorage& bodies, std::size_t depth);
    void aggregate(std::size_t node_index, const BodyStorage& bodies);
    void packBodies(std::size_t node_index, const BodyStorage& bodies);
    bool containsPoint(const SpatialBounds& bounds, const Vec3& position) const;
    bool containsBody(const SpatialBounds& bounds, const Vec3& position) const;
    SpatialBounds childBounds(const SpatialBounds& bounds, std::size_t child) const;
};

inline BarnesHutTree::NodeView BarnesHutTree::node(std::size_t index) const {
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

inline BarnesHutTree::PackedNodeView BarnesHutTree::packedNode(std::size_t index) const {
    const Node& value = nodes_[index];
    return {value.bounds, value.children, value.child_count, value.packed_indices,
            value.packed_x, value.packed_y, value.packed_z, value.packed_masses,
            value.packed_count, value.mass, value.center_of_mass};
}

}
