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
    NodeView node(std::size_t index) const;
    Vec3 accelerationOn(std::size_t target, const BodyStorage& bodies,
                        double gravitational_constant, double softening_length,
                        double opening_angle) const;

private:
    struct Node {
        SpatialBounds bounds;
        std::array<std::size_t, 8> children{};
        std::uint8_t child_count{};
        std::vector<std::size_t> bodies;
        std::size_t packed_offset{};
        double mass{};
        Vec3 center_of_mass{};
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
    Vec3 accelerationFromNode(std::size_t node_index, std::size_t target, const Vec3& target_position,
                              const BodyStorage& bodies, double gravitational_constant,
                              double softening_squared, double opening_angle) const;
    bool containsPoint(const SpatialBounds& bounds, const Vec3& position) const;
    bool containsBody(const SpatialBounds& bounds, const Vec3& position) const;
    SpatialBounds childBounds(const SpatialBounds& bounds, std::size_t child) const;
};

}
