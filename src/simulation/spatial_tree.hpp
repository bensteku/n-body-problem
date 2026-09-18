#pragma once

#include "body_storage.hpp"
#include "dimension.hpp"
#include "spatial_bounds.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace nbody {

// A transient broad-phase index. It is a quadtree in 2D and an octree in 3D.
// BodyStorage remains authoritative; nodes contain only body indices.
class SpatialTree {
public:
    explicit SpatialTree(Dimension dimension, std::size_t leaf_capacity = 16,
                         std::size_t maximum_depth = 12, double looseness = 1.0);

    void rebuild(const BodyStorage& bodies);
    Dimension dimension() const { return dimension_; }
    void potentialContactPairs(const BodyStorage& bodies,
                               std::vector<std::pair<std::size_t, std::size_t>>& output) const;
    void potentialBodyPairs(const BodyStorage& bodies,
                            std::vector<std::pair<std::size_t, std::size_t>>& output) const;

private:
    struct Node {
        SpatialBounds bounds;
        std::vector<std::size_t> bodies;
        std::vector<std::size_t> children;
    };

    Dimension dimension_;
    std::size_t leaf_capacity_;
    std::size_t maximum_depth_;
    double looseness_;
    std::vector<Node> nodes_;

    void insert(std::size_t node_index, std::size_t body_index,
                const BodyStorage& bodies, std::size_t depth);
    void subdivide(std::size_t node_index, const BodyStorage& bodies, std::size_t depth);
    void query(std::size_t node_index, std::size_t body_index, const BodyStorage& bodies,
               std::vector<std::size_t>& output) const;
    bool containsSphere(const SpatialBounds& bounds, const Vec3& position, double radius) const;
    bool intersectsSphere(const SpatialBounds& bounds, const Vec3& position, double radius) const;
    SpatialBounds childBounds(const SpatialBounds& bounds, std::size_t child) const;
};

}
