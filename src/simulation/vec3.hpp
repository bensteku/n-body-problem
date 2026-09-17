#pragma once

#include <cmath>

#include "dimension.hpp"

namespace nbody {

struct Vec3 {
    double x{};
    double y{};
    double z{};

    constexpr Vec3 operator+(const Vec3& other) const { return {x + other.x, y + other.y, z + other.z}; }
    constexpr Vec3 operator-(const Vec3& other) const { return {x - other.x, y - other.y, z - other.z}; }
    constexpr Vec3 operator*(double scalar) const { return {x * scalar, y * scalar, z * scalar}; }
    constexpr Vec3& operator+=(const Vec3& other) { x += other.x; y += other.y; z += other.z; return *this; }
    constexpr Vec3& operator-=(const Vec3& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }

    double lengthSquared() const { return x * x + y * y + z * z; }
    double lengthSquared2D() const { return x * x + y * y; }
    double lengthSquared(Dimension dimension) const {
        return dimension == Dimension::Two ? lengthSquared2D() : lengthSquared();
    }
    double length2D() const { return std::sqrt(lengthSquared2D()); }
    double length(Dimension dimension) const { return std::sqrt(lengthSquared(dimension)); }
    double dot2D(const Vec3& other) const { return x * other.x + y * other.y; }
    double dot(const Vec3& other, Dimension dimension) const {
        return dimension == Dimension::Two ? dot2D(other) : x * other.x + y * other.y + z * other.z;
    }
    bool isFinite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }
};

inline constexpr bool operator==(const Vec3& left, const Vec3& right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

}
