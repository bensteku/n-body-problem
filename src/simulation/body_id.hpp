#pragma once

#include <cstdint>

namespace nbody {

struct BodyId {
    std::uint64_t value{};
    constexpr bool isValid() const { return value != 0; }
};

inline constexpr bool operator==(BodyId left, BodyId right) {
    return left.value == right.value;
}

}
