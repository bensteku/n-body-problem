#pragma once

#include <iostream>
#include <stdexcept>

namespace nbody_test {

inline void require(bool condition, const char* expression, const char* file, int line) {
    if (condition) return;
    std::cerr << file << ':' << line << ": requirement failed: " << expression << '\n';
    throw std::runtime_error("test requirement failed");
}

}

#define REQUIRE(condition) ::nbody_test::require((condition), #condition, __FILE__, __LINE__)
