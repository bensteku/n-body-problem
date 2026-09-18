#include "frontend/vulkan_frontend.hpp"

#include <exception>
#include <iostream>

int main() {
    try {
        return nbody::frontend::VulkanFrontend{}.run();
    } catch (const std::exception& error) {
        std::cerr << "nbody frontend startup failed: " << error.what() << '\n';
        return 1;
    }
}
