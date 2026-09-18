cmake_minimum_required(VERSION 3.25)
project(n_body_problem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_MODE "greenfield" CACHE STRING "Build family: greenfield or legacy")
if (BUILD_MODE STREQUAL "legacy")
    add_subdirectory(legacy)
    return()
endif()

if (NOT BUILD_MODE STREQUAL "greenfield")
    message(FATAL_ERROR "BUILD_MODE must be greenfield or legacy (got '${BUILD_MODE}')")
endif()

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG QUIET)
if (TARGET glfw)
    set(NBODY_GLFW_TARGET glfw)
elseif (TARGET glfw3)
    set(NBODY_GLFW_TARGET glfw3)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW3 REQUIRED IMPORTED_TARGET glfw3)
    set(NBODY_GLFW_TARGET PkgConfig::GLFW3)
endif()

add_library(nbody_simulation
    src/simulation/boundary_system.cpp
    src/simulation/barnes_hut_tree.cpp
    src/simulation/body_storage.cpp
    src/simulation/collision_system.cpp
    src/simulation/simd_collision_system.cpp
    src/simulation/material_properties.cpp
    src/simulation/simd_capabilities.cpp
    src/simulation/spatial_tree.cpp
    src/simulation/world_state.cpp
    src/simulation/simulation_frame.cpp
    src/simulation/physics_session.cpp
    src/simulation/parallel_executor.cpp
    src/simulation/physics_engine_factory.cpp
    src/simulation/solvers/scalar_full_physics_engine.cpp
    src/simulation/solvers/scalar_approximated_physics_engine.cpp
    src/simulation/solvers/simd_full_physics_engine.cpp
    src/simulation/solvers/simd_approximated_physics_engine.cpp
    src/simulation/solvers/simd_avx2_kernel.cpp
    src/simulation/solvers/simd_barnes_hut_kernel.cpp
)
target_include_directories(nbody_simulation PUBLIC ${CMAKE_SOURCE_DIR}/src)
if (MSVC)
    set_source_files_properties(src/simulation/solvers/simd_avx2_kernel.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/simulation/solvers/simd_barnes_hut_kernel.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/simulation/simd_collision_system.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/simulation/solvers/simd_avx2_kernel.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
    set_source_files_properties(src/simulation/solvers/simd_barnes_hut_kernel.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
    set_source_files_properties(src/simulation/simd_collision_system.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
endif()

add_executable(nbody_app
    src/app/main.cpp
    src/frontend/vulkan_frontend.cpp
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/backends/imgui_impl_glfw.cpp
    third_party/imgui/backends/imgui_impl_vulkan.cpp
)
target_include_directories(nbody_app PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party/imgui
)
target_link_libraries(nbody_app PRIVATE nbody_simulation Vulkan::Vulkan ${NBODY_GLFW_TARGET})

add_executable(nbody_tests tests/simulation_tests.cpp)
target_link_libraries(nbody_tests PRIVATE nbody_simulation)

add_executable(nbody_benchmark tools/benchmark_main.cpp)
target_link_libraries(nbody_benchmark PRIVATE nbody_simulation)

enable_testing()
add_test(NAME nbody_simulation_tests COMMAND nbody_tests)

message(STATUS "Building greenfield targets")
message(STATUS "All Scalar, SIMD, Barnes-Hut, and GPU-fallback capabilities are included")
