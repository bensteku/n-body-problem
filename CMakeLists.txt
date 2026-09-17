cmake_minimum_required(VERSION 3.25)
project(n_body_problem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_MODE "greenfield" CACHE STRING "Build family: greenfield or legacy")
set(BUILD_VARIANT "Scalar" CACHE STRING "Compute backend: Scalar, SIMD, or GPU (Vulkan)")
set(FORCE_MODEL "Full" CACHE STRING "Force calculation: Full or BaHu")

if (BUILD_MODE STREQUAL "legacy")
    add_subdirectory(legacy)
    return()
endif()

if (NOT BUILD_MODE STREQUAL "greenfield")
    message(FATAL_ERROR "BUILD_MODE must be greenfield or legacy (got '${BUILD_MODE}')")
endif()
if (NOT BUILD_VARIANT MATCHES "^(Scalar|SIMD|GPU)$")
    message(FATAL_ERROR "BUILD_VARIANT must be Scalar, SIMD, or GPU (got '${BUILD_VARIANT}')")
endif()
if (NOT FORCE_MODEL MATCHES "^(Full|BaHu)$")
    message(FATAL_ERROR "FORCE_MODEL must be Full or BaHu (got '${FORCE_MODEL}')")
endif()

add_library(nbody_simulation
    src/simulation/boundary_system.cpp
    src/simulation/barnes_hut_tree.cpp
    src/simulation/body_storage.cpp
    src/simulation/collision_system.cpp
    src/simulation/material_properties.cpp
    src/simulation/simd_capabilities.cpp
    src/simulation/spatial_tree.cpp
    src/simulation/world_state.cpp
    src/simulation/solvers/scalar_full_solver.cpp
    src/simulation/solvers/scalar_barnes_hut_solver.cpp
    src/simulation/solvers/simd_full_solver.cpp
    src/simulation/solvers/simd_avx2_kernel.cpp
    src/simulation/solvers/simd_barnes_hut_solver.cpp
    src/simulation/solvers/simd_barnes_hut_kernel.cpp
)
target_include_directories(nbody_simulation PUBLIC ${CMAKE_SOURCE_DIR}/src)
if (MSVC)
    set_source_files_properties(src/simulation/solvers/simd_avx2_kernel.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/simulation/solvers/simd_barnes_hut_kernel.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/simulation/solvers/simd_avx2_kernel.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
    set_source_files_properties(src/simulation/solvers/simd_barnes_hut_kernel.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
endif()

add_executable(nbody_app src/app/main.cpp)
target_link_libraries(nbody_app PRIVATE nbody_simulation)
if (FORCE_MODEL STREQUAL "BaHu")
    target_compile_definitions(nbody_app PRIVATE NBODY_FORCE_MODEL_BARNES_HUT=1)
endif()
if (BUILD_VARIANT STREQUAL "SIMD")
    target_compile_definitions(nbody_app PRIVATE NBODY_BACKEND_SIMD=1)
elseif (BUILD_VARIANT STREQUAL "GPU")
    target_compile_definitions(nbody_app PRIVATE NBODY_BACKEND_GPU=1)
endif()

add_executable(nbody_tests tests/simulation_tests.cpp)
target_link_libraries(nbody_tests PRIVATE nbody_simulation)

add_executable(nbody_benchmark tools/benchmark_main.cpp)
target_link_libraries(nbody_benchmark PRIVATE nbody_simulation)

enable_testing()
add_test(NAME nbody_simulation_tests COMMAND nbody_tests)

message(STATUS "Building greenfield targets")
message(STATUS "Requested backend: ${BUILD_VARIANT}")
message(STATUS "Requested force model: ${FORCE_MODEL}")
