cmake_minimum_required(VERSION 3.14)
project(n_body_problem)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (DEFINED ENV{SFML_DIR})
	set(SFML_DIR $ENV{SFML_DIR})
endif()
set(SFML_STATIC_LIBRARIES TRUE)
find_package(SFML 2.5 REQUIRED COMPONENTS graphics window)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.h" "src/*.hpp" "main.cpp" "src/*.cu")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE sfml-graphics sfml-window)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)

message(STATUS "cmake type ${CMAKE_CONFIGURATION_TYPE}")

# Check for custom build variants
if (BUILD_VARIANT STREQUAL "SIMD")
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_SIMD)
    message(STATUS "Building with SIMD support")
elseif (BUILD_VARIANT STREQUAL "CUDA")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit REQUIRED)

    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
    message(STATUS "Building with CUDA support")
else()
    message(STATUS "Building default SISD configuration")
endif()