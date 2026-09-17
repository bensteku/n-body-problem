cmake_minimum_required(VERSION 3.25)
project(n_body_problem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (DEFINED ENV{SFML_DIR})
    set(SFML_DIR "$ENV{SFML_DIR}" CACHE PATH "SFML CMake package directory")
else()
    set(SFML_DIR "D:/Speicher/C++/SFML-3.1.0/lib/cmake/SFML" CACHE PATH "SFML CMake package directory")
endif()
set(SFML_STATIC_LIBRARIES TRUE)
find_package(SFML 3.1 CONFIG REQUIRED COMPONENTS Graphics Window)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.h" "src/*.hpp" "main.cpp")
set(PROJECT_SOURCES ${SOURCES})

if (BUILD_VARIANT STREQUAL "GPU")
    enable_language(CUDA)
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "src/*.cu")
    list(APPEND PROJECT_SOURCES ${CUDA_SOURCES})
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES native)
    find_package(CUDAToolkit REQUIRED)
endif()

add_executable(n_body_problem ${PROJECT_SOURCES})
target_link_libraries(n_body_problem PRIVATE SFML::Graphics SFML::Window)
target_include_directories(n_body_problem PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (FORCE_MODEL STREQUAL "BaHu")
    target_compile_definitions(n_body_problem PRIVATE USE_OCTREE)
else()
    message(STATUS "Legacy full pairwise force model")
endif()

if (BUILD_VARIANT STREQUAL "SIMD")
    target_compile_definitions(n_body_problem PRIVATE USE_SIMD)
    target_compile_options(n_body_problem PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/arch:AVX2>)
elseif (BUILD_VARIANT STREQUAL "GPU")
    target_compile_definitions(n_body_problem PRIVATE USE_CUDA)
    target_link_libraries(n_body_problem PRIVATE CUDA::cudart)
    target_compile_options(n_body_problem PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
else()
    message(STATUS "Legacy scalar CPU backend")
endif()
