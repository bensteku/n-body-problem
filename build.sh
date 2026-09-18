#!/usr/bin/env bash
set -euo pipefail

# Headless Linux build entry point.
# Usage:
#   ./build.sh
#   ./build.sh legacy [Scalar|SIMD|GPU] [Full|BaHu]

usage() {
    printf 'Usage: %s [legacy] [Scalar|SIMD|GPU] [Full|BaHu]\n' "$0" >&2
    exit 2
}

build_mode=greenfield
if [[ "${1:-}" == "legacy" ]]; then
    build_mode=legacy
    backend="${2:-Scalar}"
    force_model="${3:-Full}"
    case "$backend" in Scalar|SIMD|GPU) ;; *) echo "ERROR: invalid legacy backend." >&2; usage ;; esac
    case "$force_model" in Full|BaHu) ;; *) echo "ERROR: invalid legacy force model." >&2; usage ;; esac
elif [[ $# -gt 0 ]]; then
    usage
fi

if [[ "$build_mode" == legacy && $# -gt 3 ]]; then
    usage
elif [[ "$build_mode" == greenfield && $# -gt 2 ]]; then
    usage
fi

command -v cmake >/dev/null 2>&1 || {
    echo "ERROR: cmake was not found on PATH." >&2
    exit 1
}
command -v ninja >/dev/null 2>&1 || {
    echo "ERROR: ninja was not found on PATH." >&2
    exit 1
}

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$build_mode" == legacy ]]; then
    build_dir="$root_dir/build/legacy-$backend-$force_model"
else
    build_dir="$root_dir/build/greenfield"
fi
build_type="${CMAKE_BUILD_TYPE:-Release}"

echo
echo "Building $build_mode"
echo "Build type: $build_type"
echo "Build directory: $build_dir"
echo

# This only configures and builds. It never launches the graphical application;
# that keeps the script usable on headless Linux systems and in CI/VMs.
if [[ "$build_mode" == legacy ]]; then
    cmake -S "$root_dir" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DBUILD_MODE="$build_mode" \
        -DBUILD_VARIANT="$backend" \
        -DFORCE_MODEL="$force_model"
else
    cmake -S "$root_dir" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DBUILD_MODE="$build_mode"
fi

cmake --build "$build_dir" --parallel

echo
echo "BUILD SUCCEEDED"
if [[ "$build_mode" == legacy ]]; then
    echo "Executable: $build_dir/n_body_problem"
else
    echo "Executable: $build_dir/nbody_app"
fi
