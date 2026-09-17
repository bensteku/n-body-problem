#!/usr/bin/env bash
set -euo pipefail

# Headless Linux build entry point.
# Usage:
#   ./build.sh [Scalar|SIMD|GPU] [Full|BaHu]
#   ./build.sh legacy [Scalar|SIMD|GPU] [Full|BaHu]

usage() {
    printf 'Usage: %s [legacy] [Scalar|SIMD|GPU] [Full|BaHu]\n' "$0" >&2
    exit 2
}

build_mode=greenfield
backend=Scalar
force_model=Full

if [[ "${1:-}" == "legacy" ]]; then
    build_mode=legacy
    backend="${2:-$backend}"
    force_model="${3:-$force_model}"
elif [[ $# -gt 0 ]]; then
    backend="$1"
    force_model="${2:-$force_model}"
fi

case "$backend" in
    Scalar|SIMD|GPU) ;;
    *) echo "ERROR: backend must be Scalar, SIMD, or GPU." >&2; usage ;;
esac

case "$force_model" in
    Full|BaHu) ;;
    *) echo "ERROR: force model must be Full or BaHu." >&2; usage ;;
esac

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
build_dir="$root_dir/build/$build_mode-$backend-$force_model"
build_type="${CMAKE_BUILD_TYPE:-Release}"

echo
echo "Building $build_mode / $backend / $force_model"
echo "Build type: $build_type"
echo "Build directory: $build_dir"
echo

# This only configures and builds. It never launches the graphical application;
# that keeps the script usable on headless Linux systems and in CI/VMs.
cmake -S "$root_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE="$build_type" \
    -DBUILD_MODE="$build_mode" \
    -DBUILD_VARIANT="$backend" \
    -DFORCE_MODEL="$force_model"

cmake --build "$build_dir" --parallel

echo
echo "BUILD SUCCEEDED"
if [[ "$build_mode" == legacy ]]; then
    echo "Executable: $build_dir/n_body_problem"
else
    echo "Executable: $build_dir/nbody_app"
fi
