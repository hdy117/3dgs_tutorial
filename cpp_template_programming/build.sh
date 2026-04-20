#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=========================================="
echo "  C++ Template Programming Tutorial"
echo "  Build script"
echo "=========================================="

# Clean previous build
if [ -d "${BUILD_DIR}" ]; then
    echo "[clean] Removing old build directory..."
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
echo ""
echo "[cmake] Configuring..."
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Build all examples in parallel
echo ""
echo "[build] Compiling (parallel, 4 cores)..."
make -j4

echo ""
echo "=========================================="
echo "  All examples compiled successfully!"
echo "=========================================="
echo ""
echo "Run individual examples:"
find "${BUILD_DIR}/bin" -type f -executable | sort
