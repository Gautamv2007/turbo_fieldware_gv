#!/usr/bin/env bash
# Serial test runner. Shared Metal state makes in-process parallel tests
# unreliable. The MLX Swift package does not compile Metal shaders with the
# SwiftPM command-line build, so build its metallib from MLX's CMake target.
# Pass any extra arguments through, for example --filter.

if [[ "${1:-}" == "--package-path" ]]; then
  shift 2
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_directory/.."

# Resolve package checkouts before building MLX's Metal library. SwiftPM's
# command-line test build will compile and run the package tests afterward.
swift package resolve || exit $?

mlx_checkout_path="$PWD/.build/checkouts/mlx-swift/Source/Cmlx"
mlx_cmake_source="$mlx_checkout_path/mlx"
mlx_metal_build="$PWD/.build/mlx-metallib-build"
if [[ ! -f "$mlx_cmake_source/CMakeLists.txt" ]]; then
  echo "MLX CMake sources were not found at $mlx_cmake_source." >&2
  exit 1
fi

if ! printf '__METAL_VERSION__\n' | xcrun -sdk macosx metal -E -x metal -P - >/dev/null 2>&1; then
  xcodebuild -downloadComponent MetalToolchain || exit $?
fi
if ! printf '__METAL_VERSION__\n' | xcrun -sdk macosx metal -E -x metal -P - >/dev/null 2>&1; then
  echo 'The Metal compiler is unavailable after installing MetalToolchain.' >&2
  exit 1
fi

cmake -S "$mlx_cmake_source" -B "$mlx_metal_build" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DFETCHCONTENT_SOURCE_DIR_METAL_CPP="$mlx_checkout_path/metal-cpp" \
  -DFETCHCONTENT_SOURCE_DIR_JSON="$mlx_checkout_path/json" \
  -DFETCHCONTENT_SOURCE_DIR_FMT="$mlx_checkout_path/fmt" \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_METAL=ON \
  -DMLX_BUILD_CPU=ON \
  -DMLX_BUILD_GGUF=OFF \
  -DMLX_BUILD_SAFETENSORS=OFF \
  -DMLX_USE_CCACHE=OFF \
  -DMLX_METAL_PATH="$mlx_metal_build/kernels" || exit $?
cmake --build "$mlx_metal_build" --target mlx-metallib --parallel 2 || exit $?

metallib_path="$mlx_metal_build/kernels/mlx.metallib"
if [[ ! -f "$metallib_path" ]]; then
  echo "MLX CMake did not produce $metallib_path." >&2
  exit 1
fi

created_default_metallib_link=0
cleanup() {
  if [[ "$created_default_metallib_link" -eq 1 ]]; then
    rm -f default.metallib
  fi
}
trap cleanup EXIT

if [[ ! -f default.metallib ]]; then
  ln -sf "$metallib_path" default.metallib || exit $?
  created_default_metallib_link=1
fi

swift test --no-parallel "$@"
