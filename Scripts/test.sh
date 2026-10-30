#!/usr/bin/env bash
# Serial test runner. Shared Metal state makes in-process parallel tests
# unreliable. SwiftPM nests MLX's metallib inside each test bundle, so build
# tests first and link it beside the test executable where MLX looks for it.
# Pass any extra arguments through, for example --filter.

if [[ "${1:-}" == "--package-path" ]]; then
  shift 2
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_directory/.."

# Build/discover tests without executing them, then make the bundled metallib
# visible at the path MLX checks first. This works around MLX's lookup failing
# under SwiftPM's swift-testing runner.
swift test list >/dev/null || exit $?

metallib_suffix='/Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib'
linked_metallibs=0
while IFS= read -r -d '' metallib_path; do
  case "$metallib_path" in
    *.xctest"$metallib_suffix") ;;
    *) continue ;;
  esac

  test_bundle="${metallib_path%"$metallib_suffix"}"
  test_binary_directory="$test_bundle/Contents/MacOS"
  [[ -d "$test_binary_directory" ]] || continue
  ln -sf '../Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib' \
    "$test_binary_directory/mlx.metallib"
  linked_metallibs=$((linked_metallibs + 1))
done < <(find .build -type f -path "*${metallib_suffix}" -print0)

if [[ "$linked_metallibs" -eq 0 ]]; then
  echo 'MLX default.metallib was not found in any built test bundle.' >&2
  exit 1
fi

exec swift test --skip-build --no-parallel "$@"
