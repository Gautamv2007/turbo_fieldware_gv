#!/usr/bin/env bash
# Serial test runner. Shared Metal state makes in-process parallel tests
# unreliable. MLX also cannot locate its metallib in SwiftPM's Swift Build
# test-runner layout, so use the native layout where the test bundle is visible.
# Pass any extra arguments through, for example --filter.

if [[ "${1:-}" == "--package-path" ]]; then
  shift 2
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_directory/.."
exec swift test --build-system native --no-parallel "$@"
