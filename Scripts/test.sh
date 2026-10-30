#!/usr/bin/env bash
# Serial test runner. Shared Metal state makes in-process parallel tests
# unreliable. MLX's test-bundle lookup can miss the metallib, so build tests
# first and expose the built library at MLX's working-directory fallback path.
# Pass any extra arguments through, for example --filter.

if [[ "${1:-}" == "--package-path" ]]; then
  shift 2
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_directory/.."

# Build/discover tests without executing them, then locate the generated MLX
# metallib. The release build may have emitted it outside the test bundle.
swift test list >/dev/null || exit $?

metallib_path="$(find .build -type f -name default.metallib -print -quit)"
if [[ -z "$metallib_path" ]]; then
  echo 'MLX default.metallib was not produced under .build.' >&2
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

swift test --skip-build --no-parallel "$@"
