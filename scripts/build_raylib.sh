#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

clean="${CLEAN:-0}"
build_tinyfin_rl="${BUILD_TINYFIN_RL:-1}"

if [[ ! -d "$root/raylib-src/src" ]]; then
  echo "error: missing raylib sources at $root/raylib-src/src" >&2
  echo "hint: init submodules / fetch raylib-src" >&2
  exit 1
fi

if [[ "$clean" == "1" ]]; then
  make -C "$root/raylib-src/src" clean
fi

make -C "$root/raylib-src/src"

echo "raylib built in: $root/raylib-src/src"
echo "runtime hint: export LD_LIBRARY_PATH=\"$root/raylib-src/src:$root/tinyfin:${LD_LIBRARY_PATH:-}\""

if [[ "$build_tinyfin_rl" == "1" ]]; then
  TFRL_SKIP_RAYLIB=1 USE_RAYLIB=1 bash "$root/scripts/build_tinyfin_rl.sh"
else
  echo "build hint: USE_RAYLIB=1 scripts/build_tinyfin_rl.sh"
fi
