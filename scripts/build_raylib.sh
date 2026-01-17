#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

clean="${CLEAN:-0}"

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

