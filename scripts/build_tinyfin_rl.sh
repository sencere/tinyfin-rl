#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

use_raylib="${USE_RAYLIB:-0}"
skip_raylib="${TFRL_SKIP_RAYLIB:-0}"
clean="${CLEAN:-0}"
build_tinyfin="${BUILD_TINYFIN:-1}"
build_env_lib_only="${BUILD_ENV_LIB_ONLY:-0}"

if [[ "$build_env_lib_only" == "1" ]]; then
  build_tinyfin="0"
fi

if [[ "$build_tinyfin" == "1" && ! -f "$root/tinyfin/libtinyfin.so" ]]; then
  ENABLE_CUDA="${ENABLE_CUDA:-0}" CLEAN="$clean" bash "$root/scripts/build_tinyfin.sh"
fi

if [[ "$use_raylib" == "1" && "$skip_raylib" != "1" ]]; then
  bash "$root/scripts/build_raylib.sh"
fi

if [[ "$clean" == "1" ]]; then
  make -C "$root" clean
fi

if [[ "$build_env_lib_only" == "1" ]]; then
  make -C "$root" "$root/build/libtfrl_env.so"
  echo "libtfrl_env built: $root/build/libtfrl_env.so"
  exit 0
fi

if [[ "$use_raylib" == "1" ]]; then
  make -C "$root" USE_RAYLIB=1
else
  make -C "$root"
fi

echo "tinyfin-rl built: $root/build/tinyfin-rl"
echo "libtfrl_env built: $root/build/libtfrl_env.so"
if [[ "$use_raylib" == "1" ]]; then
  echo "runtime hint: export LD_LIBRARY_PATH=\"$root/raylib-src/src:$root/tinyfin:${LD_LIBRARY_PATH:-}\""
fi
