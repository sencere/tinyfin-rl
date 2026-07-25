#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

use_raylib="${USE_RAYLIB:-0}"
raylib_mode="${RAYLIB_MODE:-auto}"
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
  if [[ "$raylib_mode" == "auto" ]]; then
    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists raylib; then
      raylib_mode="system"
      echo "using system raylib: $(pkg-config --modversion raylib)"
    else
      raylib_mode="vendored"
    fi
  fi
  case "$raylib_mode" in
    vendored) BUILD_TINYFIN_RL=0 TFRL_RAYLIB_DEP=1 bash "$root/scripts/build_raylib.sh" ;;
    system)
      if ! command -v pkg-config >/dev/null 2>&1 || ! pkg-config --exists raylib; then
        echo "error: RAYLIB_MODE=system requires raylib.pc to be available through pkg-config" >&2
        exit 1
      fi
      ;;
    *) echo "error: RAYLIB_MODE must be auto, system, or vendored" >&2; exit 1 ;;
  esac
fi

if [[ "$clean" == "1" ]]; then
  make -C "$root" clean
fi

if [[ "$build_env_lib_only" == "1" ]]; then
  make -C "$root" "$root/build/libtfrl_env.so"
  echo "libtfrl_env built: $root/build/libtfrl_env.so"
  exit 0
fi

mkdir -p "$root/build"
viewer_mode="headless"
if [[ "$use_raylib" == "1" ]]; then
  viewer_mode="raylib"
fi
viewer_mode_file="$root/build/.tinyfin_rl_viewer_mode"
if [[ -f "$root/build/tinyfin-rl" ]]; then
  current_viewer_mode=""
  if [[ -f "$viewer_mode_file" ]]; then
    current_viewer_mode="$(cat "$viewer_mode_file")"
  fi
  if [[ "$current_viewer_mode" != "$viewer_mode" ]]; then
    rm -f "$root/build/tinyfin-rl"
  fi
fi

if [[ "$use_raylib" == "1" ]]; then
  make -C "$root" USE_RAYLIB=1 RAYLIB_MODE="$raylib_mode"
else
  make -C "$root"
fi
printf "%s\n" "$viewer_mode" > "$viewer_mode_file"

echo "tinyfin-rl built: $root/build/tinyfin-rl"
echo "libtfrl_env built: $root/build/libtfrl_env.so"
if [[ "$use_raylib" == "1" ]]; then
  if [[ "$raylib_mode" == "vendored" ]]; then
    echo "runtime hint: export LD_LIBRARY_PATH=\"$root/raylib-src/src:$root/tinyfin:${LD_LIBRARY_PATH:-}\""
  else
    echo "raylib mode: system"
  fi
fi
