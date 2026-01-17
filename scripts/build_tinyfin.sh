#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

enable_cuda="${ENABLE_CUDA:-0}"
clean="${CLEAN:-0}"

if [[ ! -d "$root/tinyfin" ]]; then
  echo "error: missing tinyfin at $root/tinyfin" >&2
  echo "hint: init submodules / fetch tinyfin" >&2
  exit 1
fi

if [[ "$clean" == "1" ]]; then
  make -C "$root/tinyfin" clean
fi

if [[ "$enable_cuda" == "1" ]]; then
  make -C "$root/tinyfin" ENABLE_CUDA=1 libtinyfin.so
else
  make -C "$root/tinyfin" libtinyfin.so
fi

echo "tinyfin built: $root/tinyfin/libtinyfin.so"

