#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="cpu"
with_raylib="0"
clean="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      backend="${2:-}"
      shift 2
      ;;
    --backend=*)
      backend="${1#*=}"
      shift
      ;;
    --with-raylib)
      with_raylib="1"
      shift
      ;;
    --clean)
      clean="1"
      shift
      ;;
    -h|--help)
      echo "usage: $0 [--backend cpu|cuda] [--with-raylib] [--clean]"
      exit 0
      ;;
    *)
      echo "error: unknown option '$1'" >&2
      exit 1
      ;;
  esac
done

case "$backend" in
  cpu)
    export ENABLE_CUDA=0
    ;;
  cuda|gpu)
    export ENABLE_CUDA=1
    ;;
  *)
    echo "error: unsupported backend '$backend' (expected cpu or cuda)" >&2
    exit 1
    ;;
esac

export USE_RAYLIB="$with_raylib"
export CLEAN="$clean"
bash "$root/scripts/build_tinyfin_rl.sh"
