#!/usr/bin/env bash
set -euo pipefail

steps="${1:-20000}"
bin="${TFRL_BIN:-build/tinyfin-rl}"
out_dir="runs/perf_baselines"
run_cuda="${RUN_CUDA:-0}"

mkdir -p "$out_dir"

export TFRL_DASHBOARD=0

run_case() {
  local env="$1"
  local backend="$2"
  local device="$3"
  local out="${out_dir}/${env}_${backend}.json"

  echo "baseline: env=${env} backend=${backend} steps=${steps}"
  "$bin" train --algo dqn --env "$env" --steps "$steps" --envs 16 --threads 4 \
    --backend "$backend" --device "$device" --render off --log-every 1000000 \
    --profile-json "$out"
}

run_case "maze_rooms" "cpu" "cpu"
run_case "lineworld" "cpu" "cpu"

if [ "$run_cuda" -eq 1 ]; then
  run_case "maze_rooms" "cuda" "gpu"
  run_case "lineworld" "cuda" "gpu"
fi
