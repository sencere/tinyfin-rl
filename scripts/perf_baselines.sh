#!/usr/bin/env bash
set -euo pipefail

steps="${1:-20000}"
bin="${TFRL_BIN:-build/tinyfin-rl}"
out_dir="runs/perf_baselines"
run_cuda="${RUN_CUDA:-0}"
run_blas="${RUN_BLAS:-0}"

mkdir -p "$out_dir"

export TFRL_DASHBOARD=0

run_case() {
  local env="$1"
  local backend="$2"
  local device="$3"
  local out="${out_dir}/${env}_${backend}.json"
  local threads="${THREADS:-4}"
  local thread_args="--threads ${threads}"
  local train_args=""
  if [ "$backend" = "cpu" ] || [ "$backend" = "blas" ]; then
    if [ "$backend" = "blas" ]; then
      export TINYFIN_BACKEND=blas
    fi
  else
    train_args="--train-every ${TRAIN_EVERY:-8} --batch-size ${BATCH_SIZE:-128}"
    export TINYFIN_CUDA_RESIDENT="${TINYFIN_CUDA_RESIDENT:-1}"
  fi

  echo "baseline: env=${env} backend=${backend} steps=${steps}"
  "$bin" train --algo dqn --env "$env" --steps "$steps" --envs 1 $thread_args $train_args \
    --backend "$backend" --device "$device" --render off --log-every 1000000 \
    --profile-json "$out"
}

run_case "maze_rooms" "cpu" "cpu"
run_case "lineworld" "cpu" "cpu"

if [ "$run_blas" -eq 1 ]; then
  run_case "maze_rooms" "blas" "cpu"
  run_case "lineworld" "blas" "cpu"
fi

if [ "$run_cuda" -eq 1 ]; then
  run_case "maze_rooms" "cuda" "gpu"
  run_case "lineworld" "cuda" "gpu"
fi
