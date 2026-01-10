#!/usr/bin/env bash
set -euo pipefail

storage="${STORAGE:-array}"
min_steps="${MIN_STEPS_PER_S:-0}"
envs="${ENVS:-64}"
steps="${STEPS_PER_ENV:-200}"

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

python examples/benchmark_rollout.py \
  --storage "${storage}" \
  --num-envs "${envs}" \
  --steps "${steps}" \
  --iters 5 \
  --warmup 1 \
  --format text \
  --min-steps-per-s "${min_steps}"
