#!/usr/bin/env bash
set -euo pipefail

storage="${1:-array}"
min_steps="${2:-0}"
envs="${3:-64}"
steps="${4:-200}"

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
