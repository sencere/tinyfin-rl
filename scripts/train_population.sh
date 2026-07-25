#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin="${TFRL_BIN:-$root/build/tinyfin-rl}"
algo="nca-ppo"
env_name=""
seeds="1,2,3,4"
steps=1000000
episodes=10
out_dir="$root/runs/population"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo) algo="$2"; shift 2;;
    --env) env_name="$2"; shift 2;;
    --seeds) seeds="$2"; shift 2;;
    --steps) steps="$2"; shift 2;;
    --episodes) episodes="$2"; shift 2;;
    --output) out_dir="$2"; shift 2;;
    -h|--help)
      echo "usage: $0 --env ENV [--algo ALGO] [--seeds 1,2,3] [--steps N] [--episodes N] [--output DIR]"
      exit 0;;
    *) echo "unknown option: $1" >&2; exit 1;;
  esac
done

[[ -n "$env_name" ]] || { echo "error: --env is required" >&2; exit 1; }
mkdir -p "$out_dir"
best_score=""
best_model=""
IFS=',' read -r -a seed_list <<< "$seeds"

for seed in "${seed_list[@]}"; do
  model="$out_dir/seed_${seed}.model"
  log="$out_dir/seed_${seed}.eval.log"
  echo "training seed $seed"
  "$bin" train --algo "$algo" --env "$env_name" --steps "$steps" --seed "$seed" --save "$model" --render off
  "$bin" eval --algo "$algo" --env "$env_name" --load "$model" --episodes "$episodes" --seed "$seed" --deterministic --render off > "$log"
  score="$(awk -F'[ =]' '/^avg_return=/{print $2; exit}' "$log")"
  [[ -n "$score" ]] || { echo "error: no avg_return found for seed $seed" >&2; cat "$log" >&2; exit 1; }
  echo "seed $seed score $score"
  if [[ -z "$best_score" ]] || awk "BEGIN { exit !($score > $best_score) }"; then
    best_score="$score"
    best_model="$model"
  fi
done

cp "$best_model" "$out_dir/best.model"
printf 'best_model=%s\nbest_score=%s\n' "$best_model" "$best_score" > "$out_dir/leaderboard.txt"
echo "best model: $out_dir/best.model (score $best_score)"
