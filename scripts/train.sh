#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin="${TFRL_BIN:-"$root/build/tinyfin-rl"}"

algo="${ALGO:-dqn}"
env_name="${ENV_NAME:-maze_rooms}"
steps="${STEPS:-2000}"
envs="${ENVS:-16}"
threads="${THREADS:-4}"
backend="${BACKEND:-cuda}"
device="${DEVICE:-gpu}"
render="${RENDER:-off}"
log_every="${LOG_EVERY:-200}"
profile="${PROFILE:-1}"
save_dir="${SAVE_DIR:-"$root/runs"}"
load_prefix="${LOAD_PREFIX:-}"

# Extra algo tuning (pass only when set).
lr="${LR:-}"
epsilon="${EPSILON:-}"
replay_size="${REPLAY_SIZE:-}"
batch_size="${BATCH_SIZE:-}"
per_alpha="${PER_ALPHA:-}"
per_beta="${PER_BETA:-}"

# Docs-recommended defaults for DQN family unless overridden.
if [[ "$algo" == "dqn" || "$algo" == "rainbow" || "$algo" == "qrdqn" || "$algo" == "iqn" ]]; then
  replay_size="${replay_size:-10000}"
  batch_size="${batch_size:-128}"
  per_alpha="${per_alpha:-0.6}"
  per_beta="${per_beta:-0.4}"
fi

# Tetris defaults (single env, CPU-friendly) unless overridden.
if [[ "$env_name" == "tetris" || "$env_name" == "tetris_disc" ]]; then
  steps="${STEPS:-20000}"
  envs="${ENVS:-1}"
  threads="${THREADS:-1}"
  backend="${BACKEND:-cpu}"
  device="${DEVICE:-cpu}"
  log_every="${LOG_EVERY:-50}"
  replay_size="${replay_size:-20000}"
  batch_size="${batch_size:-128}"
  epsilon="${epsilon:-0.2}"
fi

mkdir -p "$save_dir"

ts="$(date +%Y%m%d_%H%M%S)"
run_name="${RUN_NAME:-${algo}_${env_name}_${ts}}"
save_prefix="${SAVE_PREFIX:-"$save_dir/$run_name"}"
trace_out="${TRACE_OUT:-"$save_prefix.tft"}"

export TINYFIN_BACKEND="${TINYFIN_BACKEND:-$backend}"
export TINYFIN_DEVICE="${TINYFIN_DEVICE:-$device}"

cmd=(
  "$bin" train
  --algo "$algo"
  --env "$env_name"
  --steps "$steps"
  --envs "$envs"
  --threads "$threads"
  --backend "$backend"
  --device "$device"
  --render "$render"
  --log-every "$log_every"
  --save "$save_prefix"
  --trace-out "$trace_out"
)

if [[ -n "$lr" ]]; then
  cmd+=(--lr "$lr")
fi

if [[ -n "$epsilon" ]]; then
  cmd+=(--epsilon "$epsilon")
fi

if [[ -n "$replay_size" ]]; then
  cmd+=(--replay-size "$replay_size")
fi

if [[ -n "$batch_size" ]]; then
  cmd+=(--batch-size "$batch_size")
fi

if [[ -n "$per_alpha" ]]; then
  cmd+=(--per-alpha "$per_alpha")
fi

if [[ -n "$per_beta" ]]; then
  cmd+=(--per-beta "$per_beta")
fi

if [[ -n "$load_prefix" ]]; then
  cmd+=(--load "$load_prefix")
fi

if [[ "$profile" == "1" ]]; then
  cmd+=(--profile)
fi

cmd+=("$@")

printf '+ %q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

echo "saved checkpoint: ${save_prefix}.q.{w,b}.tensor"
echo "saved trace:      ${trace_out}"
echo "replay:           ${bin} replay --trace-in ${trace_out} --render-fps 60"
