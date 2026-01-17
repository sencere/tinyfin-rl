#!/usr/bin/env bash
set -euo pipefail

bin="${TFRL_BIN:-build/tinyfin-rl}"
steps="${STEPS:-200}"

export TFRL_DASHBOARD=0

echo "[smoke] mp dqn"
"$bin" train --algo dqn --env maze_rooms --steps "$steps" \
  --mp-actors 2 --mp-queue 1024 --mp-sync-every 50 --mp-sync-path runs/mp_sync_smoke \
  --log-every 1000000 --render off

echo "[smoke] impala telemetry"
"$bin" train --algo impala --env maze_rooms --steps "$steps" \
  --actor-count 2 --learner-batch 2 --learner-batch-auto --queue-capacity 256 \
  --log-every 50 --profile --render off
