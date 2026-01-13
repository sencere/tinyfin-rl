#!/usr/bin/env bash
set -euo pipefail

bin="${TFRL_BIN:-build/tinyfin-rl}"
mkdir -p runs

"$bin" train --algo dqn --env maze_rooms --steps 2000 --trace-out runs/demo.tft
"$bin" replay --trace-in runs/demo.tft --render-fps 10
