#!/usr/bin/env bash
set -euo pipefail

steps="${1:-100000}"
bin="${TFRL_BIN:-build/tinyfin-rl}"

echo "benchmark: ${steps} steps"
time "$bin" train --algo dqn --steps "$steps" --render off --log-every 1000000 > /dev/null
