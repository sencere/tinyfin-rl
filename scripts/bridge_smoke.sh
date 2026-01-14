#!/usr/bin/env sh
set -eu

if ! command -v python3 >/dev/null 2>&1; then
  echo "bridge_smoke: python3 not found, skipping"
  exit 0
fi

if ! python3 - <<'PY'
try:
    import gymnasium  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
then
  echo "bridge_smoke: gymnasium not installed, skipping"
  exit 0
fi

sock="/tmp/tfrl_py_bridge.sock"
rm -f "$sock"
python3 python/bridge_server.py --socket "$sock" >/tmp/tfrl_bridge.log 2>&1 &
pid=$!

trap 'kill $pid >/dev/null 2>&1 || true; rm -f "$sock"' EXIT

tries=0
while [ ! -S "$sock" ] && [ $tries -lt 50 ]; do
  tries=$((tries + 1))
  sleep 0.1
done

export TFRL_PY_BRIDGE="$sock"
./build/tinyfin-rl train --algo dqn --env py:gymnasium:CartPole-v1 --steps 10 --deterministic --seed 1 >/tmp/tfrl_bridge_run.log 2>&1
