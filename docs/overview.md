# Overview

Tinyfin-RL is a C-first RL system with a single executable and a deterministic
environment core. Tinyfin supplies tensors and autograd; Tinyfin-RL owns the
control loop.

## Quickstart

```bash
make
./build/tinyfin-rl train --algo dqn --steps 1000
./build/tinyfin-rl train --algo dqn --env lineworld --steps 500
./build/tinyfin-rl train --algo dqn --env lineworld_duo --steps 500
./build/tinyfin-rl train --algo dqn --env coin_maze_duo --steps 500 --share-policy
./build/tinyfin-rl train --algo dqn --env py:gymnasium:CartPole-v1 --steps 500
./build/tinyfin-rl train --algo dqn --env lineworld_cont --steps 500
./build/tinyfin-rl train --algo random --env point1d --steps 500
./build/tinyfin-rl train --algo dqn --env coin_maze --steps 500
./build/tinyfin-rl train --algo dqn --steps 1000 --save runs/dqn
./build/tinyfin-rl eval --algo dqn --episodes 5 --load runs/dqn
./build/tinyfin-rl train --algo dqn --env lineworld --envs 16 --steps 2000
./build/tinyfin-rl train --algo dqn --env lineworld --envs 8 --threads 4 --render live --render-env 0
```

With raylib:

```bash
make USE_RAYLIB=1
./build/tinyfin-rl train --algo dqn --steps 1000 --render live --render-fps 10
```

## Modes

```bash
./build/tinyfin-rl train --algo dqn --steps 2000 --trace-out runs/run.tft
./build/tinyfin-rl eval --algo dqn --episodes 10 --render live
make -C raylib-src/src
make clean
make USE_RAYLIB=1
export LD_LIBRARY_PATH="$PWD/raylib-src/src:$PWD/tinyfin:$LD_LIBRARY_PATH"
./build/tinyfin-rl replay --trace-in runs/run.tft --render-fps 10
```

End-to-end demo:

```bash
./scripts/run_demo.sh
```

Tests:

```bash
./scripts/run_tests.sh
```

Golden runs:

```bash
./scripts/golden_runs.sh
```

Includes `lineworld`, `lineworld_cont`, `maze_rooms`, and `point1d` checks.

CI smoke:

```bash
./scripts/ci.sh
```

Bridge smoke (optional):

```bash
./scripts/bridge_smoke.sh
```

## Determinism

For strict reproducibility, use `--deterministic` (single-thread stepping +
fixed seed). See `docs/determinism.md` for details and caveats.
