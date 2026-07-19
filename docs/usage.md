# Tinyfin-RL v2 Usage

This document describes the current v2 implementation state. The project is
C-first, uses Tinyfin for tensor/autograd work, and runs on CPU by default.

## Build

Use the v2 convenience wrapper for the normal path:

```bash
./scripts/build.sh
```

Build with CUDA support in Tinyfin:

```bash
./scripts/build.sh --backend cuda
```

Build with the optional raylib viewer:

```bash
./scripts/build.sh --with-raylib
```

Build with CUDA and raylib:

```bash
./scripts/build.sh --backend cuda --with-raylib
```

The lower-level Make targets still work:

```bash
make
make PERF=1
make build/tinyfin-prod
```

## Shell Setup

For interactive use, source the environment helper:

```bash
. ./scripts/setup_env.sh
```

It exports:

- `TFRL_HOME`
- `PATH` with `build/`
- `LD_LIBRARY_PATH` with `build/`, `tinyfin/`, and `raylib-src/src/`

After sourcing it, `tinyfin-rl` can be run without the `./build/` prefix.

## Supported Algorithms

v2 currently exposes:

- `dqn`
- `ppo`
- `nca`
- `random`

Removed v1 placeholder names fail fast: `rainbow`, `qrdqn`, `iqn`, `iql`,
`a2c`, `a3c`, `trpo`, `sac`, `td3`, `impala`, and `mcts`.

The current `dqn` and `ppo` implementations are compatibility baselines behind
the v2 API. `nca` is implemented as a trainable 1D cellular policy with
deterministic rollout, TD-style updates, diagnostics, and save/load support.

## List Environments

```bash
./build/tinyfin-rl list-envs
```

Core v2 envs:

- `lineworld`
- `maze_rooms`
- `coin_maze`

Other envs may still appear while the v2 split is in progress.

## Train

Basic DQN smoke run:

```bash
./build/tinyfin-rl train --algo dqn --env lineworld --steps 1000
```

Maze run:

```bash
./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 2000
```

PPO run:

```bash
./build/tinyfin-rl train --algo ppo --env maze_rooms --steps 2000 --steps-per-batch 64 --epochs 2
```

NCA run:

```bash
./build/tinyfin-rl train --algo nca --env lineworld --steps 2000 --epsilon 0.1 --save runs/lineworld.nca
./build/tinyfin-rl eval --algo nca --env lineworld --load runs/lineworld.nca --episodes 10 --deterministic
```

Run multiple environments:

```bash
./build/tinyfin-rl train --algo dqn --env lineworld --envs 16 --steps 2000
```

Run with deterministic stepping:

```bash
./build/tinyfin-rl train --algo dqn --env lineworld --steps 1000 --deterministic --seed 1
```

Write a trace:

```bash
./build/tinyfin-rl train --algo dqn --env lineworld --steps 2000 --trace-out runs/lineworld.tft
```

## Eval

```bash
./build/tinyfin-rl eval --algo dqn --env lineworld --episodes 10 --deterministic --seed 1
```

With a checkpoint prefix:

```bash
./build/tinyfin-rl eval --algo dqn --env lineworld --load runs/dqn --episodes 10
```

## Replay

Replay needs render support. Build with raylib first:

```bash
./scripts/build.sh --with-raylib
./build/tinyfin-rl replay --trace-in runs/lineworld.tft --render-fps 10
```

`trace_meta: ...` is only the trace header printed before the replay window
opens. If no window opens and you see `viewer not available`, rebuild with
`./scripts/build.sh --with-raylib`.

Dump trace metadata without rendering:

```bash
./build/tinyfin-rl replay --trace-in runs/lineworld.tft --dump-meta-json
```

## Dashboard And Metrics

Disable the dashboard for machine-readable logs:

```bash
TFRL_DASHBOARD=0 ./build/tinyfin-rl train --algo dqn --env lineworld --steps 1000 --log-every 100
```

Enable profile output:

```bash
./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 2000 --profile
```

Write profile JSON:

```bash
./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 2000 --profile-json runs/profile.json
```

## Env Manifests

v2 adds manifest files for core environments:

- `envs/lineworld/env.toml`
- `envs/maze_rooms/env.toml`
- `envs/coin_maze/env.toml`

These document the env name, kind, library name, observation/action types, size,
and max step count.

The normal build still compiles environments into the CLI and produces the
combined `build/libtfrl_env.so` for bindings. It also builds separated core env
modules:

- `build/libtfrl_env_lineworld.so`
- `build/libtfrl_env_maze_rooms.so`
- `build/libtfrl_env_coin_maze.so`

The runner can load those modules directly by passing the library path as the
environment:

```bash
./build/tinyfin-rl train --algo random --env build/libtfrl_env_lineworld.so --steps 100
```

Manifest path loading is also supported:

```bash
./build/tinyfin-rl train --algo random --env envs/lineworld/env.toml --steps 100
```

Full manifest-backed discovery without central source edits is still planned.

For the full create, compile, train, evaluate, and production env workflow, see
`docs/env_lifecycle.md`.

## Public C API

Include the v2 public headers from `include/tfrl`:

```c
#include "tfrl/tfrl.h"
```

Or include narrower headers:

```c
#include "tfrl/env.h"
#include "tfrl/algo.h"
#include "tfrl/runner.h"
```

Minimal environment use:

```c
tfrl_env_config cfg = {.name = "lineworld", .seed = 1};
tfrl_env *env = tfrl_env_create(&cfg);
tfrl_obs obs = tfrl_env_reset(env, 1);
tfrl_action action = {.index = 1};
tfrl_step_result step = tfrl_env_step(env, action);
tfrl_env_destroy(env);
```

The old internal include paths under `src/core` still exist as compatibility
shims, but new code should include `tfrl/...`.

## Tests

```bash
./scripts/run_tests.sh
```

Golden smoke runs:

```bash
./scripts/golden_runs.sh
```

Quick checks used during v2 skeleton work:

```bash
make all tests
./scripts/run_tests.sh
./build/tinyfin-rl train --algo dqn --env lineworld --steps 20 --log-every 10 --deterministic --seed 1
./build/tinyfin-rl train --algo ppo --env maze_rooms --steps 10 --log-every 5 --deterministic --seed 1
```
