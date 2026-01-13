# tinyfin-rl

<p align="center">
  <img src="resources/logo/logo.png" alt="tinyfin logo" />
</p>

Tinyfin-RL is a **C-first reinforcement learning system** with a single
executable, a deterministic environment core, and Tinyfin for tensors +
autograd. Rendering is optional and never drives simulation.

## What’s In Here

- `src/` single-binary implementation (`tinyfin-rl`)
- `tinyfin/` tensor + autograd backend
- `raylib-src/` optional viewer dependency
- `build/libtfrl_env.so` shared env library for Python bindings
- `architecture.md` and `roadmap.md` for direction

## Build

Headless build (no raylib):

```bash
make
./build/tinyfin-rl train --algo dqn --steps 1000
```

With raylib viewer:

```bash
make USE_RAYLIB=1
./build/tinyfin-rl train --algo dqn --steps 1000 --render live --render-fps 10
```

Select environment:

```bash
./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 1000
./build/tinyfin-rl train --algo dqn --env lineworld --steps 500
./build/tinyfin-rl train --algo dqn --env lineworld_cont --steps 500
./build/tinyfin-rl train --algo random --env point1d --steps 500
./build/tinyfin-rl train --algo dqn --env lineworld --envs 16 --steps 2000
./build/tinyfin-rl train --algo dqn --env lineworld --envs 8 --render live --render-env 0
```

## Algorithms

```bash
./build/tinyfin-rl train --algo dqn --steps 2000
./build/tinyfin-rl train --algo rainbow --steps 2000
./build/tinyfin-rl train --algo qrdqn --steps 2000
./build/tinyfin-rl train --algo reinforce --steps 2000 --steps-per-batch 512
./build/tinyfin-rl train --algo ppo --steps 2000 --steps-per-batch 64 --epochs 2 --clip-eps 0.2
./build/tinyfin-rl train --algo a2c --steps 2000 --steps-per-batch 64
./build/tinyfin-rl train --algo trpo --steps 2000 --steps-per-batch 64 --clip-eps 0.01
./build/tinyfin-rl train --algo sac --steps 2000 --entropy-coef 0.2
./build/tinyfin-rl train --algo td3 --steps 2000
./build/tinyfin-rl train --algo impala --steps 2000 --steps-per-batch 64
./build/tinyfin-rl train --algo dqn --steps 2000 --save runs/dqn
./build/tinyfin-rl eval --algo dqn --episodes 5 --load runs/dqn
```

## Modes

Train:

```bash
./build/tinyfin-rl train --algo dqn --steps 2000 --trace-out runs/run.tft
```

Eval:

```bash
./build/tinyfin-rl eval --algo dqn --episodes 10 --render live
```

Replay:

```bash
make -C raylib-src/src
make clean
make USE_RAYLIB=1
export LD_LIBRARY_PATH="$PWD/raylib-src/src:$PWD/tinyfin:$LD_LIBRARY_PATH"
./build/tinyfin-rl replay --trace-in runs/run.tft --render-fps 10
```

## Notes

- The canonical environment is a four-room grid (`maze_rooms`).
- Reference environments: `lineworld`, `lineworld_cont`, `point1d`.
- DQN/REINFORCE/PPO plus A2C/TRPO/IMPALA/Rainbow/QR-DQN/SAC/TD3 are implemented in C using Tinyfin.
- Rendering is optional and uses render snapshots, not env-owned raylib.

## Docs

- `docs/overview.md` quickstart and modes
- `docs/algorithms.md` DQN/REINFORCE/PPO usage
- `docs/render_snapshot.md` snapshot format
- `docs/trace.md` trace format
- `docs/perf.md` batch stepping + benchmark
- `docs/cli.md` CLI reference
- `docs/training.md` training recipes
- `docs/replay.md` trace replay
- `docs/install.md` build notes
- `docs/env_api.md` env API + Python binding
- `scripts/run_tests.sh` lightweight test runner
- `python/README.md` optional Python wrapper
- `architecture.md` system design
- `roadmap.md` milestones
