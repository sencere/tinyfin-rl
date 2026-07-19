# CLI

`tinyfin-rl` is a single executable with three modes: `train`, `eval`, `replay`.

## Train

```bash
./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 2000
```

Common flags:

- `--algo dqn|ppo|nca|random`
- `--env maze_rooms|lineworld|lineworld_duo|lineworld_cont|point1d|coin_maze|coin_maze_duo`
- `--env py:gymnasium:ENV_ID` (Python bridge)
- `--env py:pettingzoo:MODULE` (Python bridge, e.g. `py:pettingzoo:butterfly.pistonball_v6`)
- `--env py:retro:ENV_ID` (Python bridge)
- `--envs N` (vectorized stepping)
- `--threads N` (parallel stepping)
- `--agents N` (expected agent count; validates env)
- `--steps N`
- `--seed N`
- `--gamma G`
- `--lr LR`
- `--epsilon E` (DQN)
- `--clip-eps E` (PPO)
- `--kl-target K` (PPO)
- `--gae-lambda L` (PPO)
- `--steps-per-batch N` (PPO/REINFORCE)
- `--epochs N` (PPO)
- `--replay-size N` (DQN)
- `--batch-size N` (DQN)
- `--train-every N` (DQN)
- `--learning-starts N` (DQN)
- `--grad-steps N` (DQN)
- `--per-alpha N` (DQN)
- `--per-beta N` (DQN)
- `--entropy-coef N` (PPO)
- `--epsilon E` (DQN/NCA exploration)
- `--save PATH`
- `--load PATH`
- `--log-every N`
- `--render off|live`
- `--render-env N`
- `--render-every N`
- `--render-fps N`
- `--trace-out FILE`
- `--profile` (print profiling summary)
- `--profile-json FILE` (write JSON profile summary)
- `--mp-actors N` (multiprocess actor count for DQN)
- `--mp-queue N` (shared queue capacity for mp mode)
- `--mp-sync-every N` (sync interval for actor policy reload)
- `--mp-sync-path PATH` (sync checkpoint prefix)
- `--deterministic` (single-thread stepping + fixed seed when `--seed` is 0)
- `--share-policy` (use one policy for all agents)
- `--backend cpu|cuda|blas`
- `--device cpu|gpu`

Removed v1 placeholder algorithm names now fail fast: `rainbow`, `qrdqn`,
`iqn`, `iql`, `a2c`, `a3c`, `trpo`, `sac`, `td3`, `impala`, and `mcts`.

Defaults:

- Algorithm defaults are applied from the schema in `src/core/algo_factory.c` (defaults v1).
- Omitted flags fall back to per-algo defaults rather than global CLI values.
- Resolved defaults are logged at startup as an `algo_config` line.

CUDA backend:

- Build Tinyfin with CUDA support (`ENABLE_CUDA=1`) and run with `--backend cuda --device gpu`.
- You can also set `TINYFIN_BACKEND=cuda` and `TINYFIN_DEVICE=gpu` in the environment.

## Eval

```bash
./build/tinyfin-rl eval --algo dqn --env maze_rooms --episodes 10 --load runs/dqn
```

## Replay

```bash
./build/tinyfin-rl replay --trace-in runs/run.tft --render-fps 10
./build/tinyfin-rl replay --trace-in runs/run.tft --dump-meta-json
```

## Production Runner

Minimal inference-only binary that loads saved weights and steps one env:

```bash
make build/tinyfin-prod
./build/tinyfin-prod --algo dqn --env lineworld --load runs/dqn --steps 1000 --deterministic
```

Flags:

- `--algo NAME`
- `--env NAME`
- `--load PATH`
- `--steps N`
- `--episodes N`
- `--seed N`
- `--deterministic`
- `--print-every N` (emit action/reward every N steps)
