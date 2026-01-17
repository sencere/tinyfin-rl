# CLI

`tinyfin-rl` is a single executable with three modes: `train`, `eval`, `replay`.

## Train

```bash
./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 2000
```

Common flags:

- `--algo dqn|rainbow|qrdqn|iqn|ppo|reinforce|a2c|a3c|trpo|sac|td3|impala|mcts|random`
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
- `--epsilon E` (DQN family)
- `--clip-eps E` (PPO)
- `--kl-target K` (PPO)
- `--gae-lambda L` (PPO)
- `--steps-per-batch N` (PPO/REINFORCE)
- `--epochs N` (PPO)
- `--replay-size N` (DQN family, SAC, TD3)
- `--batch-size N` (DQN family, SAC, TD3)
- `--train-every N` (DQN family, SAC, TD3)
- `--learning-starts N` (DQN family, SAC, TD3)
- `--grad-steps N` (DQN family, SAC, TD3)
- `--per-alpha N` (DQN family, SAC, TD3)
- `--per-beta N` (DQN family, SAC, TD3)
- `--c51-atoms N` (Rainbow)
- `--c51-vmin V` (Rainbow)
- `--c51-vmax V` (Rainbow)
- `--iqn-quantiles N` (IQN)
- `--iqn-tau-samples N` (IQN)
- `--actor-count N` (IMPALA split)
- `--learner-batch N` (IMPALA split)
- `--learner-batch-auto` (adaptive batch sizing for IMPALA)
- `--queue-capacity N` (IMPALA split)
- `--mcts-sims N` (MCTS)
- `--mcts-depth N` (MCTS)
- `--entropy-coef N` (A2C/IMPALA/SAC)
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
- `--mp-actors N` (multiprocess actor count for off-policy algos)
- `--mp-queue N` (shared queue capacity for mp mode)
- `--mp-sync-every N` (sync interval for actor policy reload)
- `--mp-sync-path PATH` (sync checkpoint prefix)
- `--deterministic` (single-thread stepping + fixed seed when `--seed` is 0)
- `--share-policy` (use one policy for all agents)
- `--backend cpu|cuda|blas`
- `--device cpu|gpu`

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
