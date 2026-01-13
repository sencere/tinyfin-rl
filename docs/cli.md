# CLI

`tinyfin-rl` is a single executable with three modes: `train`, `eval`, `replay`.

## Train

```bash
./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 2000
```

Common flags:

- `--algo dqn|rainbow|qrdqn|ppo|reinforce|a2c|trpo|sac|td3|impala|mcts|random`
- `--env maze_rooms|lineworld|lineworld_cont|point1d|coin_maze`
- `--envs N` (vectorized stepping)
- `--threads N` (parallel stepping)
- `--steps N`
- `--seed N`
- `--gamma G`
- `--lr LR`
- `--epsilon E` (DQN family)
- `--clip-eps E` (PPO)
- `--steps-per-batch N` (PPO/REINFORCE)
- `--epochs N` (PPO)
- `--replay-size N` (DQN family, SAC, TD3)
- `--batch-size N` (DQN family, SAC, TD3)
- `--per-alpha N` (DQN family, SAC, TD3)
- `--per-beta N` (DQN family, SAC, TD3)
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
- `--deterministic` (single-thread stepping + fixed seed when `--seed` is 0)

Defaults:

- Algorithm defaults are applied from the schema in `src/core/algo_factory.c` (defaults v1).
- Omitted flags fall back to per-algo defaults rather than global CLI values.

## Eval

```bash
./build/tinyfin-rl eval --algo dqn --env maze_rooms --episodes 10 --load runs/dqn
```

## Replay

```bash
./build/tinyfin-rl replay --trace-in runs/run.tft --render-fps 10
```
