# CLI

`tinyfin-rl` is a single executable with three modes: `train`, `eval`, `replay`.

## Train

```bash
./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 2000
```

Common flags:

- `--algo dqn|ppo|reinforce`
- `--env maze_rooms|lineworld`
- `--envs N` (vectorized stepping)
- `--steps N`
- `--seed N`
- `--gamma G`
- `--lr LR`
- `--epsilon E` (DQN)
- `--clip-eps E` (PPO)
- `--steps-per-batch N` (PPO/REINFORCE)
- `--epochs N` (PPO)
- `--save PATH`
- `--load PATH`
- `--log-every N`
- `--render off|live`
- `--render-every N`
- `--render-fps N`
- `--trace-out FILE`

## Eval

```bash
./build/tinyfin-rl eval --algo dqn --env maze_rooms --episodes 10 --load runs/dqn
```

## Replay

```bash
./build/tinyfin-rl replay --trace-in runs/run.tft --render-fps 10
```
