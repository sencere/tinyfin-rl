# Training

## DQN

```bash
./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 2000 --save runs/dqn
./build/tinyfin-rl eval --algo dqn --env maze_rooms --episodes 5 --load runs/dqn
```

## REINFORCE

```bash
./build/tinyfin-rl train --algo reinforce --env lineworld --steps 2000 --steps-per-batch 256
```

## PPO

```bash
./build/tinyfin-rl train --algo ppo --env maze_rooms --steps 2000 --steps-per-batch 64 --epochs 2 --clip-eps 0.2
```

## Checkpoints

Use `--save PATH` and `--load PATH`:

```bash
./build/tinyfin-rl train --algo dqn --steps 2000 --save runs/dqn
./build/tinyfin-rl eval --algo dqn --episodes 5 --load runs/dqn
```
