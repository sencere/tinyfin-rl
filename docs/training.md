# Training

## DQN

```bash
./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 2000 --save runs/dqn
./build/tinyfin-rl eval --algo dqn --env maze_rooms --episodes 5 --load runs/dqn
```

Use replay/PER tuning:

```bash
./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 2000 --replay-size 10000 --batch-size 64 --per-alpha 0.6 --per-beta 0.4
```

## v2 Algorithm Scope

Tinyfin-RL v2 exposes `dqn`, `ppo`, `nca`, and `random`. Removed v1 placeholder names
fail fast until they have real, tested implementations.

## Coin Maze (DQN)

```bash
./build/tinyfin-rl train --algo dqn --env coin_maze --steps 2000 --replay-size 10000 --batch-size 64
```

## PPO

```bash
./build/tinyfin-rl train --algo ppo --env maze_rooms --steps 2000 --steps-per-batch 64 --epochs 2 --clip-eps 0.2 --gae-lambda 0.95 --entropy-coef 0.01 --kl-target 0.01
```

## Multi-Agent (Lineworld Duo)

```bash
./build/tinyfin-rl train --algo dqn --env lineworld_duo --steps 2000 --replay-size 10000 --batch-size 64
```

## Multi-Agent (Coin Maze Duo)

```bash
./build/tinyfin-rl train --algo dqn --env coin_maze_duo --steps 2000 --replay-size 10000 --batch-size 64 --share-policy
```

## Python Bridge (Gymnasium)

```bash
./build/tinyfin-rl train --algo dqn --env py:gymnasium:CartPole-v1 --steps 2000
```

## Checkpoints

Use `--save PATH` and `--load PATH`:

```bash
./build/tinyfin-rl train --algo dqn --steps 2000 --save runs/dqn
./build/tinyfin-rl eval --algo dqn --episodes 5 --load runs/dqn
```

Inference-only runner:

```bash
make build/tinyfin-prod
./build/tinyfin-prod --algo dqn --env lineworld --load runs/dqn --steps 1000 --deterministic
```

## Render after training

Record a trace and replay it:

```bash
./build/tinyfin-rl train --algo dqn --steps 2000 --trace-out runs/run.tft
```

Then build the viewer and replay:

```bash
./scripts/build.sh --with-raylib
./build/tinyfin-rl replay --trace-in runs/run.tft --render-fps 10
```
