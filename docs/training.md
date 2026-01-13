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

## Rainbow / QR-DQN

```bash
./build/tinyfin-rl train --algo rainbow --env maze_rooms --steps 2000 --replay-size 10000 --batch-size 64
./build/tinyfin-rl train --algo qrdqn --env maze_rooms --steps 2000 --replay-size 10000 --batch-size 64
```

## Coin Maze (DQN)

```bash
./build/tinyfin-rl train --algo dqn --env coin_maze --steps 2000 --replay-size 10000 --batch-size 64
```

## REINFORCE

```bash
./build/tinyfin-rl train --algo reinforce --env lineworld --steps 2000 --steps-per-batch 256
```

## PPO

```bash
./build/tinyfin-rl train --algo ppo --env maze_rooms --steps 2000 --steps-per-batch 64 --epochs 2 --clip-eps 0.2 --gae-lambda 0.95 --entropy-coef 0.01
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

## A2C / TRPO / IMPALA

```bash
./build/tinyfin-rl train --algo a2c --env maze_rooms --steps 2000 --steps-per-batch 64
./build/tinyfin-rl train --algo trpo --env maze_rooms --steps 2000 --steps-per-batch 64 --clip-eps 0.01
./build/tinyfin-rl train --algo impala --env maze_rooms --steps 2000 --steps-per-batch 64 --clip-eps 1.0
```

## SAC / TD3 (Continuous)

```bash
./build/tinyfin-rl train --algo sac --env point1d --steps 2000 --replay-size 10000 --batch-size 64 --entropy-coef 0.2
./build/tinyfin-rl train --algo td3 --env point1d --steps 2000 --replay-size 10000 --batch-size 64
```

Notes:

- SAC uses a stochastic Gaussian policy with tanh squashing and log-prob correction for entropy.
- TD3 remains deterministic for continuous actions.

## MCTS (Planner)

```bash
./build/tinyfin-rl train --algo mcts --env maze_rooms --steps 2000 --mcts-sims 400 --mcts-depth 80
```

## Checkpoints

Use `--save PATH` and `--load PATH`:

```bash
./build/tinyfin-rl train --algo dqn --steps 2000 --save runs/dqn
./build/tinyfin-rl eval --algo dqn --episodes 5 --load runs/dqn
```

## Render after training

Record a trace and replay it:

```bash
./build/tinyfin-rl train --algo dqn --steps 2000 --trace-out runs/run.tft
```

Then build the viewer and replay:

```bash
make -C raylib-src/src
make clean
make USE_RAYLIB=1
export LD_LIBRARY_PATH="$PWD/raylib-src/src:$PWD/tinyfin:$LD_LIBRARY_PATH"
./build/tinyfin-rl replay --trace-in runs/run.tft --render-fps 10
```
