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
