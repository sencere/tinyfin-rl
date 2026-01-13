# grid_soccer (C env)

Single-agent grid soccer: push the ball into the goal.

## Build

```bash
cd tinyfin-rl/environments/grid_soccer
make
```

## Run viewer

```bash
./grid_soccer
```

## Train (C trainers)

```bash
cd tinyfin-rl
make train_dqn train_ppo
./build/train_dqn --env environments/grid_soccer/libgrid_soccer.so --obs-n 2304 --action-n 4
./build/train_ppo --env environments/grid_soccer/libgrid_soccer.so --obs-n 2304 --action-n 4
```

## Renderer sample (dlopen)

```bash
cd tinyfin-rl/environments/common
make
./plugin_viewer ../grid_soccer
```
