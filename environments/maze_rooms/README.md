# maze_rooms (C env)

Grid navigation with four rooms and doorways. Sparse reward at the goal.

## Build

```bash
cd tinyfin-rl/environments/maze_rooms
make
```

## Run viewer

```bash
./maze_rooms
```

## Train (C trainers)

```bash
cd tinyfin-rl
make train_dqn train_ppo
./build/train_dqn --env environments/maze_rooms/libmaze_rooms.so --obs-n 100 --action-n 4
./build/train_ppo --env environments/maze_rooms/libmaze_rooms.so --obs-n 100 --action-n 4
```

## Renderer sample (dlopen)

```bash
cd tinyfin-rl/environments/common
make
./plugin_viewer ../maze_rooms
```
