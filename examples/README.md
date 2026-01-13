# Examples (C-first)

These examples use the C trainer ABI and environment plugins. Python examples use Gymnasium to illustrate algorithm behavior.

## Build

```bash
cd tinyfin-rl
make train_dqn train_ppo
```

## Run with env plugins

```bash
./build/train_dqn --env environments/lineworld/liblineworld.so --obs-n 7 --action-n 2
./build/train_ppo --env environments/first-env/libfirst_env.so --obs-n 100 --action-n 4
```

Additional C envs:

```bash
./build/train_dqn --env environments/maze_rooms/libmaze_rooms.so --obs-n 100 --action-n 4
./build/train_dqn --env environments/cliffwalk/libcliffwalk.so --obs-n 48 --action-n 4
./build/train_dqn --env environments/grid_soccer/libgrid_soccer.so --obs-n 2304 --action-n 4
```

## Python + Gymnasium (optional)

```bash
python examples/python/dqn_cartpole.py
python examples/python/ppo_cartpole.py
```

Python calling C trainers:

```bash
python examples/python/train_c_dqn.py
```
