# Training (C-first)

This repo uses C-first trainers backed by Tinyfin. Build the trainer library and run DQN/PPO against C environments.

## Build trainers

```bash
cd tinyfin-rl
make tfrl_train train_dqn train_ppo
```

## Train on C environments

```bash
./build/train_dqn --env environments/lineworld/liblineworld.so --obs-n 7 --action-n 2
./build/train_ppo --env environments/first-env/libfirst_env.so --obs-n 100 --action-n 4
./build/train_dqn --env environments/maze_rooms/libmaze_rooms.so --obs-n 100 --action-n 4
./build/train_dqn --env environments/cliffwalk/libcliffwalk.so --obs-n 48 --action-n 4
./build/train_dqn --env environments/grid_soccer/libgrid_soccer.so --obs-n 2304 --action-n 4
```

## Save/load and evaluation

```bash
./build/train_dqn --env environments/lineworld/liblineworld.so --obs-n 7 --action-n 2 --save runs/lineworld_dqn
./build/train_dqn --env environments/lineworld/liblineworld.so --obs-n 7 --action-n 2 --load runs/lineworld_dqn --eval-steps 500

./build/train_ppo --env environments/first-env/libfirst_env.so --obs-n 100 --action-n 4 --save runs/first_env_ppo
./build/train_ppo --env environments/first-env/libfirst_env.so --obs-n 100 --action-n 4 --load runs/first_env_ppo --eval-steps 500
```

Evaluation always runs after training. If you want eval-only, keep updates/steps minimal and use `--load` + `--eval-steps`.

## Render a trained DQN

```bash
make play_dqn
./build/play_dqn maze_rooms --load runs/maze_dqn --fps 10
```

## Notes

- `cartpole_simple` uses `Box` observations. The current C trainers assume discrete obs, so discretize first or use the Gymnasium Python examples.
- Use env metadata when available (`tfrl_env_metadata_get`) to avoid hardcoding `obs_n`/`action_n`.
