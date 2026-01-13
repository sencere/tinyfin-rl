# DQN Walkthrough

Use the C trainer with a C env plugin:

```bash
cd tinyfin-rl
make train_dqn
./build/train_dqn --env environments/lineworld/liblineworld.so --obs-n 7 --action-n 2
./build/train_dqn --env environments/lineworld/liblineworld.so --obs-n 7 --action-n 2 --save runs/lineworld_dqn
./build/train_dqn --env environments/lineworld/liblineworld.so --obs-n 7 --action-n 2 --load runs/lineworld_dqn --eval-steps 500
make play_dqn
./build/play_dqn lineworld --load runs/lineworld_dqn --fps 10
```

Python can call the same C trainer via bindings (see `docs/python_bindings.md`).
