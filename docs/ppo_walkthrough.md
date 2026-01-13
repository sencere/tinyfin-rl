# PPO Walkthrough

Use the C trainer with a C env plugin:

```bash
cd tinyfin-rl
make train_ppo
./build/train_ppo --env environments/first-env/libfirst_env.so --obs-n 100 --action-n 4
./build/train_ppo --env environments/first-env/libfirst_env.so --obs-n 100 --action-n 4 --save runs/first_env_ppo
./build/train_ppo --env environments/first-env/libfirst_env.so --obs-n 100 --action-n 4 --load runs/first_env_ppo --eval-steps 500
```

Python can call the same C trainer via bindings (see `docs/python_bindings.md`).
