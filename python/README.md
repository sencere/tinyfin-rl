# Python (Optional)

Python does not own the environment loop. It only calls the C binary.

For Gymnasium/PettingZoo/Retro, use the Python bridge in `docs/python_bridge.md`.

```bash
python3 -m python.train
```

You can override the binary path:

```bash
TFRL_BIN=build/tinyfin-rl python3 -m python.train
```

For checkpoints, pass `save_path` or `load_path` in `TrainConfig`.

## Env bindings

Build the shared env library and use `python/env.py`:

```bash
make
python3 -c "from python.env import EnvLib; env = EnvLib().create('maze_rooms'); print(env.spec()); env.close()"
```
