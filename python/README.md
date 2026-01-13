# Python (Optional)

Python does not own the environment loop. It only calls the C binary.

```bash
python3 -m python.train
```

You can override the binary path:

```bash
TFRL_BIN=build/tinyfin-rl python3 -m python.train
```

For checkpoints, pass `save_path` or `load_path` in `TrainConfig`.
