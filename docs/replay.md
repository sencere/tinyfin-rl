# Replay

Replays are created by writing trace frames during training or evaluation.

## Record

```bash
./build/tinyfin-rl train --algo dqn --steps 2000 --trace-out runs/run.tft
```

## Replay

```bash
make -C raylib-src/src
make clean
make USE_RAYLIB=1
export LD_LIBRARY_PATH="$PWD/raylib-src/src:$PWD/tinyfin:$LD_LIBRARY_PATH"
./build/tinyfin-rl replay --trace-in runs/run.tft --render-fps 10
```

If the trace was recorded with metadata (v2 header), the viewer displays the
resolved config line at the bottom.
