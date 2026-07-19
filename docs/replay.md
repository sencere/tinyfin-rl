# Replay

Replays are created by writing trace frames during training or evaluation.

## Record

```bash
./build/tinyfin-rl train --algo dqn --steps 2000 --trace-out runs/run.tft
```

## Replay

```bash
./scripts/build.sh --with-raylib
./build/tinyfin-rl replay --trace-in runs/run.tft --render-fps 10
```

If the trace was recorded with metadata (v2 header), the viewer displays the
resolved config line at the bottom.

The replay CLI also prints the trace metadata to stdout when available.

Use `--dump-meta-json` to emit the metadata as a JSON object and exit.
