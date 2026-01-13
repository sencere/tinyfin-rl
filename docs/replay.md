# Replay

Replays are created by writing trace frames during training or evaluation.

## Record

```bash
./build/tinyfin-rl train --algo dqn --steps 2000 --trace-out runs/run.tft
```

## Replay

```bash
./build/tinyfin-rl replay --trace-in runs/run.tft --render-fps 10
```
