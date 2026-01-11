# tinyfin-rl lineworld (headless)

Minimal 1D lineworld using the tinyfin-rl C API. The agent starts at position 0 and moves left/right toward a goal.

## Build

```bash
cd tinyfin-rl/env/lineworld
make
```

## Run (random policy)

```bash
./lineworld --episodes 5 --seed 1234
```

## Train (Q-learning)

```bash
./train_q --episodes 2000 --report 200 --out q_table.csv --metrics train_metrics.csv
./train_q --eval --in q_table.csv --episodes 200 --metrics eval_metrics.json --metrics-format json
```

## Plugin build (shared library)

```bash
make liblineworld.so
```
