# tinyfin-rl lineworld (headless)

Minimal 1D lineworld using the tinyfin-rl C API. The agent starts at position 0 and moves left/right toward a goal.

## Build

```bash
cd tinyfin-rl/environments/lineworld
make
```

## Run (random policy)

```bash
./lineworld --episodes 5 --seed 1234
```

## Renderer sample (C)

```bash
make view
./view
```

If raylib is not installed system-wide, the build uses `raylib-src`:

```bash
make RAYLIB_DIR=../../raylib-src
```

## Renderer sample (dlopen)

```bash
cd tinyfin-rl/environments/common
make
TFRL_ENV_PLUGIN_LIB=../lineworld/liblineworld.so \
TFRL_RENDER_LIB=../lineworld/liblineworld_render.so \
TFRL_RENDER_GRID_SIZE=7 \
TFRL_RENDER_CELL_SIZE=64 \
./plugin_viewer
```

Short form (auto-detects libs + viewer.conf):

```bash
./plugin_viewer ../lineworld
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

## Renderer build (shared library)

```bash
make liblineworld_render.so
```
