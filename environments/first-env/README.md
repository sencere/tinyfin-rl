# tinyfin-rl first-env (raylib)

This is a minimal raylib-driven gridworld using the tinyfin-rl C API.

## Build (raylib-quickstart)

The raylib quickstart submodule lives at `tinyfin-rl/raylib`.

Replace the quickstart `src/main.c` with this env and build via the quickstart flow:

```bash
cd tinyfin-rl/raylib
ln -sf ../environments/first-env/main.c src/main.c
cd build
./premake5 gmake2
cd ..
make
```

On Windows/macOS, use the quickstart batch files or premake binaries as described in `tinyfin-rl/raylib/README.md`.

## Build (Linux without premake)

If the bundled premake binary is too new for your glibc, build directly against raylib.

Option A: system raylib (pkg-config):

```bash
cd tinyfin-rl/environments/first-env
make
./first_env
```

Option B: local raylib source:

1) Use the `tinyfin-rl/raylib-src` submodule (or place the raylib source there).
2) Then run:

```bash
cd tinyfin-rl/environments/first-env
make RAYLIB_DIR=../../raylib-src
./first_env
```

## Renderer sample (C)

Build and run the C viewer that uses the renderer ABI:

```bash
cd tinyfin-rl/environments/first-env
make view
./view
```

## Renderer sample (dlopen)

Build the generic plugin viewer and run it with the env plugin + renderer:

```bash
cd tinyfin-rl/environments/common
make
TFRL_ENV_PLUGIN_LIB=../first-env/libfirst_env.so \
TFRL_RENDER_LIB=../first-env/libfirst_env_render.so \
./plugin_viewer
```

Short form (auto-detects libs + viewer.conf):

```bash
./plugin_viewer ../first-env
```

## Train (Q-learning)

This runs a small tabular Q-learning loop against the gridworld dynamics:

```bash
cd tinyfin-rl/environments/first-env
make train_q
./train_q --episodes 2000 --report 200 --out q_table.csv --metrics train_metrics.csv
./train_q --eval --in q_table.csv --episodes 200 --metrics eval_metrics.json --metrics-format json
```

Or run the helper script:

```bash
cd tinyfin-rl/environments/first-env
./run_train_view.sh
```

## Autoplay in raylib viewer

Train first to generate `q_table.csv`, then press `P` to toggle autoplay:

```bash
cd tinyfin-rl/environments/first-env
./first_env
```

## Plugin build (shared library)

Build a tiny env plugin that can be loaded via `examples/c/env_plugin_loader.c`:

```bash
cd tinyfin-rl/env/first-env
make libfirst_env.so
```

## Python bindings + raylib renderer

Build the env plugin and the raylib renderer shared library:

```bash
cd tinyfin-rl/env/first-env
make libfirst_env.so libfirst_env_render.so
```

Then load the plugin and renderer from Python (see `docs/python_bindings.md`).
