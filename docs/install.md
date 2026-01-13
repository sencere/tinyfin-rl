# Installation

Tinyfin-RL is intended to be embedded into a C-first project with a tensor backend. At the moment, this repository contains Python scaffolding, but it is not the long-term deployment target.

## Local usage

Python bindings load C environments through the plugin ABI; C envs remain the source of truth.

## Optional dependencies

- Raylib visualization is intended for C integration (shared library via `raylib-src` or system raylib).
- Gymnasium is only used by temporary Python stubs.

## Tinyfin C backend

Tinyfin lives in a separate repository. When you have a shared library, point to it with:

```bash
export TINYFIN_LIB=/path/to/libtinyfin.so
```

Then use `CBackend` to load symbols.

This repository includes the Tinyfin submodule in `tinyfin/` for local builds when the C build system is added.

## Raylib shared library (Linux)

If you are using the bundled `raylib-src`, build the shared library:

```bash
cd tinyfin-rl/raylib-src/src
make RAYLIB_LIBTYPE=SHARED
```

Then build the environment renderers (example):

```bash
cd tinyfin-rl/environments/first-env
make libfirst_env_render.so
```

## C trainer library

Build the Tinyfin-backed DQN/PPO trainers:

```bash
cd tinyfin-rl
make tfrl_train
```
