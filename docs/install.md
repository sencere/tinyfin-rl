# Installation

Tinyfin-RL is intended to be embedded into a C-first project with a tensor backend. At the moment, this repository contains Python scaffolding, but it is not the long-term deployment target.

## Local usage

Use the Python modules as temporary scaffolding while the C modules are built. The examples are placeholders and will be replaced by C-based samples.

## Optional dependencies

- Raylib visualization is intended for C integration.
- Gymnasium is only used by temporary Python stubs.

## Tinyfin C backend

Tinyfin lives in a separate repository. When you have a shared library, point to it with:

```bash
export TINYFIN_LIB=/path/to/libtinyfin.so
```

Then use `CBackend` to load symbols.

This repository also includes a minimal embedded snapshot in `csrc/tinyfin/` that can be built directly into tinyfin-rl when the C build system is added.
