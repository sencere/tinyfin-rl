# Install

## Requirements

- C compiler
- Tinyfin built in `tinyfin/`
- Optional: raylib (build from `raylib-src/` for viewer)

## Build

```bash
make
```

With CUDA (Tinyfin backend):

```bash
make -C tinyfin ENABLE_CUDA=1 libtinyfin.so
make
```

Run with CUDA:

```bash
./build/tinyfin-rl train --algo dqn --backend cuda --device gpu
```

With raylib:

```bash
make USE_RAYLIB=1
```

Env shared library for Python bindings:

```bash
make
ls build/libtfrl_env.so
```

## CMake (optional)

```bash
cmake -S . -B build-cmake
cmake --build build-cmake
```

With raylib:

```bash
cmake -S . -B build-cmake -DTFRL_WITH_RAYLIB=ON
cmake --build build-cmake
```
