# Install

## Requirements

- C compiler
- Tinyfin built in `tinyfin/`
- Optional: raylib (build from `raylib-src/` for viewer)

## Build

```bash
./scripts/build.sh
```

With CUDA (Tinyfin backend):

```bash
./scripts/build.sh --backend cuda
```

Run with CUDA:

```bash
./build/tinyfin-rl train --algo dqn --backend cuda --device gpu
```

With raylib:

```bash
./scripts/build.sh --with-raylib
```

Set shell paths for interactive use:

```bash
. ./scripts/setup_env.sh
```

Env shared library for Python bindings:

```bash
make
ls build/libtfrl_env.so
```

Separated core env modules:

```bash
ls build/libtfrl_env_lineworld.so
ls build/libtfrl_env_maze_rooms.so
ls build/libtfrl_env_coin_maze.so
```

Run against a module path:

```bash
./build/tinyfin-rl train --algo random --env build/libtfrl_env_lineworld.so --steps 100
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
