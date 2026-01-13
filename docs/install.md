# Install

## Requirements

- C compiler
- Tinyfin built in `tinyfin/`
- Optional: raylib (build from `raylib-src/` for viewer)

## Build

```bash
make
```

With raylib:

```bash
make USE_RAYLIB=1
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
