message for the agent: We could introduce a lib folder and pack it with raylib and tinyfin if this is possible. Then we dont need to keep a sub-repo


Yes. On Linux, raylib can be built and used as a shared object, typically `libraylib.so`. The official build system supports both shared and static libraries. ([GitHub][1])

### Build with CMake

```bash
git clone https://github.com/raysan5/raylib.git
cd raylib

cmake -S . -B build \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build
```

This produces a versioned shared library and symlinks similar to:

```text
libraylib.so
libraylib.so.6
libraylib.so.6.0.0
```

Raylib’s CMake configuration sets the library version, SONAME and position-independent code when shared-library building is enabled. ([GitHub][2])

You can also use raylib’s Makefile:

```bash
cd raylib/src
make clean
make RAYLIB_LIBTYPE=SHARED
```

The Makefile explicitly supports `RAYLIB_LIBTYPE=SHARED` and generates `libraylib.so` symlinks on Linux. ([GitHub][1])

### Link your program against it

```bash
gcc main.c \
  -I/path/to/raylib/src \
  -L/path/to/raylib/src \
  -lraylib \
  -o game
```

At runtime, the dynamic linker must be able to locate the `.so`. For testing:

```bash
LD_LIBRARY_PATH=/path/to/raylib/src ./game
```

Or embed a relative runtime path and place `libraylib.so` beside the executable:

```bash
gcc main.c \
  -I/path/to/raylib/src \
  -L/path/to/raylib/src \
  -Wl,-rpath,'$ORIGIN' \
  -lraylib \
  -o game
```

Then distribute something like:

```text
game
libraylib.so
libraylib.so.6
libraylib.so.6.0.0
```

Keep the versioned target and symlinks intact, because the executable may depend on the library’s SONAME rather than the unversioned `libraylib.so` filename. Raylib’s own Makefile also notes that runtime discovery can be handled through RPATH, `LD_LIBRARY_PATH`, or `ldconfig`. ([GitHub][1])

[1]: https://github.com/raysan5/raylib/blob/master/src/Makefile "raylib/src/Makefile at master · raysan5/raylib · GitHub"
[2]: https://github.com/raysan5/raylib/blob/master/src/CMakeLists.txt "raylib/src/CMakeLists.txt at master · raysan5/raylib · GitHub"

