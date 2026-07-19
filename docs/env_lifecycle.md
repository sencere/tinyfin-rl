# Environment Lifecycle

This page shows the current v2 workflow for creating an environment, compiling
it, training on it, evaluating it, and using it from the production runner.

## 1. Create An Env

Today, C env implementations still live under `src/envs/`. Add the env logic in
a dedicated folder:

```text
src/envs/my_env/my_env.c
src/envs/my_env/render.c
```

Implement the standard env entry points:

```c
const tfrl_env_spec *tfrl_env_spec_my_env(void);
tfrl_obs tfrl_env_reset_my_env(tfrl_env *env, uint64_t seed);
tfrl_step_result tfrl_env_step_my_env(tfrl_env *env, tfrl_action action);
size_t tfrl_env_render_bytes_my_env(const tfrl_env *env);
size_t tfrl_env_render_write_my_env(tfrl_env *env, void *buffer, size_t buffer_len);
```

For the current registry-backed path, also:

- Add declarations to `src/envs/envs_internal.h`.
- Add the env ops row to `src/envs/registry.c`.
- Add the source files to `ENV_SRCS` in `Makefile`.
- Add the source files to `SRC_ENVS` in `CMakeLists.txt`.

## 2. Add A Manifest

Create `envs/my_env/env.toml`:

```toml
name = "my_env"
kind = "grid"
library = "libtfrl_env_my_env.so"
obs_type = "discrete"
action_type = "discrete"
agents = 1
width = 8
height = 8
max_steps = 200
```

Manifest loading currently uses `name` and `library`. The other fields document
the contract and are reserved for fuller discovery/validation.

## 3. Compile

Build the normal CLI, combined env library, and separated core env modules:

```bash
make
```

Existing separated core env modules are emitted under `build/`:

```bash
build/libtfrl_env_lineworld.so
build/libtfrl_env_maze_rooms.so
build/libtfrl_env_coin_maze.so
```

For a new env module, add a Make target like:

```make
ENV_MODULE_MY_ENV_SRCS = \
	$(ENV_MODULE_CORE_SRCS) \
	src/envs/my_env/my_env.c \
	src/envs/my_env/render.c

ENV_LIB_MY_ENV ?= $(BIN_DIR)/libtfrl_env_my_env.so

$(ENV_LIB_MY_ENV): $(ENV_MODULE_MY_ENV_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -DTFRL_ENV_MODULE_MY_ENV -DTFRL_ENV_NO_PYBRIDGE -DTFRL_ENV_NO_DYNAMIC -fPIC -shared -Isrc -o $@ $^
```

Then add the module define to `src/envs/registry.c` so that module build
contains only that env.

## 4. Train

Train with a built-in registry name:

```bash
./build/tinyfin-rl train --algo dqn --env my_env --steps 2000 --save runs/my_env.dqn
```

Train with a separated env module path:

```bash
./build/tinyfin-rl train --algo dqn --env build/libtfrl_env_my_env.so --steps 2000 --save runs/my_env.dqn
```

Train with a manifest path:

```bash
./build/tinyfin-rl train --algo dqn --env envs/my_env/env.toml --steps 2000 --save runs/my_env.dqn
```

For deterministic smoke runs:

```bash
TFRL_DASHBOARD=0 ./build/tinyfin-rl train --algo random --env envs/my_env/env.toml --steps 20 --deterministic --seed 1
```

## 5. Evaluate

Evaluate a saved policy against the same env name, module path, or manifest
path used for training:

```bash
./build/tinyfin-rl eval --algo dqn --env envs/my_env/env.toml --load runs/my_env.dqn --episodes 10 --deterministic --seed 1
```

For NCA:

```bash
./build/tinyfin-rl train --algo nca --env envs/lineworld/env.toml --steps 2000 --save runs/lineworld.nca
./build/tinyfin-rl eval --algo nca --env envs/lineworld/env.toml --load runs/lineworld.nca --episodes 10 --deterministic
```

## 6. Production Runner

Compile the inference-oriented runner:

```bash
make build/tinyfin-prod
```

Run a saved checkpoint with a registry env:

```bash
./build/tinyfin-prod --algo dqn --env my_env --load runs/my_env.dqn --steps 1000 --deterministic
```

Run with a manifest-backed env module:

```bash
./build/tinyfin-prod --algo dqn --env envs/my_env/env.toml --load runs/my_env.dqn --steps 1000 --deterministic
```

Run with a direct env module path:

```bash
./build/tinyfin-prod --algo dqn --env build/libtfrl_env_my_env.so --load runs/my_env.dqn --steps 1000 --deterministic
```

The production runner is for inference loops. Training still happens through
`tinyfin-rl train`.

## 7. Validate

Run the standard checks:

```bash
./scripts/run_tests.sh
```

Current env-module smoke coverage verifies that separated core env modules can
be opened, created, reset, stepped, and rendered:

```bash
./build/test_env_modules
./build/test_env_dynamic_load
```

For a new env, add targeted tests for reset, step, terminal behavior, invalid
actions, determinism, render snapshots, and trace replay.
