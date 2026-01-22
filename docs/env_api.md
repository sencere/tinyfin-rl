# Environment API

The environment API is a small C interface and is exported via `libtfrl_env.so`.

## Types

- `tfrl_obs` / `tfrl_action` (discrete indices or small fixed-size float vectors)
- `tfrl_step_result` (observation, reward, done)
- `tfrl_env_spec` (spaces, shapes, dtypes, bounds)
- `tfrl_env_spec.agent_count` (number of agents in the env)
- `tfrl_obs_layout` / `tfrl_obs_field` (optional layout for structured observations)

## Example

```c
tfrl_env_config cfg = {.name = "maze_rooms", .seed = 0};
tfrl_env *env = tfrl_env_create(&cfg);
tfrl_obs obs = tfrl_env_reset(env, 0);
tfrl_action action = {.index = 0};
tfrl_step_result step = tfrl_env_step(env, action);
```

For box spaces, use `data_len` + `data[]` (up to `TFRL_MAX_BOX_DIMS`):

```c
tfrl_action action = {0};
action.data_len = 1;
action.data[0] = 0.25f;
```

## Observation Layout (Optional)

You can describe structured observations as a set of fields that are flattened
into a single 1D box. The layout lives in `tfrl_env_spec.obs_layout` and can be
used to pack or unpack fields consistently.

```c
static const tfrl_obs_field MY_FIELDS[] = {
    {.name = "pos", .len = 2, .dims = 1, .shape = {2, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "vel", .len = 2, .dims = 1, .shape = {2, 0}, .dtype = TFRL_DTYPE_FLOAT32},
};

static const tfrl_obs_layout MY_LAYOUT = {
    .field_count = 2,
    .fields = MY_FIELDS,
};

static const tfrl_env_spec MY_SPEC = {
    /* ... */
    .obs_type = TFRL_SPACE_BOX,
    .obs_dims = 4,
    .obs_layout = &MY_LAYOUT,
};

float pos[2] = {x, y};
float vel[2] = {vx, vy};
const float *fields[] = {pos, vel};
tfrl_obs_flatten(&MY_LAYOUT, fields, &obs);
```

Multi-agent helpers:

```c
tfrl_obs obs_multi[2];
tfrl_action actions[2] = {{.index = 1}, {.index = 1}};
tfrl_step_result steps[2];
int agents = tfrl_env_reset_multi(env, 0, obs_multi, 2);
int got = tfrl_env_step_multi(env, actions, agents, steps, 2);
```

Batch helpers (single-agent env arrays):

```c
tfrl_env_reset_batch(envs, env_count, 0, obs);
tfrl_env_step_batch(envs, env_count, actions, steps);
```

Batch reset with explicit seeds (for sparse resets):

```c
tfrl_env_reset_batch_seeds(envs, reset_count, seeds, obs);
```

## Environments

- `maze_rooms`
- `lineworld`
- `lineworld_duo`
- `lineworld_cont`
- `point1d`
- `coin_maze`
- `coin_maze_duo`
- `snake`
- `floppy`
- `tetris`
- `arkanoid`
- `pang`
- `py:gymnasium:ENV_ID` (Python bridge)
- `py:pettingzoo:MODULE` (Python bridge)
- `py:retro:ENV_ID` (Python bridge)

## Adding a New C Env

1) Add a new file under `src/envs/` that implements:
   - `tfrl_env_spec_*`
   - `tfrl_env_reset_*`
   - `tfrl_env_step_*`
2) Register it in `src/envs/registry.c` with its name, ops, and agent count.
3) Add the new file to the build lists in `CMakeLists.txt` and `Makefile`.

`tfrl_env_create` no longer needs per-env branches; it uses the registry.

## Python

Python can load the shared library:

```python
from python.env import EnvLib

env = EnvLib().create("maze_rooms", seed=0)
print(env.spec())
obs = env.reset()
obs, reward, done = env.step(0)
env.close()
```
