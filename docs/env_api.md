# Environment API

The environment API is a small C interface and is exported via `libtfrl_env.so`.

## Types

- `tfrl_obs` / `tfrl_action` (discrete indices or small fixed-size float vectors)
- `tfrl_step_result` (observation, reward, done)
- `tfrl_env_spec` (spaces, shapes, dtypes, bounds)
- `tfrl_env_spec.agent_count` (number of agents in the env)

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

Multi-agent helpers:

```c
tfrl_obs obs_multi[2];
tfrl_action actions[2] = {{.index = 1}, {.index = 1}};
tfrl_step_result steps[2];
int agents = tfrl_env_reset_multi(env, 0, obs_multi, 2);
int got = tfrl_env_step_multi(env, actions, agents, steps, 2);
```

## Environments

- `maze_rooms`
- `lineworld`
- `lineworld_duo`
- `lineworld_cont`
- `point1d`
- `coin_maze`
- `coin_maze_duo`

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
