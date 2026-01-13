# Tinyfin-RL Architecture

Tinyfin-RL is a **C-first reinforcement learning system** built around a single
executable, a deterministic environment core, and Tinyfin as the tensor +
autograd backend. Rendering is optional and never drives simulation.

## Design Goals

- One executable (`tinyfin-rl`)
- One control loop (train / eval / replay)
- Environment is the source of truth
- Algorithms are configurable (`--algo dqn|ppo|reinforce`)
- Tinyfin handles math; Tinyfin-RL handles control
- Rendering is optional and isolated

## Core Runtime Flow

```
CLI -> Runner -> Env + Algo -> (optional) Trace + Viewer
```

The runner owns simulation and decides if training, evaluation, or replay is
executed.

## Environment API

Minimal C API for a deterministic, renderer-agnostic environment:

```c
tfrl_env *tfrl_env_create(const tfrl_env_config *cfg);
tfrl_obs tfrl_env_reset(tfrl_env *env, uint64_t seed);
tfrl_step_result tfrl_env_step(tfrl_env *env, tfrl_action action);
const tfrl_env_spec *tfrl_env_get_spec(const tfrl_env *env);
```

`tfrl_env_spec` includes space types, shapes, dtypes, and bounds for both
observations and actions.

Rendering uses snapshots:

```c
size_t tfrl_env_render_bytes_needed(const tfrl_env *env);
size_t tfrl_env_render_write(tfrl_env *env, void *buffer, size_t buffer_len);
```

The environment **does not include raylib** and never opens windows.

## Algorithm API

Algorithms are small C modules chosen at runtime:

```c
tfrl_algo tfrl_algo_create(const tfrl_algo_config *cfg, const tfrl_env_spec *spec);
tfrl_action algo_act(tfrl_obs obs);
void algo_act_batch(const tfrl_obs *obs, int count, tfrl_action *out_actions);
void algo_update(const tfrl_transition *transition);
void algo_save(const char *path);
```

Algorithms may use Tinyfin for models and optimization. DQN, REINFORCE, and PPO
use Tinyfin `Linear` layers with SGD/Adam.

## Rendering

Rendering is a viewer module that consumes render snapshots and owns raylib.
It is optional and only compiled when `USE_RAYLIB=1` (Makefile) or
`-DTFRL_WITH_RAYLIB=ON` (CMake).

## Vectorized Stepping

The environment API includes a batch stepping helper:

```c
void tfrl_env_step_batch(tfrl_env **envs, int env_count,
                         const tfrl_action *actions,
                         tfrl_step_result *out_steps);
```

Algorithms may implement `act_batch` for batched inference (DQN does).

## Trace Replay

Traces are a sequence of render snapshots. `tinyfin-rl replay` reads a trace
file and feeds frames to the viewer.

## Repository Layout

```
tinyfin-rl/
  src/
    main.c
    cli/
    runner/
    core/
      env_api.h
      env_example.c
      algo_api.h
      algo_dqn.c
      trace.c
      render_snapshot.h
    viewer/
      viewer.h
      viewer_raylib.c
      viewer_stub.c
  tinyfin/         # tensor + autograd
  raylib-src/      # optional raylib
  roadmap.md
```

## Current Focus

- One binary running train/eval/replay
- One canonical environment (`maze_rooms`) + one small validation env (`lineworld`)
- DQN/REINFORCE/PPO implemented in C with Tinyfin
- Render snapshots + optional raylib viewer
