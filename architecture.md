# Tinyfin-RL v2 Architecture

Tinyfin-RL is a C-first reinforcement learning system built around a single
CLI binary, a deterministic environment API, optional render snapshots, and
Tinyfin as the tensor/autograd backend.

This document describes the current v2 implementation state. It also calls out
where the code is still in transition.

## Layers

```text
CLI
  -> runner
    -> algorithm
    -> environment
    -> trace writer
    -> optional viewer

production runner
  -> checkpoint loader
  -> algorithm inference
  -> environment or client integration
```

The runner coordinates work. Environments own simulation state. Algorithms own
policy/update state. Rendering consumes snapshots and does not drive stepping.

## Public API

Public headers live in `include/tfrl/`:

- `tfrl/env.h`
- `tfrl/algo.h`
- `tfrl/runner.h`
- `tfrl/metrics.h`
- `tfrl/checkpoint.h`
- `tfrl/render_snapshot.h`
- `tfrl/tfrl.h`

The old headers under `src/core/` and `src/runner/` are compatibility shims for
internal source files. New code should include `tfrl/...`.

## Environment API

The core environment API is renderer-agnostic:

```c
tfrl_env *tfrl_env_create(const tfrl_env_config *cfg);
void tfrl_env_destroy(tfrl_env *env);
tfrl_obs tfrl_env_reset(tfrl_env *env, uint64_t seed);
tfrl_step_result tfrl_env_step(tfrl_env *env, tfrl_action action);
const tfrl_env_spec *tfrl_env_get_spec(const tfrl_env *env);
```

`tfrl_env_spec` describes observation/action spaces, shapes, dtypes, bounds,
and agent count.

Batch and multi-agent helpers are also part of the public API:

```c
int tfrl_env_reset_multi(tfrl_env *env, uint64_t seed, tfrl_obs *out_obs, int max_agents);
int tfrl_env_step_multi(tfrl_env *env, const tfrl_action *actions, int action_count,
                        tfrl_step_result *out_steps, int max_agents);
void tfrl_env_step_batch(tfrl_env **envs, int env_count,
                         const tfrl_action *actions,
                         tfrl_step_result *out_steps);
```

## Environment Modules

Current state:

- Environment implementations still live under `src/envs/`.
- The normal build compiles them into `tinyfin-rl`.
- `build/libtfrl_env.so` exposes the combined env library for bindings.
- Core env manifests exist under `envs/*/env.toml`.
- Each core env builds as a separate shared library.
- The runner can load envs statically by name or dynamically by library path.

Remaining target state:

- Adding an env should not require editing central registry code.

## Algorithm API

Algorithms share one C interface:

```c
tfrl_algo tfrl_algo_create(const tfrl_algo_config *cfg, const tfrl_env_spec *spec);
tfrl_action tfrl_algo_act(tfrl_algo *algo, tfrl_env *env, tfrl_obs obs);
void tfrl_algo_act_batch(tfrl_algo *algo, const tfrl_obs *obs, int count,
                         tfrl_action *out_actions);
void tfrl_algo_destroy(tfrl_algo *algo);
```

v2 exposes only:

- `dqn`
- `ppo`
- `nca`
- `random`

Removed v1 placeholder algorithm names fail fast. The current `dqn` and `ppo`
paths are compatibility baselines behind the v2 names. `nca` is implemented as
a trainable cellular policy with deterministic rollout and save/load support.
The next milestone is real Tinyfin-backed DQN/PPO and full Tinyfin autograd
optimization for NCA.

## Rendering

Rendering is snapshot-based:

```c
size_t tfrl_env_render_bytes_needed(const tfrl_env *env);
size_t tfrl_env_render_write(tfrl_env *env, void *buffer, size_t buffer_len);
```

Environments write snapshots. The viewer consumes snapshots. Raylib is optional
and only belongs to the viewer layer.

## Build Targets

Current Make targets:

- `build/tinyfin-rl`
- `build/tinyfin-prod`
- `build/libtfrl_env.so`
- smoke tests under `build/`

CMake targets mirror the same current shape where CMake is available. The
desired v2 target list adds per-env libraries and separate algorithm libraries.

## Production

`tinyfin-prod` is the current inference-oriented binary. The stable production
ABI and checkpoint bundle format are still planned work.
