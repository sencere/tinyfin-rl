# C API Overview

This document describes the minimal C API surface in `tinyfin_rl/rl.h`.

## Core types

- `tfrl_env`: environment instance backed by a vtable.
- `tfrl_agent`: policy/agent instance backed by a vtable.
- `tfrl_trainer`: trainer instance backed by a vtable.
- `tfrl_value`: scalar or buffer payload for observations/actions.
- `tfrl_step`: step result with `reward`, `terminated`, `truncated`.
- `tfrl_transition`: transition record passed to agents.

## Vtables

Each interface is modeled as a vtable with a context pointer:

- `tfrl_env_vtable`: `reset`, `step`, optional `seed`, optional `destroy`
- `tfrl_agent_vtable`: `act`, optional `observe`, optional `destroy`
- `tfrl_trainer_vtable`: `train`, optional `destroy`

All vtable functions return a `tfrl_status`.

## Minimal usage

```c
#include "tinyfin_rl/rl.h"

typedef struct {
    int state;
} my_env;

static int my_reset(void *ctx, tfrl_value *out_obs) {
    my_env *env = (my_env *)ctx;
    env->state = 0;
    out_obs->type = TFRL_VALUE_I64;
    out_obs->as.i64 = env->state;
    return TFRL_OK;
}

static int my_step(void *ctx, const tfrl_value *action, tfrl_step *out_step) {
    my_env *env = (my_env *)ctx;
    env->state += (int)action->as.i64;
    out_step->observation.type = TFRL_VALUE_I64;
    out_step->observation.as.i64 = env->state;
    out_step->reward = 1.0;
    out_step->terminated = 0;
    out_step->truncated = 0;
    return TFRL_OK;
}

int main(void) {
    my_env env_state = {0};
    const tfrl_env_vtable env_vtable = {
        .reset = my_reset,
        .step = my_step,
        .seed = NULL,
        .destroy = NULL
    };
    tfrl_env env = {&env_state, &env_vtable};
    tfrl_value obs = {0};
    tfrl_env_reset(&env, &obs);
    return 0;
}
```

See `environments/first-env/view.c` and `environments/lineworld/view.c` for renderer-driven examples.

## plugin_viewer

`environments/common/plugin_viewer` loads an env plugin + renderer pair and steps with random actions.
You can pass an env directory name (uses `viewer.conf`) or explicit shared library paths.

```bash
cd environments/common
make
./plugin_viewer lineworld --fps 10
```

`--fps` (or `TFRL_RENDER_FPS` / `viewer.conf: fps`) throttles the render loop.

## Env plugins

`tinyfin_rl/rl_plugin.h` defines a minimal plugin interface for environments. A plugin exposes
`tfrl_env_plugin_get()` which returns a struct with `create()` and `destroy()` hooks.

See `environments/first-env/plugin_grid.c` for a minimal plugin implementation.

## Renderer ABI

`tinyfin_rl/rl_render.h` defines the renderer ABI used by C-only raylib renderers. Renderers receive a `tfrl_value` so they can handle discrete or buffer observations.

## Trainer ABI

`tinyfin_rl/rl_train.h` defines C-first DQN/PPO trainers backed by Tinyfin.

## Render snapshots (planned)

`tinyfin_rl/rl_render_snapshot.h` defines an optional, renderer-agnostic snapshot ABI.

## Env metadata

`tinyfin_rl/rl_env_info.h` describes optional metadata (`tfrl_env_metadata_get`) for action/observation space introspection.
