// Example environment template for Tinyfin-RL.
// This file is NOT compiled by default. Copy it into src/envs/envs.c or
// wire it into the build if you want to use it.

#include "core/env_api.h"

#include <stdlib.h>

#define EXAMPLE_W 10
#define EXAMPLE_MAX_STEPS 50

struct tfrl_env {
    int position;
    int steps;
};

static const tfrl_env_spec EXAMPLE_SPEC = {
    .name = "example_env",
    .obs_n = EXAMPLE_W,
    .action_n = 2,
    .max_steps = EXAMPLE_MAX_STEPS,
    .width = EXAMPLE_W,
    .height = 1,
    .obs_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {EXAMPLE_W, 0},
    .action_shape = {2, 0},
    .obs_dtype = TFRL_DTYPE_INT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = (double)(EXAMPLE_W - 1),
    .action_low = 0.0,
    .action_high = 1.0,
    .agent_count = 1,
};

tfrl_env *tfrl_env_create(const tfrl_env_config *cfg) {
    (void)cfg;
    tfrl_env *env = (tfrl_env *)calloc(1, sizeof(tfrl_env));
    if (!env) return NULL;
    env->position = 0;
    env->steps = 0;
    return env;
}

void tfrl_env_destroy(tfrl_env *env) {
    free(env);
}

tfrl_obs tfrl_env_reset(tfrl_env *env, uint64_t seed) {
    (void)seed;
    env->position = 0;
    env->steps = 0;
    tfrl_obs obs = {0};
    obs.index = env->position;
    return obs;
}

tfrl_step_result tfrl_env_step(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    if (act == 0 && env->position > 0) {
        env->position -= 1;
    } else if (act == 1 && env->position < EXAMPLE_W - 1) {
        env->position += 1;
    }
    env->steps += 1;

    int reached = (env->position == EXAMPLE_W - 1);
    int done = reached || (env->steps >= EXAMPLE_MAX_STEPS);

    tfrl_step_result out = {0};
    out.observation.index = env->position;
    out.reward = reached ? 1.0 : -0.01;
    out.done = done;
    return out;
}

const tfrl_env_spec *tfrl_env_get_spec(const tfrl_env *env) {
    (void)env;
    return &EXAMPLE_SPEC;
}

int tfrl_env_agent_count(const tfrl_env *env) {
    (void)env;
    return 1;
}

int tfrl_env_reset_multi(tfrl_env *env, uint64_t seed, tfrl_obs *out_obs, int max_agents) {
    if (!env || !out_obs || max_agents <= 0) return 0;
    out_obs[0] = tfrl_env_reset(env, seed);
    return 1;
}

int tfrl_env_step_multi(tfrl_env *env, const tfrl_action *actions, int action_count, tfrl_step_result *out_steps, int max_agents) {
    if (!env || !actions || !out_steps || max_agents <= 0 || action_count <= 0) return 0;
    out_steps[0] = tfrl_env_step(env, actions[0]);
    return 1;
}

void tfrl_env_step_batch(tfrl_env **envs, int env_count, const tfrl_action *actions, tfrl_step_result *out_steps) {
    if (!envs || !actions || !out_steps || env_count <= 0) return;
    for (int i = 0; i < env_count; i++) {
        out_steps[i] = tfrl_env_step(envs[i], actions[i]);
    }
}

size_t tfrl_env_render_bytes_needed(const tfrl_env *env) {
    (void)env;
    return 0;
}

size_t tfrl_env_render_write(tfrl_env *env, void *buffer, size_t buffer_len) {
    (void)env;
    (void)buffer;
    (void)buffer_len;
    return 0;
}
