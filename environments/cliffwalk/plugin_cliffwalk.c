#include <stdlib.h>

#include "tinyfin_rl/rl.h"
#include "tinyfin_rl/rl_env_info.h"
#include "tinyfin_rl/rl_plugin.h"

#define CLIFF_W 12
#define CLIFF_H 4
#define MAX_STEPS 200

typedef struct {
    int x;
    int y;
    int steps;
} cliff_env;

static int cliff_reset(void *ctx, tfrl_value *out_obs) {
    cliff_env *env = (cliff_env *)ctx;
    env->x = 0;
    env->y = CLIFF_H - 1;
    env->steps = 0;
    out_obs->type = TFRL_VALUE_I64;
    out_obs->as.i64 = env->y * CLIFF_W + env->x;
    return TFRL_OK;
}

static int cliff_step(void *ctx, const tfrl_value *action, tfrl_step *out_step) {
    cliff_env *env = (cliff_env *)ctx;
    int act = (int)action->as.i64;
    int nx = env->x;
    int ny = env->y;
    if (act == 0) ny -= 1;
    else if (act == 1) ny += 1;
    else if (act == 2) nx -= 1;
    else if (act == 3) nx += 1;
    if (nx < 0) nx = 0;
    if (nx >= CLIFF_W) nx = CLIFF_W - 1;
    if (ny < 0) ny = 0;
    if (ny >= CLIFF_H) ny = CLIFF_H - 1;
    env->x = nx;
    env->y = ny;
    env->steps += 1;

    int reached = (env->x == CLIFF_W - 1 && env->y == CLIFF_H - 1);
    int cliff = (env->y == CLIFF_H - 1 && env->x > 0 && env->x < CLIFF_W - 1);
    double reward = -1.0;
    int terminated = reached || cliff;

    out_step->observation.type = TFRL_VALUE_I64;
    out_step->observation.as.i64 = env->y * CLIFF_W + env->x;
    out_step->reward = cliff ? -100.0 : reward;
    out_step->terminated = terminated;
    out_step->truncated = env->steps >= MAX_STEPS;
    if (cliff) {
        env->x = 0;
        env->y = CLIFF_H - 1;
    }
    return TFRL_OK;
}

static void cliff_destroy(void *ctx) {
    free(ctx);
}

static const tfrl_env_vtable CLIFF_VTABLE = {
    .reset = cliff_reset,
    .step = cliff_step,
    .seed = NULL,
    .destroy = cliff_destroy
};

static tfrl_env cliff_create(void) {
    cliff_env *env = (cliff_env *)calloc(1, sizeof(cliff_env));
    tfrl_env out = {env, &CLIFF_VTABLE};
    return out;
}

static void cliff_plugin_destroy(tfrl_env *env) {
    if (env && env->vtable && env->vtable->destroy) {
        env->vtable->destroy(env->ctx);
    }
}

static const tfrl_env_plugin CLIFF_PLUGIN = {
    .api_version = TFRL_ENV_PLUGIN_API_VERSION,
    .name = "cliffwalk",
    .create = cliff_create,
    .destroy = cliff_plugin_destroy
};

static const tfrl_env_metadata CLIFF_METADATA = {
    .api_version = TFRL_ENV_INFO_API_VERSION,
    .name = "cliffwalk",
    .observation_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .observation_n = CLIFF_W * CLIFF_H,
    .action_n = 4,
    .observation_dims = 1,
    .action_dims = 1,
    .observation_low = 0.0,
    .observation_high = (double)(CLIFF_W * CLIFF_H - 1),
    .action_low = 0.0,
    .action_high = 3.0
};

const tfrl_env_plugin *tfrl_env_plugin_get(void) {
    return &CLIFF_PLUGIN;
}

const tfrl_env_metadata *tfrl_env_metadata_get(void) {
    return &CLIFF_METADATA;
}
