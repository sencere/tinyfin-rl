#include <stdlib.h>

#include "tinyfin_rl/rl.h"
#include "tinyfin_rl/rl_env_info.h"
#include "tinyfin_rl/rl_plugin.h"

#define LINE_LEN 7
#define MAX_STEPS 30

typedef struct {
    int pos;
    int goal;
    int steps;
} line_env;

static int line_reset(void *ctx, tfrl_value *out_obs) {
    line_env *env = (line_env *)ctx;
    env->pos = 0;
    env->goal = LINE_LEN - 1;
    env->steps = 0;
    out_obs->type = TFRL_VALUE_I64;
    out_obs->as.i64 = env->pos;
    return TFRL_OK;
}

static int line_step(void *ctx, const tfrl_value *action, tfrl_step *out_step) {
    line_env *env = (line_env *)ctx;
    int act = (int)action->as.i64;
    if (act == 0 && env->pos > 0) {
        env->pos -= 1;
    } else if (act == 1 && env->pos < LINE_LEN - 1) {
        env->pos += 1;
    }
    env->steps += 1;
    int reached = env->pos == env->goal;

    out_step->observation.type = TFRL_VALUE_I64;
    out_step->observation.as.i64 = env->pos;
    out_step->reward = reached ? 1.0 : -0.01;
    out_step->terminated = reached;
    out_step->truncated = env->steps >= MAX_STEPS;
    return TFRL_OK;
}

static void line_destroy(void *ctx) {
    free(ctx);
}

static const tfrl_env_vtable LINE_VTABLE = {
    .reset = line_reset,
    .step = line_step,
    .seed = NULL,
    .destroy = line_destroy
};

static tfrl_env line_create(void) {
    line_env *env = (line_env *)calloc(1, sizeof(line_env));
    tfrl_env out = {env, &LINE_VTABLE};
    return out;
}

static void line_plugin_destroy(tfrl_env *env) {
    if (env && env->vtable && env->vtable->destroy) {
        env->vtable->destroy(env->ctx);
    }
}

static const tfrl_env_plugin LINE_PLUGIN = {
    .api_version = TFRL_ENV_PLUGIN_API_VERSION,
    .name = "lineworld",
    .create = line_create,
    .destroy = line_plugin_destroy
};

static const tfrl_env_metadata LINE_METADATA = {
    .api_version = TFRL_ENV_INFO_API_VERSION,
    .name = "lineworld",
    .observation_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .observation_n = LINE_LEN,
    .action_n = 2,
    .observation_dims = 1,
    .action_dims = 1,
    .observation_low = 0.0,
    .observation_high = (double)(LINE_LEN - 1),
    .action_low = 0.0,
    .action_high = 1.0
};

const tfrl_env_plugin *tfrl_env_plugin_get(void) {
    return &LINE_PLUGIN;
}

const tfrl_env_metadata *tfrl_env_metadata_get(void) {
    return &LINE_METADATA;
}
