#include <stdlib.h>

#include "tinyfin_rl/rl.h"
#include "tinyfin_rl/rl_env_info.h"
#include "tinyfin_rl/rl_plugin.h"

#define SOCCER_W 8
#define SOCCER_H 6
#define MAX_STEPS 200

typedef struct {
    int ax;
    int ay;
    int bx;
    int by;
    int steps;
} soccer_env;

static int encode_obs(const soccer_env *env) {
    int a = env->ay * SOCCER_W + env->ax;
    int b = env->by * SOCCER_W + env->bx;
    return b * (SOCCER_W * SOCCER_H) + a;
}

static void reset_positions(soccer_env *env) {
    env->ax = 1;
    env->ay = SOCCER_H / 2;
    env->bx = SOCCER_W / 2;
    env->by = SOCCER_H / 2;
    env->steps = 0;
}

static int soccer_reset(void *ctx, tfrl_value *out_obs) {
    soccer_env *env = (soccer_env *)ctx;
    reset_positions(env);
    out_obs->type = TFRL_VALUE_I64;
    out_obs->as.i64 = encode_obs(env);
    return TFRL_OK;
}

static int soccer_step(void *ctx, const tfrl_value *action, tfrl_step *out_step) {
    soccer_env *env = (soccer_env *)ctx;
    int act = (int)action->as.i64;
    int dx = 0, dy = 0;
    if (act == 0) dy = -1;
    else if (act == 1) dy = 1;
    else if (act == 2) dx = -1;
    else if (act == 3) dx = 1;

    int nax = env->ax + dx;
    int nay = env->ay + dy;
    if (nax < 0) nax = 0;
    if (nax >= SOCCER_W) nax = SOCCER_W - 1;
    if (nay < 0) nay = 0;
    if (nay >= SOCCER_H) nay = SOCCER_H - 1;

    if (nax == env->bx && nay == env->by) {
        int nbx = env->bx + dx;
        int nby = env->by + dy;
        if (nbx >= 0 && nbx < SOCCER_W && nby >= 0 && nby < SOCCER_H) {
            env->bx = nbx;
            env->by = nby;
            env->ax = nax;
            env->ay = nay;
        }
    } else {
        env->ax = nax;
        env->ay = nay;
    }

    env->steps += 1;
    int goal_x = SOCCER_W - 1;
    int goal_y = SOCCER_H / 2;
    int scored = (env->bx == goal_x && env->by == goal_y);

    out_step->observation.type = TFRL_VALUE_I64;
    out_step->observation.as.i64 = encode_obs(env);
    out_step->reward = scored ? 1.0 : -0.01;
    out_step->terminated = scored;
    out_step->truncated = env->steps >= MAX_STEPS;
    return TFRL_OK;
}

static void soccer_destroy(void *ctx) {
    free(ctx);
}

static const tfrl_env_vtable SOCCER_VTABLE = {
    .reset = soccer_reset,
    .step = soccer_step,
    .seed = NULL,
    .destroy = soccer_destroy
};

static tfrl_env soccer_create(void) {
    soccer_env *env = (soccer_env *)calloc(1, sizeof(soccer_env));
    tfrl_env out = {env, &SOCCER_VTABLE};
    return out;
}

static void soccer_plugin_destroy(tfrl_env *env) {
    if (env && env->vtable && env->vtable->destroy) {
        env->vtable->destroy(env->ctx);
    }
}

static const tfrl_env_plugin SOCCER_PLUGIN = {
    .api_version = TFRL_ENV_PLUGIN_API_VERSION,
    .name = "grid_soccer",
    .create = soccer_create,
    .destroy = soccer_plugin_destroy
};

static const tfrl_env_metadata SOCCER_METADATA = {
    .api_version = TFRL_ENV_INFO_API_VERSION,
    .name = "grid_soccer",
    .observation_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .observation_n = (SOCCER_W * SOCCER_H) * (SOCCER_W * SOCCER_H),
    .action_n = 4,
    .observation_dims = 1,
    .action_dims = 1,
    .observation_low = 0.0,
    .observation_high = (double)((SOCCER_W * SOCCER_H) * (SOCCER_W * SOCCER_H) - 1),
    .action_low = 0.0,
    .action_high = 3.0
};

const tfrl_env_plugin *tfrl_env_plugin_get(void) {
    return &SOCCER_PLUGIN;
}

const tfrl_env_metadata *tfrl_env_metadata_get(void) {
    return &SOCCER_METADATA;
}
