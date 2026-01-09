#include <stdlib.h>

#include "tinyfin_rl/rl.h"
#include "tinyfin_rl/rl_plugin.h"

#define GRID_SIZE 10
#define MAX_STEPS 200

typedef struct {
    int x;
    int y;
    int steps;
    int goal_x;
    int goal_y;
} grid_env;

static int grid_reset(void *ctx, tfrl_value *out_obs) {
    grid_env *env = (grid_env *)ctx;
    env->x = 0;
    env->y = 0;
    env->steps = 0;
    env->goal_x = GRID_SIZE - 1;
    env->goal_y = GRID_SIZE - 1;
    out_obs->type = TFRL_VALUE_I64;
    out_obs->as.i64 = env->y * GRID_SIZE + env->x;
    return TFRL_OK;
}

static int grid_step(void *ctx, const tfrl_value *action, tfrl_step *out_step) {
    grid_env *env = (grid_env *)ctx;
    int act = (int)action->as.i64;
    if (act == 0 && env->y > 0) {
        env->y -= 1;
    } else if (act == 1 && env->y < GRID_SIZE - 1) {
        env->y += 1;
    } else if (act == 2 && env->x > 0) {
        env->x -= 1;
    } else if (act == 3 && env->x < GRID_SIZE - 1) {
        env->x += 1;
    }
    env->steps += 1;
    int reached = env->x == env->goal_x && env->y == env->goal_y;

    out_step->observation.type = TFRL_VALUE_I64;
    out_step->observation.as.i64 = env->y * GRID_SIZE + env->x;
    out_step->reward = reached ? 1.0 : -0.01;
    out_step->terminated = reached;
    out_step->truncated = env->steps >= MAX_STEPS;
    return TFRL_OK;
}

static void grid_destroy(void *ctx) {
    free(ctx);
}

static const tfrl_env_vtable GRID_VTABLE = {
    .reset = grid_reset,
    .step = grid_step,
    .seed = NULL,
    .destroy = grid_destroy
};

static tfrl_env grid_create(void) {
    grid_env *env = (grid_env *)calloc(1, sizeof(grid_env));
    tfrl_env out = {env, &GRID_VTABLE};
    return out;
}

static void grid_plugin_destroy(tfrl_env *env) {
    if (env && env->vtable && env->vtable->destroy) {
        env->vtable->destroy(env->ctx);
    }
}

static const tfrl_env_plugin GRID_PLUGIN = {
    .api_version = TFRL_ENV_PLUGIN_API_VERSION,
    .name = "first-env-grid",
    .create = grid_create,
    .destroy = grid_plugin_destroy
};

const tfrl_env_plugin *tfrl_env_plugin_get(void) {
    return &GRID_PLUGIN;
}
