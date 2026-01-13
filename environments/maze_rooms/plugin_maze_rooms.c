#include <stdlib.h>

#include "tinyfin_rl/rl.h"
#include "tinyfin_rl/rl_env_info.h"
#include "tinyfin_rl/rl_plugin.h"

#define MAZE_W 10
#define MAZE_H 10
#define MAX_STEPS 200

typedef struct {
    int x;
    int y;
    int steps;
} maze_env;

static int is_wall(int x, int y) {
    if (x < 0 || x >= MAZE_W || y < 0 || y >= MAZE_H) return 1;
    if (x == 4 && y != 2 && y != 7) return 1;
    if (y == 4 && x != 2 && x != 7) return 1;
    return 0;
}

static int maze_reset(void *ctx, tfrl_value *out_obs) {
    maze_env *env = (maze_env *)ctx;
    env->x = 0;
    env->y = 0;
    env->steps = 0;
    out_obs->type = TFRL_VALUE_I64;
    out_obs->as.i64 = env->y * MAZE_W + env->x;
    return TFRL_OK;
}

static int maze_step(void *ctx, const tfrl_value *action, tfrl_step *out_step) {
    maze_env *env = (maze_env *)ctx;
    int act = (int)action->as.i64;
    int nx = env->x;
    int ny = env->y;
    if (act == 0) ny -= 1;
    else if (act == 1) ny += 1;
    else if (act == 2) nx -= 1;
    else if (act == 3) nx += 1;
    if (!is_wall(nx, ny)) {
        env->x = nx;
        env->y = ny;
    }
    env->steps += 1;
    int reached = (env->x == MAZE_W - 1 && env->y == MAZE_H - 1);

    out_step->observation.type = TFRL_VALUE_I64;
    out_step->observation.as.i64 = env->y * MAZE_W + env->x;
    out_step->reward = reached ? 1.0 : -0.01;
    out_step->terminated = reached;
    out_step->truncated = env->steps >= MAX_STEPS;
    return TFRL_OK;
}

static void maze_destroy(void *ctx) {
    free(ctx);
}

static const tfrl_env_vtable MAZE_VTABLE = {
    .reset = maze_reset,
    .step = maze_step,
    .seed = NULL,
    .destroy = maze_destroy
};

static tfrl_env maze_create(void) {
    maze_env *env = (maze_env *)calloc(1, sizeof(maze_env));
    tfrl_env out = {env, &MAZE_VTABLE};
    return out;
}

static void maze_plugin_destroy(tfrl_env *env) {
    if (env && env->vtable && env->vtable->destroy) {
        env->vtable->destroy(env->ctx);
    }
}

static const tfrl_env_plugin MAZE_PLUGIN = {
    .api_version = TFRL_ENV_PLUGIN_API_VERSION,
    .name = "maze_rooms",
    .create = maze_create,
    .destroy = maze_plugin_destroy
};

static const tfrl_env_metadata MAZE_METADATA = {
    .api_version = TFRL_ENV_INFO_API_VERSION,
    .name = "maze_rooms",
    .observation_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .observation_n = MAZE_W * MAZE_H,
    .action_n = 4,
    .observation_dims = 1,
    .action_dims = 1,
    .observation_low = 0.0,
    .observation_high = (double)(MAZE_W * MAZE_H - 1),
    .action_low = 0.0,
    .action_high = 3.0
};

const tfrl_env_plugin *tfrl_env_plugin_get(void) {
    return &MAZE_PLUGIN;
}

const tfrl_env_metadata *tfrl_env_metadata_get(void) {
    return &MAZE_METADATA;
}
