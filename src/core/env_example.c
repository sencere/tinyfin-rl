#include <stdlib.h>
#include <string.h>

#include "env_api.h"
#include "render_snapshot.h"

typedef enum {
    TFRL_ENV_MAZE = 0,
    TFRL_ENV_LINEWORLD = 1,
} tfrl_env_kind;

#define MAZE_W 10
#define MAZE_H 10
#define MAZE_MAX_STEPS 200

#define LINEWORLD_W 7
#define LINEWORLD_H 1
#define LINEWORLD_MAX_STEPS 50

struct tfrl_env {
    tfrl_env_kind kind;
    int x;
    int y;
    int steps;
    uint32_t frame_index;
};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t agent_x;
    uint32_t agent_y;
    uint32_t goal_x;
    uint32_t goal_y;
    uint32_t step;
    uint32_t max_steps;
} tfrl_grid_snapshot;

static int is_wall_maze(int x, int y) {
    if (x < 0 || x >= MAZE_W || y < 0 || y >= MAZE_H) return 1;
    if (x == 4 && y != 2 && y != 7) return 1;
    if (y == 4 && x != 2 && x != 7) return 1;
    return 0;
}

static const tfrl_env_spec MAZE_SPEC = {
    .name = "maze_rooms",
    .obs_n = MAZE_W * MAZE_H,
    .action_n = 4,
    .max_steps = MAZE_MAX_STEPS,
    .width = MAZE_W,
    .height = MAZE_H,
};

static const tfrl_env_spec LINEWORLD_SPEC = {
    .name = "lineworld",
    .obs_n = LINEWORLD_W * LINEWORLD_H,
    .action_n = 2,
    .max_steps = LINEWORLD_MAX_STEPS,
    .width = LINEWORLD_W,
    .height = LINEWORLD_H,
};

tfrl_env *tfrl_env_create(const tfrl_env_config *cfg) {
    tfrl_env *env = (tfrl_env *)calloc(1, sizeof(tfrl_env));
    if (!env) return NULL;
    if (cfg && cfg->name && strcmp(cfg->name, "lineworld") == 0) {
        env->kind = TFRL_ENV_LINEWORLD;
    } else {
        env->kind = TFRL_ENV_MAZE;
    }
    tfrl_env_reset(env, cfg ? cfg->seed : 0);
    return env;
}

void tfrl_env_destroy(tfrl_env *env) {
    free(env);
}

tfrl_obs tfrl_env_reset(tfrl_env *env, uint64_t seed) {
    (void)seed;
    env->x = 0;
    env->y = 0;
    env->steps = 0;
    env->frame_index = 0;
    tfrl_obs obs = {0};
    if (env->kind == TFRL_ENV_LINEWORLD) {
        obs.index = env->x;
    } else {
        obs.index = env->y * MAZE_W + env->x;
    }
    return obs;
}

tfrl_step_result tfrl_env_step(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    int nx = env->x;
    int ny = env->y;
    if (env->kind == TFRL_ENV_LINEWORLD) {
        if (act == 0) nx -= 1;
        else if (act == 1) nx += 1;
        if (nx >= 0 && nx < LINEWORLD_W) {
            env->x = nx;
        }
    } else {
        if (act == 0) ny -= 1;
        else if (act == 1) ny += 1;
        else if (act == 2) nx -= 1;
        else if (act == 3) nx += 1;
        if (!is_wall_maze(nx, ny)) {
            env->x = nx;
            env->y = ny;
        }
    }
    env->steps += 1;
    int reached = 0;
    int done = 0;
    if (env->kind == TFRL_ENV_LINEWORLD) {
        reached = (env->x == LINEWORLD_W - 1);
        done = reached || (env->steps >= LINEWORLD_MAX_STEPS);
    } else {
        reached = (env->x == MAZE_W - 1 && env->y == MAZE_H - 1);
        done = reached || (env->steps >= MAZE_MAX_STEPS);
    }

    tfrl_step_result out = {0};
    if (env->kind == TFRL_ENV_LINEWORLD) {
        out.observation.index = env->x;
    } else {
        out.observation.index = env->y * MAZE_W + env->x;
    }
    out.reward = reached ? 1.0 : -0.01;
    out.done = done;
    return out;
}

const tfrl_env_spec *tfrl_env_get_spec(const tfrl_env *env) {
    if (env->kind == TFRL_ENV_LINEWORLD) {
        return &LINEWORLD_SPEC;
    }
    return &MAZE_SPEC;
}

void tfrl_env_step_batch(tfrl_env **envs, int env_count, const tfrl_action *actions, tfrl_step_result *out_steps) {
    if (!envs || !actions || !out_steps || env_count <= 0) return;
    for (int i = 0; i < env_count; i++) {
        out_steps[i] = tfrl_env_step(envs[i], actions[i]);
    }
}

size_t tfrl_env_render_bytes_needed(const tfrl_env *env) {
    int w = env->kind == TFRL_ENV_LINEWORLD ? LINEWORLD_W : MAZE_W;
    int h = env->kind == TFRL_ENV_LINEWORLD ? LINEWORLD_H : MAZE_H;
    size_t payload_bytes = sizeof(tfrl_grid_snapshot) + (w * h);
    return sizeof(tfrl_render_snapshot_header) + payload_bytes;
}

size_t tfrl_env_render_write(tfrl_env *env, void *buffer, size_t buffer_len) {
    if (!env || !buffer) return 0;
    int w = env->kind == TFRL_ENV_LINEWORLD ? LINEWORLD_W : MAZE_W;
    int h = env->kind == TFRL_ENV_LINEWORLD ? LINEWORLD_H : MAZE_H;
    size_t payload_bytes = sizeof(tfrl_grid_snapshot) + (w * h);
    size_t total = sizeof(tfrl_render_snapshot_header) + payload_bytes;
    if (buffer_len < total) return 0;

    tfrl_render_snapshot_header header = {
        .api_version = TFRL_SNAPSHOT_API_VERSION,
        .frame_index = env->frame_index,
        .payload_bytes = (uint32_t)payload_bytes,
    };
    memcpy(buffer, &header, sizeof(header));

    tfrl_grid_snapshot payload = {
        .width = (uint32_t)w,
        .height = (uint32_t)h,
        .agent_x = (uint32_t)env->x,
        .agent_y = (uint32_t)env->y,
        .goal_x = (uint32_t)(w - 1),
        .goal_y = (uint32_t)(h - 1),
        .step = (uint32_t)env->steps,
        .max_steps = (uint32_t)(env->kind == TFRL_ENV_LINEWORLD ? LINEWORLD_MAX_STEPS : MAZE_MAX_STEPS),
    };
    unsigned char *payload_dst = (unsigned char *)buffer + sizeof(header);
    memcpy(payload_dst, &payload, sizeof(payload));

    unsigned char *walls = payload_dst + sizeof(payload);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (env->kind == TFRL_ENV_LINEWORLD) {
                walls[y * w + x] = 0;
            } else {
                walls[y * w + x] = (unsigned char)(is_wall_maze(x, y) ? 1 : 0);
            }
        }
    }
    env->frame_index++;
    return total;
}
