#include <stdlib.h>
#include <string.h>

#include "env_api.h"
#include "render_snapshot.h"

typedef enum {
    TFRL_ENV_MAZE = 0,
    TFRL_ENV_LINEWORLD = 1,
    TFRL_ENV_POINT1D = 2,
    TFRL_ENV_LINEWORLD_CONT = 3,
    TFRL_ENV_COIN_MAZE = 4,
    TFRL_ENV_LINEWORLD_DUO = 5,
} tfrl_env_kind;

#define MAZE_W 10
#define MAZE_H 10
#define MAZE_MAX_STEPS 200

#define LINEWORLD_W 7
#define LINEWORLD_H 1
#define LINEWORLD_MAX_STEPS 50
#define LINEWORLD_DUO_AGENTS 2

#define POINT1D_W 21
#define POINT1D_H 1
#define POINT1D_MAX_STEPS 100
#define POINT1D_STEP_SCALE 0.1f

#define COIN_MAZE_W 10
#define COIN_MAZE_H 10
#define COIN_MAZE_MAX_STEPS 200
#define COIN_MAZE_COINS 4

struct tfrl_env {
    tfrl_env_kind kind;
    int x;
    int y;
    int x2;
    int y2;
    float pos;
    int steps;
    int coins_x[COIN_MAZE_COINS];
    int coins_y[COIN_MAZE_COINS];
    int coins_collected[COIN_MAZE_COINS];
    float last_reward;
    int last_done;
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

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t agent_x;
    uint32_t agent_y;
    uint32_t goal_x;
    uint32_t goal_y;
    uint32_t step;
    uint32_t max_steps;
    float reward;
    uint32_t done;
    uint32_t env_kind;
} tfrl_grid_snapshot_v2;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t agent_x;
    uint32_t agent_y;
    uint32_t goal_x;
    uint32_t goal_y;
    uint32_t step;
    uint32_t max_steps;
    float reward;
    uint32_t done;
    uint32_t env_kind;
    uint32_t entity_count;
} tfrl_grid_snapshot_v3;

typedef struct {
    uint32_t type;
    float x;
    float y;
    float value;
} tfrl_entity_snapshot;

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
    .obs_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {MAZE_W * MAZE_H, 0},
    .action_shape = {4, 0},
    .obs_dtype = TFRL_DTYPE_INT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = (double)(MAZE_W * MAZE_H - 1),
    .action_low = 0.0,
    .action_high = 3.0,
    .agent_count = 1,
};

static const tfrl_env_spec LINEWORLD_SPEC = {
    .name = "lineworld",
    .obs_n = LINEWORLD_W * LINEWORLD_H,
    .action_n = 2,
    .max_steps = LINEWORLD_MAX_STEPS,
    .width = LINEWORLD_W,
    .height = LINEWORLD_H,
    .obs_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {LINEWORLD_W * LINEWORLD_H, 0},
    .action_shape = {2, 0},
    .obs_dtype = TFRL_DTYPE_INT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = (double)(LINEWORLD_W * LINEWORLD_H - 1),
    .action_low = 0.0,
    .action_high = 1.0,
    .agent_count = 1,
};

static const tfrl_env_spec POINT1D_SPEC = {
    .name = "point1d",
    .obs_n = 1,
    .action_n = 1,
    .max_steps = POINT1D_MAX_STEPS,
    .width = POINT1D_W,
    .height = POINT1D_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_BOX,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {1, 0},
    .action_shape = {1, 0},
    .obs_dtype = TFRL_DTYPE_FLOAT32,
    .action_dtype = TFRL_DTYPE_FLOAT32,
    .obs_low = -1.0,
    .obs_high = 1.0,
    .action_low = -1.0,
    .action_high = 1.0,
    .agent_count = 1,
};

static const tfrl_env_spec LINEWORLD_CONT_SPEC = {
    .name = "lineworld_cont",
    .obs_n = 1,
    .action_n = 2,
    .max_steps = LINEWORLD_MAX_STEPS,
    .width = LINEWORLD_W,
    .height = LINEWORLD_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {1, 0},
    .action_shape = {2, 0},
    .obs_dtype = TFRL_DTYPE_FLOAT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = -1.0,
    .obs_high = 1.0,
    .action_low = 0.0,
    .action_high = 1.0,
    .agent_count = 1,
};

static const tfrl_env_spec LINEWORLD_DUO_SPEC = {
    .name = "lineworld_duo",
    .obs_n = LINEWORLD_W * LINEWORLD_H,
    .action_n = 2,
    .max_steps = LINEWORLD_MAX_STEPS,
    .width = LINEWORLD_W,
    .height = LINEWORLD_H,
    .obs_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {LINEWORLD_W * LINEWORLD_H, 0},
    .action_shape = {2, 0},
    .obs_dtype = TFRL_DTYPE_INT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = (double)(LINEWORLD_W * LINEWORLD_H - 1),
    .action_low = 0.0,
    .action_high = 1.0,
    .agent_count = LINEWORLD_DUO_AGENTS,
};

static const tfrl_env_spec COIN_MAZE_SPEC = {
    .name = "coin_maze",
    .obs_n = COIN_MAZE_W * COIN_MAZE_H,
    .action_n = 4,
    .max_steps = COIN_MAZE_MAX_STEPS,
    .width = COIN_MAZE_W,
    .height = COIN_MAZE_H,
    .obs_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {COIN_MAZE_W * COIN_MAZE_H, 0},
    .action_shape = {4, 0},
    .obs_dtype = TFRL_DTYPE_INT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = (double)(COIN_MAZE_W * COIN_MAZE_H - 1),
    .action_low = 0.0,
    .action_high = 3.0,
    .agent_count = 1,
};

tfrl_env *tfrl_env_create(const tfrl_env_config *cfg) {
    tfrl_env *env = (tfrl_env *)calloc(1, sizeof(tfrl_env));
    if (!env) return NULL;
    if (cfg && cfg->name && strcmp(cfg->name, "lineworld") == 0) {
        env->kind = TFRL_ENV_LINEWORLD;
    } else if (cfg && cfg->name && strcmp(cfg->name, "lineworld_duo") == 0) {
        env->kind = TFRL_ENV_LINEWORLD_DUO;
    } else if (cfg && cfg->name && strcmp(cfg->name, "lineworld_cont") == 0) {
        env->kind = TFRL_ENV_LINEWORLD_CONT;
    } else if (cfg && cfg->name && strcmp(cfg->name, "point1d") == 0) {
        env->kind = TFRL_ENV_POINT1D;
    } else if (cfg && cfg->name && strcmp(cfg->name, "coin_maze") == 0) {
        env->kind = TFRL_ENV_COIN_MAZE;
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
    env->x2 = 1;
    env->y2 = 0;
    env->pos = -1.0f;
    env->steps = 0;
    for (int i = 0; i < COIN_MAZE_COINS; i++) {
        env->coins_collected[i] = 0;
    }
    env->coins_x[0] = 1;
    env->coins_y[0] = 1;
    env->coins_x[1] = 3;
    env->coins_y[1] = 2;
    env->coins_x[2] = 6;
    env->coins_y[2] = 5;
    env->coins_x[3] = 8;
    env->coins_y[3] = 7;
    env->last_reward = 0.0f;
    env->last_done = 0;
    env->frame_index = 0;
    tfrl_obs obs = {0};
    if (env->kind == TFRL_ENV_LINEWORLD) {
        obs.index = env->x;
    } else if (env->kind == TFRL_ENV_LINEWORLD_DUO) {
        obs.index = env->x;
    } else if (env->kind == TFRL_ENV_LINEWORLD_CONT) {
        obs.data_len = 1;
        obs.data[0] = (float)env->x / (float)(LINEWORLD_W - 1) * 2.0f - 1.0f;
    } else if (env->kind == TFRL_ENV_POINT1D) {
        obs.data_len = 1;
        obs.data[0] = env->pos;
    } else if (env->kind == TFRL_ENV_COIN_MAZE) {
        obs.index = env->y * COIN_MAZE_W + env->x;
    } else {
        obs.index = env->y * MAZE_W + env->x;
    }
    return obs;
}

tfrl_step_result tfrl_env_step(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    int nx = env->x;
    int ny = env->y;
    if (env->kind == TFRL_ENV_LINEWORLD || env->kind == TFRL_ENV_LINEWORLD_CONT || env->kind == TFRL_ENV_LINEWORLD_DUO) {
        if (act == 0) nx -= 1;
        else if (act == 1) nx += 1;
        if (nx >= 0 && nx < LINEWORLD_W) {
            env->x = nx;
        }
    } else if (env->kind == TFRL_ENV_POINT1D) {
        float a = 0.0f;
        if (action.data_len > 0) {
            a = action.data[0];
        }
        if (a < -1.0f) a = -1.0f;
        if (a > 1.0f) a = 1.0f;
        env->pos += a * POINT1D_STEP_SCALE;
        if (env->pos < -1.0f) env->pos = -1.0f;
        if (env->pos > 1.0f) env->pos = 1.0f;
    } else if (env->kind == TFRL_ENV_COIN_MAZE) {
        if (act == 0) ny -= 1;
        else if (act == 1) ny += 1;
        else if (act == 2) nx -= 1;
        else if (act == 3) nx += 1;
        if (nx < 0) nx = 0;
        if (ny < 0) ny = 0;
        if (nx >= COIN_MAZE_W) nx = COIN_MAZE_W - 1;
        if (ny >= COIN_MAZE_H) ny = COIN_MAZE_H - 1;
        env->x = nx;
        env->y = ny;
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
    } else if (env->kind == TFRL_ENV_LINEWORLD_DUO) {
        reached = (env->x == LINEWORLD_W - 1) && (env->x2 == LINEWORLD_W - 1);
        done = reached || (env->steps >= LINEWORLD_MAX_STEPS);
    } else if (env->kind == TFRL_ENV_LINEWORLD_CONT) {
        reached = (env->x == LINEWORLD_W - 1);
        done = reached || (env->steps >= LINEWORLD_MAX_STEPS);
    } else if (env->kind == TFRL_ENV_POINT1D) {
        reached = (env->pos >= 0.95f);
        done = reached || (env->steps >= POINT1D_MAX_STEPS);
    } else if (env->kind == TFRL_ENV_COIN_MAZE) {
        int all = 1;
        for (int i = 0; i < COIN_MAZE_COINS; i++) {
            if (!env->coins_collected[i]) {
                all = 0;
                break;
            }
        }
        reached = all && (env->x == COIN_MAZE_W - 1 && env->y == COIN_MAZE_H - 1);
        done = reached || (env->steps >= COIN_MAZE_MAX_STEPS);
    } else {
        reached = (env->x == MAZE_W - 1 && env->y == MAZE_H - 1);
        done = reached || (env->steps >= MAZE_MAX_STEPS);
    }

    tfrl_step_result out = {0};
    if (env->kind == TFRL_ENV_LINEWORLD) {
        out.observation.index = env->x;
    } else if (env->kind == TFRL_ENV_LINEWORLD_DUO) {
        out.observation.index = env->x;
    } else if (env->kind == TFRL_ENV_LINEWORLD_CONT) {
        out.observation.data_len = 1;
        out.observation.data[0] = (float)env->x / (float)(LINEWORLD_W - 1) * 2.0f - 1.0f;
    } else if (env->kind == TFRL_ENV_POINT1D) {
        out.observation.data_len = 1;
        out.observation.data[0] = env->pos;
    } else if (env->kind == TFRL_ENV_COIN_MAZE) {
        out.observation.index = env->y * COIN_MAZE_W + env->x;
    } else {
        out.observation.index = env->y * MAZE_W + env->x;
    }
    if (env->kind == TFRL_ENV_POINT1D) {
        float dist = 1.0f - env->pos;
        if (dist < 0.0f) dist = -dist;
        out.reward = reached ? 1.0 : -dist;
    } else if (env->kind == TFRL_ENV_LINEWORLD_DUO) {
        out.reward = reached ? 1.0 : -0.01;
    } else if (env->kind == TFRL_ENV_COIN_MAZE) {
        float reward = -0.01f;
        for (int i = 0; i < COIN_MAZE_COINS; i++) {
            if (!env->coins_collected[i] && env->x == env->coins_x[i] && env->y == env->coins_y[i]) {
                env->coins_collected[i] = 1;
                reward += 0.2f;
            }
        }
        if (reached) reward += 1.0f;
        out.reward = reward;
    } else {
        out.reward = reached ? 1.0 : -0.01;
    }
    out.done = done;
    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}

const tfrl_env_spec *tfrl_env_get_spec(const tfrl_env *env) {
    if (env->kind == TFRL_ENV_LINEWORLD) {
        return &LINEWORLD_SPEC;
    }
    if (env->kind == TFRL_ENV_LINEWORLD_CONT) {
        return &LINEWORLD_CONT_SPEC;
    }
    if (env->kind == TFRL_ENV_LINEWORLD_DUO) {
        return &LINEWORLD_DUO_SPEC;
    }
    if (env->kind == TFRL_ENV_POINT1D) {
        return &POINT1D_SPEC;
    }
    if (env->kind == TFRL_ENV_COIN_MAZE) {
        return &COIN_MAZE_SPEC;
    }
    return &MAZE_SPEC;
}

int tfrl_env_agent_count(const tfrl_env *env) {
    if (!env) return 0;
    if (env->kind == TFRL_ENV_LINEWORLD_DUO) return LINEWORLD_DUO_AGENTS;
    return 1;
}

int tfrl_env_reset_multi(tfrl_env *env, uint64_t seed, tfrl_obs *out_obs, int max_agents) {
    if (!env || !out_obs || max_agents <= 0) return 0;
    int count = tfrl_env_agent_count(env);
    if (max_agents < count) return 0;
    tfrl_obs obs0 = tfrl_env_reset(env, seed);
    out_obs[0] = obs0;
    if (count > 1) {
        tfrl_obs obs1 = {0};
        obs1.index = env->x2;
        out_obs[1] = obs1;
    }
    return count;
}

int tfrl_env_step_multi(tfrl_env *env, const tfrl_action *actions, int action_count, tfrl_step_result *out_steps, int max_agents) {
    if (!env || !actions || !out_steps || max_agents <= 0) return 0;
    int count = tfrl_env_agent_count(env);
    if (action_count < count || max_agents < count) return 0;
    if (count == 1) {
        out_steps[0] = tfrl_env_step(env, actions[0]);
        return 1;
    }
    int nx0 = env->x;
    int nx1 = env->x2;
    if (actions[0].index == 0) nx0 -= 1;
    else if (actions[0].index == 1) nx0 += 1;
    if (actions[1].index == 0) nx1 -= 1;
    else if (actions[1].index == 1) nx1 += 1;
    if (nx0 >= 0 && nx0 < LINEWORLD_W) env->x = nx0;
    if (nx1 >= 0 && nx1 < LINEWORLD_W) env->x2 = nx1;
    env->steps += 1;
    int reached = (env->x == LINEWORLD_W - 1) && (env->x2 == LINEWORLD_W - 1);
    int done = reached || (env->steps >= LINEWORLD_MAX_STEPS);
    for (int i = 0; i < count; i++) {
        out_steps[i].observation.index = (i == 0) ? env->x : env->x2;
        out_steps[i].reward = reached ? 1.0 : -0.01;
        out_steps[i].done = done;
    }
    env->last_reward = (float)out_steps[0].reward;
    env->last_done = done;
    return count;
}

void tfrl_env_step_batch(tfrl_env **envs, int env_count, const tfrl_action *actions, tfrl_step_result *out_steps) {
    if (!envs || !actions || !out_steps || env_count <= 0) return;
    for (int i = 0; i < env_count; i++) {
        out_steps[i] = tfrl_env_step(envs[i], actions[i]);
    }
}

size_t tfrl_env_render_bytes_needed(const tfrl_env *env) {
    int w = MAZE_W;
    int h = MAZE_H;
    if (env->kind == TFRL_ENV_LINEWORLD || env->kind == TFRL_ENV_LINEWORLD_CONT) {
        w = LINEWORLD_W;
        h = LINEWORLD_H;
    } else if (env->kind == TFRL_ENV_POINT1D) {
        w = POINT1D_W;
        h = POINT1D_H;
    } else if (env->kind == TFRL_ENV_COIN_MAZE) {
        w = COIN_MAZE_W;
        h = COIN_MAZE_H;
    }
    int entity_count = 0;
    if (env->kind == TFRL_ENV_COIN_MAZE) {
        for (int i = 0; i < COIN_MAZE_COINS; i++) {
            if (!env->coins_collected[i]) entity_count++;
        }
    }
    if (env->kind == TFRL_ENV_LINEWORLD_DUO) {
        entity_count += 1;
    }
    size_t payload_bytes = sizeof(tfrl_grid_snapshot_v3) + (w * h) + (entity_count * sizeof(tfrl_entity_snapshot));
    return sizeof(tfrl_render_snapshot_header) + payload_bytes;
}

size_t tfrl_env_render_write(tfrl_env *env, void *buffer, size_t buffer_len) {
    if (!env || !buffer) return 0;
    int w = MAZE_W;
    int h = MAZE_H;
    if (env->kind == TFRL_ENV_LINEWORLD || env->kind == TFRL_ENV_LINEWORLD_CONT) {
        w = LINEWORLD_W;
        h = LINEWORLD_H;
    } else if (env->kind == TFRL_ENV_POINT1D) {
        w = POINT1D_W;
        h = POINT1D_H;
    } else if (env->kind == TFRL_ENV_COIN_MAZE) {
        w = COIN_MAZE_W;
        h = COIN_MAZE_H;
    }
    int entity_count = 0;
    if (env->kind == TFRL_ENV_COIN_MAZE) {
        for (int i = 0; i < COIN_MAZE_COINS; i++) {
            if (!env->coins_collected[i]) entity_count++;
        }
    }
    if (env->kind == TFRL_ENV_LINEWORLD_DUO) {
        entity_count += 1;
    }
    size_t payload_bytes = sizeof(tfrl_grid_snapshot_v3) + (w * h) + (entity_count * sizeof(tfrl_entity_snapshot));
    size_t total = sizeof(tfrl_render_snapshot_header) + payload_bytes;
    if (buffer_len < total) return 0;

    tfrl_render_snapshot_header header = {
        .api_version = TFRL_SNAPSHOT_API_VERSION,
        .frame_index = env->frame_index,
        .payload_bytes = (uint32_t)payload_bytes,
    };
    memcpy(buffer, &header, sizeof(header));

    uint32_t agent_x = (uint32_t)env->x;
    uint32_t agent_y = (uint32_t)env->y;
    uint32_t goal_x = (uint32_t)(w - 1);
    uint32_t goal_y = (uint32_t)(h - 1);
    uint32_t max_steps = (uint32_t)MAZE_MAX_STEPS;
    if (env->kind == TFRL_ENV_LINEWORLD || env->kind == TFRL_ENV_LINEWORLD_CONT || env->kind == TFRL_ENV_LINEWORLD_DUO) {
        max_steps = (uint32_t)LINEWORLD_MAX_STEPS;
    } else if (env->kind == TFRL_ENV_POINT1D) {
        max_steps = (uint32_t)POINT1D_MAX_STEPS;
        float norm = (env->pos + 1.0f) * 0.5f;
        int px = (int)(norm * (float)(w - 1));
        if (px < 0) px = 0;
        if (px >= w) px = w - 1;
        agent_x = (uint32_t)px;
        agent_y = 0;
        goal_x = (uint32_t)(w - 1);
        goal_y = 0;
    }

    tfrl_grid_snapshot_v3 payload = {
        .width = (uint32_t)w,
        .height = (uint32_t)h,
        .agent_x = agent_x,
        .agent_y = agent_y,
        .goal_x = goal_x,
        .goal_y = goal_y,
        .step = (uint32_t)env->steps,
        .max_steps = max_steps,
        .reward = env->last_reward,
        .done = (uint32_t)env->last_done,
        .env_kind = (uint32_t)env->kind,
        .entity_count = (uint32_t)entity_count,
    };
    unsigned char *payload_dst = (unsigned char *)buffer + sizeof(header);
    memcpy(payload_dst, &payload, sizeof(payload));

    unsigned char *walls = payload_dst + sizeof(payload);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
    if (env->kind == TFRL_ENV_LINEWORLD || env->kind == TFRL_ENV_LINEWORLD_CONT || env->kind == TFRL_ENV_LINEWORLD_DUO || env->kind == TFRL_ENV_POINT1D || env->kind == TFRL_ENV_COIN_MAZE) {
        walls[y * w + x] = 0;
    } else {
        walls[y * w + x] = (unsigned char)(is_wall_maze(x, y) ? 1 : 0);
    }
        }
    }
    tfrl_entity_snapshot *entities = (tfrl_entity_snapshot *)(walls + (w * h));
    if (env->kind == TFRL_ENV_COIN_MAZE) {
        int idx = 0;
        for (int i = 0; i < COIN_MAZE_COINS; i++) {
            if (env->coins_collected[i]) continue;
            entities[idx].type = 1;
            entities[idx].x = (float)env->coins_x[i];
            entities[idx].y = (float)env->coins_y[i];
            entities[idx].value = 1.0f;
            idx++;
        }
    }
    if (env->kind == TFRL_ENV_LINEWORLD_DUO) {
        entities[0].type = 2;
        entities[0].x = (float)env->x2;
        entities[0].y = (float)env->y2;
        entities[0].value = 1.0f;
    }
    env->frame_index++;
    return total;
}
