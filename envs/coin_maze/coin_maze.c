#include "envs/envs_internal.h"

#include <string.h>

static const tfrl_obs_field COIN_MAZE_OBS_FIELDS[] = {
    {.name = "grid", .len = COIN_MAZE_W * COIN_MAZE_H, .dims = 1,
     .shape = {COIN_MAZE_W * COIN_MAZE_H, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "agent_pos", .len = 2, .dims = 1, .shape = {2, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "agent2_pos", .len = 2, .dims = 1, .shape = {2, 0}, .dtype = TFRL_DTYPE_FLOAT32},
};

static const tfrl_obs_layout COIN_MAZE_OBS_LAYOUT = {
    .field_count = (int)(sizeof(COIN_MAZE_OBS_FIELDS) / sizeof(COIN_MAZE_OBS_FIELDS[0])),
    .fields = COIN_MAZE_OBS_FIELDS,
};

static const tfrl_env_spec COIN_MAZE_SPEC = {
    .name = "coin_maze",
    .obs_n = COIN_MAZE_OBS_DIMS,
    .action_n = 4,
    .max_steps = COIN_MAZE_MAX_STEPS,
    .width = COIN_MAZE_W,
    .height = COIN_MAZE_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = COIN_MAZE_OBS_DIMS,
    .action_dims = 1,
    .obs_shape = {COIN_MAZE_OBS_DIMS, 0},
    .action_shape = {4, 0},
    .obs_dtype = TFRL_DTYPE_FLOAT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = 1.0,
    .action_low = 0.0,
    .action_high = 3.0,
    .agent_count = 1,
    .obs_layout = &COIN_MAZE_OBS_LAYOUT,
};

static const tfrl_env_spec COIN_MAZE_DUO_SPEC = {
    .name = "coin_maze_duo",
    .obs_n = COIN_MAZE_OBS_DIMS,
    .action_n = 4,
    .max_steps = COIN_MAZE_MAX_STEPS,
    .width = COIN_MAZE_W,
    .height = COIN_MAZE_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = COIN_MAZE_OBS_DIMS,
    .action_dims = 1,
    .obs_shape = {COIN_MAZE_OBS_DIMS, 0},
    .action_shape = {4, 0},
    .obs_dtype = TFRL_DTYPE_FLOAT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = 1.0,
    .action_low = 0.0,
    .action_high = 3.0,
    .agent_count = LINEWORLD_DUO_AGENTS,
    .obs_layout = &COIN_MAZE_OBS_LAYOUT,
};

const tfrl_env_spec *tfrl_env_spec_coin_maze(void) { return &COIN_MAZE_SPEC; }
const tfrl_env_spec *tfrl_env_spec_coin_maze_duo(void) { return &COIN_MAZE_DUO_SPEC; }

static int coin_maze_is_wall(int x, int y) {
    if (x < 0 || y < 0 || x >= COIN_MAZE_W || y >= COIN_MAZE_H) return 1;
    if (x == 0 || y == 0 || x == COIN_MAZE_W - 1 || y == COIN_MAZE_H - 1) return 1;
    return ((x + y) % 5) == 2;
}

static void coin_maze_fill_obs(const tfrl_env *env, tfrl_obs *obs) {
    if (!obs) return;
    float grid[COIN_MAZE_W * COIN_MAZE_H];
    for (int y = 0; y < COIN_MAZE_H; y++) {
        for (int x = 0; x < COIN_MAZE_W; x++) {
            float v = 0.0f;
            if (coin_maze_is_wall(x, y)) {
                v = 1.0f;
            } else {
                for (int i = 0; i < COIN_MAZE_COINS; i++) {
                    if (!env->coins_collected[i] &&
                        env->coins_x[i] == x && env->coins_y[i] == y) {
                        v = 0.5f;
                        break;
                    }
                }
            }
            grid[y * COIN_MAZE_W + x] = v;
        }
    }

    float agent_pos[2] = {
        (float)env->x / (float)(COIN_MAZE_W - 1),
        (float)env->y / (float)(COIN_MAZE_H - 1),
    };
    float agent2_pos[2] = {
        (float)env->x2 / (float)(COIN_MAZE_W - 1),
        (float)env->y2 / (float)(COIN_MAZE_H - 1),
    };

    const float *fields[] = {grid, agent_pos, agent2_pos};
    if (!tfrl_obs_flatten(&COIN_MAZE_OBS_LAYOUT, fields, obs)) {
        obs->data_len = 0;
    }
}

static void place_coins(tfrl_env *env) {
    for (int i = 0; i < COIN_MAZE_COINS; i++) {
        int x = 1;
        int y = 1;
        for (int tries = 0; tries < 64; tries++) {
            x = tfrl_env_rand_range_int(env, 1, COIN_MAZE_W - 2);
            y = tfrl_env_rand_range_int(env, 1, COIN_MAZE_H - 2);
            if (coin_maze_is_wall(x, y)) continue;
            if (x == 1 && y == 1) continue;
            if (x == COIN_MAZE_W - 2 && y == COIN_MAZE_H - 2) continue;
            int ok = 1;
            for (int j = 0; j < i; j++) {
                if (env->coins_x[j] == x && env->coins_y[j] == y) ok = 0;
            }
            if (ok) break;
        }
        env->coins_x[i] = x;
        env->coins_y[i] = y;
        env->coins_collected[i] = 0;
    }
}

static int coins_left(const tfrl_env *env) {
    int left = 0;
    for (int i = 0; i < COIN_MAZE_COINS; i++) {
        if (!env->coins_collected[i]) left++;
    }
    return left;
}

tfrl_obs tfrl_env_reset_coin_maze(tfrl_env *env, uint64_t seed) {
    tfrl_env_reset_state(env);
    env->rng_state = (uint32_t)(seed ? seed : 1u);
    env->x = 1;
    env->y = 1;
    env->x2 = 1;
    env->y2 = 1;
    place_coins(env);
    tfrl_obs obs = {0};
    coin_maze_fill_obs(env, &obs);
    return obs;
}

tfrl_obs tfrl_env_reset_coin_maze_duo(tfrl_env *env, uint64_t seed) {
    return tfrl_env_reset_coin_maze(env, seed);
}

static void apply_action(int *x, int *y, int act) {
    int nx = *x;
    int ny = *y;
    if (act == 0) ny -= 1;
    if (act == 1) ny += 1;
    if (act == 2) nx -= 1;
    if (act == 3) nx += 1;
    if (!coin_maze_is_wall(nx, ny)) {
        *x = nx;
        *y = ny;
    }
}

static double collect_coin(tfrl_env *env, int x, int y) {
    for (int i = 0; i < COIN_MAZE_COINS; i++) {
        if (!env->coins_collected[i] && env->coins_x[i] == x && env->coins_y[i] == y) {
            env->coins_collected[i] = 1;
            return 1.0;
        }
    }
    return 0.0;
}

tfrl_step_result tfrl_env_step_coin_maze(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    if (act < 0 || act > 3) act = 0;
    apply_action(&env->x, &env->y, act);
    env->steps += 1;
    double reward = collect_coin(env, env->x, env->y);
    int all_collected = coins_left(env) == 0;
    int done = all_collected || (env->steps >= COIN_MAZE_MAX_STEPS);
    if (all_collected) reward += 2.0;
    if (!all_collected) reward -= 0.01;

    tfrl_step_result out = {0};
    coin_maze_fill_obs(env, &out.observation);
    out.reward = reward;
    out.done = done;
    env->last_reward = (float)reward;
    env->last_done = done;
    return out;
}

tfrl_step_result tfrl_env_step_coin_maze_duo(tfrl_env *env, tfrl_action action) {
    return tfrl_env_step_coin_maze(env, action);
}

int tfrl_env_step_multi_coin_maze_duo(tfrl_env *env, const tfrl_action *actions, int action_count,
                                      tfrl_step_result *out_steps, int max_agents) {
    if (!env || !actions || !out_steps || max_agents < 2 || action_count < 2) return 0;
    int a0 = actions[0].index;
    int a1 = actions[1].index;
    if (a0 < 0 || a0 > 3) a0 = 0;
    if (a1 < 0 || a1 > 3) a1 = 0;
    apply_action(&env->x, &env->y, a0);
    apply_action(&env->x2, &env->y2, a1);
    env->steps += 1;
    double reward = 0.0;
    reward += collect_coin(env, env->x, env->y);
    reward += collect_coin(env, env->x2, env->y2);
    int all_collected = coins_left(env) == 0;
    int done = all_collected || (env->steps >= COIN_MAZE_MAX_STEPS);
    if (all_collected) reward += 2.0;
    if (!all_collected) reward -= 0.01;

    for (int i = 0; i < 2; i++) {
        coin_maze_fill_obs(env, &out_steps[i].observation);
        out_steps[i].reward = reward;
        out_steps[i].done = done;
    }
    env->last_reward = (float)reward;
    env->last_done = done;
    return 2;
}
