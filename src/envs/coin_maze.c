#include "envs_internal.h"

static const tfrl_env_spec COIN_MAZE_DUO_SPEC = {
    .name = "coin_maze_duo",
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

const tfrl_env_spec *tfrl_env_spec_coin_maze(void) {
    return &COIN_MAZE_SPEC;
}

const tfrl_env_spec *tfrl_env_spec_coin_maze_duo(void) {
    return &COIN_MAZE_DUO_SPEC;
}

tfrl_obs tfrl_env_reset_coin_maze(tfrl_env *env, uint64_t seed) {
    (void)seed;
    tfrl_env_reset_state(env);
    tfrl_obs obs = {0};
    obs.index = env->y * COIN_MAZE_W + env->x;
    return obs;
}

tfrl_obs tfrl_env_reset_coin_maze_duo(tfrl_env *env, uint64_t seed) {
    return tfrl_env_reset_coin_maze(env, seed);
}

tfrl_step_result tfrl_env_step_coin_maze(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    int nx = env->x;
    int ny = env->y;
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
    env->steps += 1;

    float reward = -0.01f;
    for (int i = 0; i < COIN_MAZE_COINS; i++) {
        if (!env->coins_collected[i] && env->x == env->coins_x[i] && env->y == env->coins_y[i]) {
            env->coins_collected[i] = 1;
            reward += 0.2f;
        }
    }
    int all = 1;
    for (int i = 0; i < COIN_MAZE_COINS; i++) {
        if (!env->coins_collected[i]) {
            all = 0;
            break;
        }
    }
    int reached = all && (env->x == COIN_MAZE_W - 1 && env->y == COIN_MAZE_H - 1);
    int done = reached || (env->steps >= COIN_MAZE_MAX_STEPS);
    if (reached) reward += 1.0f;

    tfrl_step_result out = {0};
    out.observation.index = env->y * COIN_MAZE_W + env->x;
    out.reward = reward;
    out.done = done;
    env->last_reward = reward;
    env->last_done = out.done;
    return out;
}

tfrl_step_result tfrl_env_step_coin_maze_duo(tfrl_env *env, tfrl_action action) {
    return tfrl_env_step_coin_maze(env, action);
}

int tfrl_env_step_multi_coin_maze_duo(tfrl_env *env, const tfrl_action *actions, int action_count,
                                      tfrl_step_result *out_steps, int max_agents) {
    if (!env || !actions || !out_steps || max_agents < LINEWORLD_DUO_AGENTS || action_count < LINEWORLD_DUO_AGENTS) {
        return 0;
    }
    int nx0 = env->x;
    int ny0 = env->y;
    int nx1 = env->x2;
    int ny1 = env->y2;
    if (actions[0].index == 0) ny0 -= 1;
    else if (actions[0].index == 1) ny0 += 1;
    else if (actions[0].index == 2) nx0 -= 1;
    else if (actions[0].index == 3) nx0 += 1;
    if (actions[1].index == 0) ny1 -= 1;
    else if (actions[1].index == 1) ny1 += 1;
    else if (actions[1].index == 2) nx1 -= 1;
    else if (actions[1].index == 3) nx1 += 1;
    if (nx0 < 0) nx0 = 0;
    if (ny0 < 0) ny0 = 0;
    if (nx0 >= COIN_MAZE_W) nx0 = COIN_MAZE_W - 1;
    if (ny0 >= COIN_MAZE_H) ny0 = COIN_MAZE_H - 1;
    if (nx1 < 0) nx1 = 0;
    if (ny1 < 0) ny1 = 0;
    if (nx1 >= COIN_MAZE_W) nx1 = COIN_MAZE_W - 1;
    if (ny1 >= COIN_MAZE_H) ny1 = COIN_MAZE_H - 1;
    env->x = nx0;
    env->y = ny0;
    env->x2 = nx1;
    env->y2 = ny1;
    env->steps += 1;

    float reward = -0.01f;
    for (int i = 0; i < COIN_MAZE_COINS; i++) {
        if (env->coins_collected[i]) continue;
        int coin_hit = (env->x == env->coins_x[i] && env->y == env->coins_y[i]) ||
                       (env->x2 == env->coins_x[i] && env->y2 == env->coins_y[i]);
        if (coin_hit) {
            env->coins_collected[i] = 1;
            reward += 0.2f;
        }
    }
    int all = 1;
    for (int i = 0; i < COIN_MAZE_COINS; i++) {
        if (!env->coins_collected[i]) {
            all = 0;
            break;
        }
    }
    int reached = all && (env->x == COIN_MAZE_W - 1 && env->y == COIN_MAZE_H - 1) &&
                  (env->x2 == COIN_MAZE_W - 1 && env->y2 == COIN_MAZE_H - 1);
    int done = reached || (env->steps >= COIN_MAZE_MAX_STEPS);
    if (reached) reward += 1.0f;
    for (int i = 0; i < LINEWORLD_DUO_AGENTS; i++) {
        out_steps[i].observation.index = (i == 0)
                                             ? env->y * COIN_MAZE_W + env->x
                                             : env->y2 * COIN_MAZE_W + env->x2;
        out_steps[i].reward = reward;
        out_steps[i].done = done;
    }
    env->last_reward = reward;
    env->last_done = done;
    return LINEWORLD_DUO_AGENTS;
}
