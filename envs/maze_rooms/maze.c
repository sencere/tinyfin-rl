#include "envs/envs_internal.h"

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
    .obs_layout = NULL,
};

const tfrl_env_spec *tfrl_env_spec_maze(void) {
    return &MAZE_SPEC;
}

static void maze_spawn_agent(tfrl_env *env) {
    for (int y = 1; y < MAZE_H - 1; y++) {
        for (int x = 1; x < MAZE_W - 1; x++) {
            if (!tfrl_env_is_wall_maze(x, y)) {
                env->x = x;
                env->y = y;
                return;
            }
        }
    }
}

tfrl_obs tfrl_env_reset_maze(tfrl_env *env, uint64_t seed) {
    tfrl_env_reset_state(env);
    env->rng_state = (uint32_t)(seed ? seed : 1u);
    maze_spawn_agent(env);
    tfrl_obs obs = {0};
    obs.index = env->y * MAZE_W + env->x;
    return obs;
}

tfrl_step_result tfrl_env_step_maze(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    if (act < 0 || act > 3) act = 0;
    int nx = env->x;
    int ny = env->y;
    if (act == 0) ny -= 1;
    if (act == 1) ny += 1;
    if (act == 2) nx -= 1;
    if (act == 3) nx += 1;
    if (!tfrl_env_is_wall_maze(nx, ny)) {
        env->x = nx;
        env->y = ny;
    }
    env->steps += 1;
    int reached = (env->x == MAZE_W - 2 && env->y == MAZE_H - 2);
    int done = reached || (env->steps >= MAZE_MAX_STEPS);

    tfrl_step_result out = {0};
    out.observation.index = env->y * MAZE_W + env->x;
    out.reward = reached ? 1.0 : -0.001;
    out.done = done;
    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}
