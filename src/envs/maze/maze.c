#include "envs/envs_internal.h"

int tfrl_env_is_wall_maze(int x, int y) {
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

const tfrl_env_spec *tfrl_env_spec_maze(void) {
    return &MAZE_SPEC;
}

tfrl_obs tfrl_env_reset_maze(tfrl_env *env, uint64_t seed) {
    (void)seed;
    tfrl_env_reset_state(env);
    tfrl_obs obs = {0};
    obs.index = env->y * MAZE_W + env->x;
    return obs;
}

tfrl_step_result tfrl_env_step_maze(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    int nx = env->x;
    int ny = env->y;
    if (act == 0) ny -= 1;
    else if (act == 1) ny += 1;
    else if (act == 2) nx -= 1;
    else if (act == 3) nx += 1;
    if (!tfrl_env_is_wall_maze(nx, ny)) {
        env->x = nx;
        env->y = ny;
    }
    env->steps += 1;
    int reached = (env->x == MAZE_W - 1 && env->y == MAZE_H - 1);
    int done = reached || (env->steps >= MAZE_MAX_STEPS);

    tfrl_step_result out = {0};
    out.observation.index = env->y * MAZE_W + env->x;
    out.reward = reached ? 1.0 : -0.01;
    out.done = done;
    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}
