#include "envs/envs_internal.h"

#include <stdlib.h>

static void snake_place_fruit(tfrl_env *env) {
    int tries = 0;
    int fx = 0;
    int fy = 0;
    while (tries < 1000) {
        fx = tfrl_env_rand_range_int(env, 0, SNAKE_W - 1);
        fy = tfrl_env_rand_range_int(env, 0, SNAKE_H - 1);
        int hit = 0;
        for (int i = 0; i < env->snake.len; i++) {
            if (env->snake.xs[i] == fx && env->snake.ys[i] == fy) {
                hit = 1;
                break;
            }
        }
        if (!hit) break;
        tries++;
    }
    env->snake.fruit_x = fx;
    env->snake.fruit_y = fy;
}

static const tfrl_env_spec SNAKE_SPEC = {
    .name = "snake",
    .obs_n = 4,
    .action_n = 4,
    .max_steps = SNAKE_MAX_STEPS,
    .width = SNAKE_W,
    .height = SNAKE_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 4,
    .action_dims = 1,
    .obs_shape = {4, 0},
    .action_shape = {4, 0},
    .obs_dtype = TFRL_DTYPE_FLOAT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = 1.0,
    .action_low = 0.0,
    .action_high = 3.0,
    .agent_count = 1,
};

const tfrl_env_spec *tfrl_env_spec_snake(void) {
    return &SNAKE_SPEC;
}

tfrl_obs tfrl_env_reset_snake(tfrl_env *env, uint64_t seed) {
    (void)seed;
    tfrl_env_reset_state(env);
    env->snake.len = 1;
    env->snake.dir = 3;
    env->snake.xs[0] = SNAKE_W / 2;
    env->snake.ys[0] = SNAKE_H / 2;
    snake_place_fruit(env);
    env->x = env->snake.xs[0];
    env->y = env->snake.ys[0];

    tfrl_obs obs = {0};
    obs.data_len = 4;
    obs.data[0] = (float)env->snake.xs[0] / (float)(SNAKE_W - 1);
    obs.data[1] = (float)env->snake.ys[0] / (float)(SNAKE_H - 1);
    obs.data[2] = (float)env->snake.fruit_x / (float)(SNAKE_W - 1);
    obs.data[3] = (float)env->snake.fruit_y / (float)(SNAKE_H - 1);
    return obs;
}

tfrl_step_result tfrl_env_step_snake(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    if (act < 0 || act > 3) act = env->snake.dir;

    int dir = env->snake.dir;
    if ((dir == 0 && act == 1) || (dir == 1 && act == 0) ||
        (dir == 2 && act == 3) || (dir == 3 && act == 2)) {
        act = dir;
    }
    env->snake.dir = act;

    int dx = 0;
    int dy = 0;
    if (act == 0) dy = -1;
    else if (act == 1) dy = 1;
    else if (act == 2) dx = -1;
    else if (act == 3) dx = 1;

    int head_x = env->snake.xs[0];
    int head_y = env->snake.ys[0];
    int nx = head_x + dx;
    int ny = head_y + dy;

    int done = 0;
    double reward = 0.0;
    if (nx < 0 || nx >= SNAKE_W || ny < 0 || ny >= SNAKE_H) {
        done = 1;
        reward = -1.0;
    } else {
        for (int i = 0; i < env->snake.len; i++) {
            if (env->snake.xs[i] == nx && env->snake.ys[i] == ny) {
                done = 1;
                reward = -1.0;
                break;
            }
        }
    }

    if (!done) {
        int old_tail_x = env->snake.xs[env->snake.len - 1];
        int old_tail_y = env->snake.ys[env->snake.len - 1];
        for (int i = env->snake.len - 1; i >= 1; i--) {
            env->snake.xs[i] = env->snake.xs[i - 1];
            env->snake.ys[i] = env->snake.ys[i - 1];
        }
        env->snake.xs[0] = nx;
        env->snake.ys[0] = ny;

        if (nx == env->snake.fruit_x && ny == env->snake.fruit_y) {
            reward = 1.0;
            if (env->snake.len < SNAKE_MAX_LEN) {
                env->snake.xs[env->snake.len] = old_tail_x;
                env->snake.ys[env->snake.len] = old_tail_y;
                env->snake.len += 1;
            }
            snake_place_fruit(env);
        }
    }

    env->steps += 1;
    if (env->steps >= SNAKE_MAX_STEPS) done = 1;

    env->x = env->snake.xs[0];
    env->y = env->snake.ys[0];

    tfrl_step_result out = {0};
    out.observation.data_len = 4;
    out.observation.data[0] = (float)env->snake.xs[0] / (float)(SNAKE_W - 1);
    out.observation.data[1] = (float)env->snake.ys[0] / (float)(SNAKE_H - 1);
    out.observation.data[2] = (float)env->snake.fruit_x / (float)(SNAKE_W - 1);
    out.observation.data[3] = (float)env->snake.fruit_y / (float)(SNAKE_H - 1);
    out.reward = reward;
    out.done = done;
    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}
