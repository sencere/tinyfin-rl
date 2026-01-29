#include "envs/envs_internal.h"

static const tfrl_env_spec SNAKE_SPEC = {
    .name = "snake",
    .obs_n = SNAKE_OBS_DIMS,
    .action_n = 4,
    .max_steps = SNAKE_MAX_STEPS,
    .width = SNAKE_W,
    .height = SNAKE_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = SNAKE_OBS_DIMS,
    .action_dims = 1,
    .obs_shape = {SNAKE_OBS_DIMS, 0},
    .action_shape = {4, 0},
    .obs_dtype = TFRL_DTYPE_FLOAT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = 1.0,
    .action_low = 0.0,
    .action_high = 3.0,
    .agent_count = 1,
    .obs_layout = NULL,
};

const tfrl_env_spec *tfrl_env_spec_snake(void) {
    return &SNAKE_SPEC;
}

static int snake_cell_occupied(const tfrl_env *env, int x, int y) {
    for (int i = 0; i < env->snake.len; i++) {
        if (env->snake.xs[i] == x && env->snake.ys[i] == y) return 1;
    }
    return 0;
}

static void snake_spawn_fruit(tfrl_env *env) {
    for (int tries = 0; tries < 200; tries++) {
        int x = tfrl_env_rand_range_int(env, 0, SNAKE_W - 1);
        int y = tfrl_env_rand_range_int(env, 0, SNAKE_H - 1);
        if (!snake_cell_occupied(env, x, y)) {
            env->snake.fruit_x = x;
            env->snake.fruit_y = y;
            return;
        }
    }
    env->snake.fruit_x = env->snake.xs[0];
    env->snake.fruit_y = env->snake.ys[0];
}

static int snake_is_collision(const tfrl_env *env, int x, int y, int allow_tail) {
    if (x < 0 || y < 0 || x >= SNAKE_W || y >= SNAKE_H) return 1;
    int stop = env->snake.len - (allow_tail ? 1 : 0);
    if (stop < 0) stop = 0;
    for (int i = 0; i < stop; i++) {
        if (env->snake.xs[i] == x && env->snake.ys[i] == y) return 1;
    }
    return 0;
}

static int snake_is_opposite(int dir, int act) {
    if (dir == 0 && act == 1) return 1;
    if (dir == 1 && act == 0) return 1;
    if (dir == 2 && act == 3) return 1;
    if (dir == 3 && act == 2) return 1;
    return 0;
}

static float snake_phi(const tfrl_env *env, int hx, int hy) {
    int dx = hx - env->snake.fruit_x;
    if (dx < 0) dx = -dx;
    int dy = hy - env->snake.fruit_y;
    if (dy < 0) dy = -dy;
    float dist = (float)(dx + dy);
    float denom = (float)(SNAKE_W + SNAKE_H);
    if (denom <= 1e-6f) return 0.0f;
    return -0.05f * (dist / denom);
}

static void snake_fill_obs(const tfrl_env *env, tfrl_obs *obs) {
    if (!obs) return;
    int hx = env->snake.xs[0];
    int hy = env->snake.ys[0];
    int danger_up = snake_is_collision(env, hx, hy - 1, 0);
    int danger_down = snake_is_collision(env, hx, hy + 1, 0);
    int danger_left = snake_is_collision(env, hx - 1, hy, 0);
    int danger_right = snake_is_collision(env, hx + 1, hy, 0);

    int fruit_up = env->snake.fruit_y < hy;
    int fruit_down = env->snake.fruit_y > hy;
    int fruit_left = env->snake.fruit_x < hx;
    int fruit_right = env->snake.fruit_x > hx;

    int dir = env->snake.dir;

    obs->data_len = SNAKE_OBS_DIMS;
    obs->data[0] = (float)hx / (float)(SNAKE_W - 1);
    obs->data[1] = (float)hy / (float)(SNAKE_H - 1);
    obs->data[2] = (float)danger_up;
    obs->data[3] = (float)danger_down;
    obs->data[4] = (float)danger_left;
    obs->data[5] = (float)danger_right;
    obs->data[6] = (float)fruit_up;
    obs->data[7] = (float)fruit_down;
    obs->data[8] = (float)fruit_left;
    obs->data[9] = (float)fruit_right;
    obs->data[10] = (float)(dir == 0);
    obs->data[11] = (float)(dir == 1);
    obs->data[12] = (float)(dir == 2);
    obs->data[13] = (float)(dir == 3);
}

tfrl_obs tfrl_env_reset_snake(tfrl_env *env, uint64_t seed) {
    tfrl_env_reset_state(env);
    env->rng_state = (uint32_t)(seed ? seed : 1u);
    env->snake.len = 3;
    env->snake.dir = 1;
    int cx = SNAKE_W / 2;
    int cy = SNAKE_H / 2;
    for (int i = 0; i < env->snake.len; i++) {
        env->snake.xs[i] = cx - i;
        env->snake.ys[i] = cy;
    }
    snake_spawn_fruit(env);
    tfrl_obs obs = {0};
    snake_fill_obs(env, &obs);
    return obs;
}

tfrl_step_result tfrl_env_step_snake(tfrl_env *env, tfrl_action action) {
    const float gamma = 0.99f;
    const float step_penalty = -0.005f;
    const float fruit_reward = 1.0f;
    const float death_penalty = -1.0f;

    int act = action.index;
    if (act < 0 || act > 3) act = env->snake.dir;
    if (snake_is_opposite(env->snake.dir, act)) act = env->snake.dir;
    env->snake.dir = act;

    int hx = env->snake.xs[0];
    int hy = env->snake.ys[0];
    if (act == 0) hy -= 1;
    if (act == 1) hy += 1;
    if (act == 2) hx -= 1;
    if (act == 3) hx += 1;

    float phi_prev = snake_phi(env, env->snake.xs[0], env->snake.ys[0]);
    int will_grow = (hx == env->snake.fruit_x && hy == env->snake.fruit_y);
    int done = 0;
    double reward = step_penalty;

    if (snake_is_collision(env, hx, hy, !will_grow)) {
        done = 1;
        reward += death_penalty;
    } else {
        if (will_grow && env->snake.len < SNAKE_MAX_LEN) env->snake.len += 1;
        for (int i = env->snake.len - 1; i > 0; i--) {
            env->snake.xs[i] = env->snake.xs[i - 1];
            env->snake.ys[i] = env->snake.ys[i - 1];
        }
        env->snake.xs[0] = hx;
        env->snake.ys[0] = hy;
        if (will_grow) {
            reward += fruit_reward;
            snake_spawn_fruit(env);
        }
    }

    if (!done) {
        float phi_next = snake_phi(env, env->snake.xs[0], env->snake.ys[0]);
        reward += gamma * phi_next - phi_prev;
    }

    env->steps += 1;
    if (env->steps >= SNAKE_MAX_STEPS) done = 1;

    tfrl_step_result out = {0};
    snake_fill_obs(env, &out.observation);
    out.reward = reward;
    out.done = done;
    env->last_reward = (float)reward;
    env->last_done = done;
    return out;
}
