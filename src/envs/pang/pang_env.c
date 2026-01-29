#include "envs/envs_internal.h"

#include <math.h>
#include <string.h>

#define PLAYER_SPEED 4.0f
#define PLAYER_BASE_SIZE 30.0f
#define BALLS_SPEED 2.5f

static const tfrl_env_spec PANG_SPEC = {
    .name = "pang",
    .obs_n = 6,
    .action_n = 4,
    .max_steps = PANG_MAX_STEPS,
    .width = PANG_W,
    .height = PANG_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 6,
    .action_dims = 1,
    .obs_shape = {6, 0},
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

const tfrl_env_spec *tfrl_env_spec_pang(void) { return &PANG_SPEC; }

static void pang_init_balls(tfrl_env *env) {
    for (int i = 0; i < PANG_MAX_BIG; i++) {
        env->pang.big[i].radius = 24.0f;
        env->pang.big[i].x = tfrl_env_rand_range_float(env, env->pang.big[i].radius,
                                                       (float)PANG_W - env->pang.big[i].radius);
        env->pang.big[i].y = tfrl_env_rand_range_float(env, env->pang.big[i].radius,
                                                       (float)PANG_H * 0.5f);
        env->pang.big[i].vx = (i % 2 == 0) ? BALLS_SPEED : -BALLS_SPEED;
        env->pang.big[i].vy = -BALLS_SPEED;
        env->pang.big[i].active = 1;
    }
    for (int i = 0; i < PANG_MAX_MED; i++) env->pang.med[i].active = 0;
    for (int i = 0; i < PANG_MAX_SMALL; i++) env->pang.small[i].active = 0;
}

tfrl_obs tfrl_env_reset_pang(tfrl_env *env, uint64_t seed) {
    tfrl_env_reset_state(env);
    env->rng_state = (uint32_t)(seed ? seed : 1u);
    env->pang.player_x = (float)PANG_W * 0.5f;
    env->pang.player_y = (float)PANG_H;
    env->pang.ship_height = 16.0f;
    env->pang.gravity = 0.3f;
    env->pang.shot_active = 0;
    env->pang.shot_x = 0.0f;
    env->pang.shot_y = 0.0f;
    env->pang.line_x = 0.0f;
    pang_init_balls(env);

    tfrl_obs obs = {0};
    obs.data_len = 6;
    obs.data[0] = env->pang.player_x / (float)PANG_W;
    obs.data[1] = env->pang.shot_active ? 1.0f : 0.0f;
    obs.data[2] = env->pang.big[0].x / (float)PANG_W;
    obs.data[3] = env->pang.big[0].y / (float)PANG_H;
    obs.data[4] = env->pang.big[0].vx / (2.0f * BALLS_SPEED);
    obs.data[5] = env->pang.big[0].vy / (2.0f * BALLS_SPEED);
    return obs;
}

static int circle_hit(float ax, float ay, float ar, float bx, float by, float br) {
    float dx = ax - bx;
    float dy = ay - by;
    float rr = ar + br;
    return (dx * dx + dy * dy) <= (rr * rr);
}

static void pang_step_ball(tfrl_env *env, tfrl_pang_ball *b) {
    if (!b->active) return;
    b->x += b->vx;
    b->y += b->vy;
    b->vy += env->pang.gravity;
    if (b->x - b->radius <= 0.0f || b->x + b->radius >= (float)PANG_W) b->vx *= -1.0f;
    if (b->y + b->radius >= (float)PANG_H) {
        b->y = (float)PANG_H - b->radius;
        b->vy = -fabsf(b->vy) * 0.9f;
    }
}

tfrl_step_result tfrl_env_step_pang(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    if (act < 0 || act > 3) act = 0;
    if (act == 1) env->pang.player_x -= PLAYER_SPEED;
    if (act == 2) env->pang.player_x += PLAYER_SPEED;
    if (env->pang.player_x < PLAYER_BASE_SIZE / 2.0f) env->pang.player_x = PLAYER_BASE_SIZE / 2.0f;
    if (env->pang.player_x > (float)PANG_W - PLAYER_BASE_SIZE / 2.0f) env->pang.player_x = (float)PANG_W - PLAYER_BASE_SIZE / 2.0f;

    if (act == 3 && !env->pang.shot_active) {
        env->pang.shot_active = 1;
        env->pang.shot_x = env->pang.player_x;
        env->pang.shot_y = env->pang.player_y - env->pang.ship_height;
        env->pang.line_x = env->pang.player_x;
    }

    if (env->pang.shot_active) {
        env->pang.shot_y -= PLAYER_SPEED;
        if (env->pang.shot_y < 0.0f) env->pang.shot_active = 0;
    }

    for (int i = 0; i < PANG_MAX_BIG; i++) pang_step_ball(env, &env->pang.big[i]);

    double reward = -0.01;
    int done = 0;

    float collider_x = env->pang.player_x;
    float collider_y = env->pang.player_y - env->pang.ship_height * 0.5f;
    float collider_r = PLAYER_BASE_SIZE * 0.5f;
    for (int i = 0; i < PANG_MAX_BIG; i++) {
        if (env->pang.big[i].active &&
            circle_hit(collider_x, collider_y, collider_r, env->pang.big[i].x, env->pang.big[i].y, env->pang.big[i].radius)) {
            done = 1;
            reward = -1.0;
        }
    }

    if (!done && env->pang.shot_active) {
        for (int i = 0; i < PANG_MAX_BIG; i++) {
            if (!env->pang.big[i].active) continue;
            tfrl_pang_ball *b = &env->pang.big[i];
            if (fabsf(b->x - env->pang.line_x) <= b->radius && b->y + b->radius >= env->pang.shot_y) {
                b->active = 0;
                env->pang.shot_active = 0;
                reward += 1.0;
            }
        }
    }

    int any_active = 0;
    for (int i = 0; i < PANG_MAX_BIG; i++) if (env->pang.big[i].active) any_active = 1;
    if (!any_active) {
        done = 1;
        reward += 2.0;
    }

    env->steps += 1;
    if (env->steps >= PANG_MAX_STEPS) done = 1;

    tfrl_step_result out = {0};
    out.observation.data_len = 6;
    out.observation.data[0] = env->pang.player_x / (float)PANG_W;
    out.observation.data[1] = env->pang.shot_active ? 1.0f : 0.0f;
    out.observation.data[2] = env->pang.big[0].x / (float)PANG_W;
    out.observation.data[3] = env->pang.big[0].y / (float)PANG_H;
    out.observation.data[4] = env->pang.big[0].vx / (2.0f * BALLS_SPEED);
    out.observation.data[5] = env->pang.big[0].vy / (2.0f * BALLS_SPEED);
    out.reward = reward;
    out.done = done;
    env->last_reward = (float)reward;
    env->last_done = done;
    return out;
}
