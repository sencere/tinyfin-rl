#include "envs/envs_internal.h"

#include <math.h>
#include <stdlib.h>

#define PLAYER_BASE_SIZE 20.0f
#define PLAYER_SPEED 5.0f
#define BALLS_SPEED 2.0f

static int circle_hit(float ax, float ay, float ar, float bx, float by, float br) {
    float dx = ax - bx;
    float dy = ay - by;
    float rr = ar + br;
    return (dx * dx + dy * dy) <= (rr * rr);
}

static void pang_spawn_big(tfrl_env *env) {
    for (int i = 0; i < PANG_MAX_BIG; i++) {
        env->pang.big[i].radius = 40.0f;
        env->pang.big[i].x = tfrl_env_rand_range_float(env, env->pang.big[i].radius,
                                                       (float)PANG_W - env->pang.big[i].radius);
        env->pang.big[i].y = tfrl_env_rand_range_float(env, env->pang.big[i].radius,
                                                       (float)PANG_H * 0.5f);
        float vx = 0.0f;
        float vy = 0.0f;
        while (vx == 0.0f || vy == 0.0f) {
            vx = tfrl_env_rand_range_float(env, -BALLS_SPEED, BALLS_SPEED);
            vy = tfrl_env_rand_range_float(env, -BALLS_SPEED, BALLS_SPEED);
        }
        env->pang.big[i].vx = vx;
        env->pang.big[i].vy = vy;
        env->pang.big[i].active = 1;
    }
}

static int pang_spawn_smaller(tfrl_pang_ball *arr, int max, float x, float y, float vx, float vy, float radius) {
    for (int i = 0; i < max; i++) {
        if (!arr[i].active) {
            arr[i].active = 1;
            arr[i].x = x;
            arr[i].y = y;
            arr[i].vx = vx;
            arr[i].vy = vy;
            arr[i].radius = radius;
            return 1;
        }
    }
    return 0;
}

static const tfrl_env_spec PANG_SPEC = {
    .name = "pang",
    .obs_n = 4,
    .action_n = 4,
    .max_steps = PANG_MAX_STEPS,
    .width = PANG_W,
    .height = PANG_H,
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

const tfrl_env_spec *tfrl_env_spec_pang(void) {
    return &PANG_SPEC;
}

tfrl_obs tfrl_env_reset_pang(tfrl_env *env, uint64_t seed) {
    (void)seed;
    tfrl_env_reset_state(env);
    env->pang.player_x = (float)PANG_W * 0.5f;
    env->pang.player_y = (float)PANG_H;
    env->pang.ship_height = (PLAYER_BASE_SIZE / 2.0f) / tanf(20.0f * 3.14159265f / 180.0f);
    env->pang.gravity = 0.25f;
    env->pang.shot_active = 0;
    env->pang.shot_x = 0.0f;
    env->pang.shot_y = 0.0f;
    env->pang.line_x = 0.0f;

    for (int i = 0; i < PANG_MAX_MED; i++) env->pang.med[i].active = 0;
    for (int i = 0; i < PANG_MAX_SMALL; i++) env->pang.small[i].active = 0;
    pang_spawn_big(env);

    tfrl_obs obs = {0};
    obs.data_len = 4;
    obs.data[0] = env->pang.player_x / (float)PANG_W;
    obs.data[1] = 0.5f;
    obs.data[2] = 0.5f;
    obs.data[3] = 0.0f;
    return obs;
}

tfrl_step_result tfrl_env_step_pang(tfrl_env *env, tfrl_action action) {
    int act = action.index;
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
        if (env->pang.shot_y < 0.0f) {
            env->pang.shot_active = 0;
        }
    }

    for (int i = 0; i < PANG_MAX_BIG; i++) {
        if (!env->pang.big[i].active) continue;
        tfrl_pang_ball *b = &env->pang.big[i];
        b->x += b->vx;
        b->y += b->vy;
        if (b->x + b->radius >= PANG_W || b->x - b->radius <= 0) b->vx *= -1.0f;
        if (b->y - b->radius <= 0) b->vy *= -1.5f;
        if (b->y + b->radius >= PANG_H) {
            b->vy *= -1.0f;
            b->y = (float)PANG_H - b->radius;
        }
        b->vy += env->pang.gravity;
    }
    for (int i = 0; i < PANG_MAX_MED; i++) {
        if (!env->pang.med[i].active) continue;
        tfrl_pang_ball *b = &env->pang.med[i];
        b->x += b->vx;
        b->y += b->vy;
        if (b->x + b->radius >= PANG_W || b->x - b->radius <= 0) b->vx *= -1.0f;
        if (b->y - b->radius <= 0) b->vy *= -1.0f;
        if (b->y + b->radius >= PANG_H) {
            b->vy *= -1.0f;
            b->y = (float)PANG_H - b->radius;
        }
        b->vy += env->pang.gravity + 0.12f;
    }
    for (int i = 0; i < PANG_MAX_SMALL; i++) {
        if (!env->pang.small[i].active) continue;
        tfrl_pang_ball *b = &env->pang.small[i];
        b->x += b->vx;
        b->y += b->vy;
        if (b->x + b->radius >= PANG_W || b->x - b->radius <= 0) b->vx *= -1.0f;
        if (b->y - b->radius <= 0) b->vy *= -1.0f;
        if (b->y + b->radius >= PANG_H) {
            b->vy *= -1.0f;
            b->y = (float)PANG_H - b->radius;
        }
        b->vy += env->pang.gravity + 0.25f;
    }

    int done = 0;
    double reward = 0.0;

    float collider_x = env->pang.player_x;
    float collider_y = env->pang.player_y - env->pang.ship_height / 2.0f;
    float collider_r = 12.0f;
    for (int i = 0; i < PANG_MAX_BIG; i++) {
        if (env->pang.big[i].active &&
            circle_hit(collider_x, collider_y, collider_r, env->pang.big[i].x, env->pang.big[i].y, env->pang.big[i].radius)) {
            done = 1;
        }
    }
    for (int i = 0; i < PANG_MAX_MED; i++) {
        if (env->pang.med[i].active &&
            circle_hit(collider_x, collider_y, collider_r, env->pang.med[i].x, env->pang.med[i].y, env->pang.med[i].radius)) {
            done = 1;
        }
    }
    for (int i = 0; i < PANG_MAX_SMALL; i++) {
        if (env->pang.small[i].active &&
            circle_hit(collider_x, collider_y, collider_r, env->pang.small[i].x, env->pang.small[i].y, env->pang.small[i].radius)) {
            done = 1;
        }
    }
    if (done) reward = -1.0;

    if (!done && env->pang.shot_active) {
        for (int i = 0; i < PANG_MAX_BIG; i++) {
            if (!env->pang.big[i].active) continue;
            tfrl_pang_ball *b = &env->pang.big[i];
            if (b->x - b->radius <= env->pang.line_x && b->x + b->radius >= env->pang.line_x &&
                b->y + b->radius >= env->pang.shot_y) {
                b->active = 0;
                env->pang.shot_active = 0;
                reward += 1.0;
                pang_spawn_smaller(env->pang.med, PANG_MAX_MED, b->x, b->y, -BALLS_SPEED, BALLS_SPEED, 20.0f);
                pang_spawn_smaller(env->pang.med, PANG_MAX_MED, b->x, b->y, BALLS_SPEED, BALLS_SPEED, 20.0f);
                break;
            }
        }
    }
    if (!done && env->pang.shot_active) {
        for (int i = 0; i < PANG_MAX_MED; i++) {
            if (!env->pang.med[i].active) continue;
            tfrl_pang_ball *b = &env->pang.med[i];
            if (b->x - b->radius <= env->pang.line_x && b->x + b->radius >= env->pang.line_x &&
                b->y + b->radius >= env->pang.shot_y) {
                b->active = 0;
                env->pang.shot_active = 0;
                reward += 0.5;
                pang_spawn_smaller(env->pang.small, PANG_MAX_SMALL, b->x, b->y, -BALLS_SPEED, -BALLS_SPEED, 10.0f);
                pang_spawn_smaller(env->pang.small, PANG_MAX_SMALL, b->x, b->y, BALLS_SPEED, -BALLS_SPEED, 10.0f);
                break;
            }
        }
    }
    if (!done && env->pang.shot_active) {
        for (int i = 0; i < PANG_MAX_SMALL; i++) {
            if (!env->pang.small[i].active) continue;
            tfrl_pang_ball *b = &env->pang.small[i];
            if (b->x - b->radius <= env->pang.line_x && b->x + b->radius >= env->pang.line_x &&
                b->y + b->radius >= env->pang.shot_y) {
                b->active = 0;
                env->pang.shot_active = 0;
                reward += 0.25;
                break;
            }
        }
    }

    if (!done) {
        int any_active = 0;
        for (int i = 0; i < PANG_MAX_BIG; i++) if (env->pang.big[i].active) any_active = 1;
        for (int i = 0; i < PANG_MAX_MED; i++) if (env->pang.med[i].active) any_active = 1;
        for (int i = 0; i < PANG_MAX_SMALL; i++) if (env->pang.small[i].active) any_active = 1;
        if (!any_active) {
            done = 1;
            reward += 5.0;
        }
    }

    env->steps += 1;
    if (env->steps >= PANG_MAX_STEPS) done = 1;

    float nearest_x = 0.5f;
    float nearest_y = 0.5f;
    float best_dist = 1e9f;
    for (int i = 0; i < PANG_MAX_BIG; i++) {
        if (!env->pang.big[i].active) continue;
        float dx = env->pang.big[i].x - env->pang.player_x;
        float dy = env->pang.big[i].y - env->pang.player_y;
        float dist = dx * dx + dy * dy;
        if (dist < best_dist) {
            best_dist = dist;
            nearest_x = env->pang.big[i].x / (float)PANG_W;
            nearest_y = env->pang.big[i].y / (float)PANG_H;
        }
    }
    for (int i = 0; i < PANG_MAX_MED; i++) {
        if (!env->pang.med[i].active) continue;
        float dx = env->pang.med[i].x - env->pang.player_x;
        float dy = env->pang.med[i].y - env->pang.player_y;
        float dist = dx * dx + dy * dy;
        if (dist < best_dist) {
            best_dist = dist;
            nearest_x = env->pang.med[i].x / (float)PANG_W;
            nearest_y = env->pang.med[i].y / (float)PANG_H;
        }
    }
    for (int i = 0; i < PANG_MAX_SMALL; i++) {
        if (!env->pang.small[i].active) continue;
        float dx = env->pang.small[i].x - env->pang.player_x;
        float dy = env->pang.small[i].y - env->pang.player_y;
        float dist = dx * dx + dy * dy;
        if (dist < best_dist) {
            best_dist = dist;
            nearest_x = env->pang.small[i].x / (float)PANG_W;
            nearest_y = env->pang.small[i].y / (float)PANG_H;
        }
    }

    tfrl_step_result out = {0};
    out.observation.data_len = 4;
    out.observation.data[0] = env->pang.player_x / (float)PANG_W;
    out.observation.data[1] = nearest_x;
    out.observation.data[2] = nearest_y;
    out.observation.data[3] = env->pang.shot_active ? 1.0f : 0.0f;
    out.reward = reward;
    out.done = done;
    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}
