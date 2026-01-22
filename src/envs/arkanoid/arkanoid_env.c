#include "envs/envs_internal.h"

#include <math.h>

static float clamp_float(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static float reflect_x(float x, float min_x, float max_x) {
    float span = max_x - min_x;
    if (span <= 0.0f) return clamp_float(x, min_x, max_x);
    float shifted = x - min_x;
    float period = 2.0f * span;
    float m = fmodf(shifted, period);
    if (m < 0.0f) m += period;
    if (m > span) m = period - m;
    return min_x + m;
}

static float arkanoid_predict_landing_x(const tfrl_env *env) {
    float vy = env->arkanoid.ball_vy;
    if (vy <= 0.0f) return env->arkanoid.ball_x;
    float r = env->arkanoid.ball_radius;
    float paddle_y = env->arkanoid.player_y - env->arkanoid.player_h * 0.5f - r;
    float dy = paddle_y - env->arkanoid.ball_y;
    if (dy <= 0.0f) return env->arkanoid.ball_x;
    float t = dy / vy;
    float x = env->arkanoid.ball_x + env->arkanoid.ball_vx * t;
    float min_x = r;
    float max_x = (float)ARKANOID_W - r;
    return reflect_x(x, min_x, max_x);
}

enum {
    ARKANOID_OBS_DIMS = 9,
    ARKANOID_DISC_OBS_N = 4096
};

static int arkanoid_quantize(float v, float max, int bins) {
    if (max <= 0.0f) return 0;
    float t = v / max;
    if (t < 0.0f) t = 0.0f;
    if (t > 0.9999f) t = 0.9999f;
    int b = (int)(t * (float)bins);
    return clamp_int(b, 0, bins - 1);
}

static int arkanoid_hash_obs(const tfrl_env *env) {
    const int pos_bins = 16;
    const int bricks_bins = 16;
    int px = arkanoid_quantize(env->arkanoid.player_x, (float)ARKANOID_W, pos_bins);
    int bx = arkanoid_quantize(env->arkanoid.ball_x, (float)ARKANOID_W, pos_bins);
    int by = arkanoid_quantize(env->arkanoid.ball_y, (float)ARKANOID_H, pos_bins);
    int vx = (env->arkanoid.ball_vx > 0.5f) ? 2 : (env->arkanoid.ball_vx < -0.5f ? 0 : 1);
    int vy = (env->arkanoid.ball_vy > 0.5f) ? 2 : (env->arkanoid.ball_vy < -0.5f ? 0 : 1);
    int active = env->arkanoid.ball_active ? 1 : 0;
    int life = clamp_int(env->arkanoid.life, 0, ARKANOID_PLAYER_MAX_LIFE);
    int bricks = arkanoid_quantize((float)env->arkanoid.bricks_left,
                                   (float)(ARKANOID_LINES * ARKANOID_BRICKS_PER_LINE),
                                   bricks_bins);

    uint32_t h = 2166136261u;
    h ^= (uint32_t)px; h *= 16777619u;
    h ^= (uint32_t)bx; h *= 16777619u;
    h ^= (uint32_t)by; h *= 16777619u;
    h ^= (uint32_t)vx; h *= 16777619u;
    h ^= (uint32_t)vy; h *= 16777619u;
    h ^= (uint32_t)active; h *= 16777619u;
    h ^= (uint32_t)life; h *= 16777619u;
    h ^= (uint32_t)bricks; h *= 16777619u;
    return (int)(h % (uint32_t)ARKANOID_DISC_OBS_N);
}

static int circle_rect_collision(float cx, float cy, float r, float rx, float ry, float rw, float rh) {
    float closest_x = cx;
    float closest_y = cy;
    if (cx < rx) closest_x = rx;
    else if (cx > rx + rw) closest_x = rx + rw;
    if (cy < ry) closest_y = ry;
    else if (cy > ry + rh) closest_y = ry + rh;
    float dx = cx - closest_x;
    float dy = cy - closest_y;
    return (dx * dx + dy * dy) <= (r * r);
}

static const tfrl_env_spec ARKANOID_SPEC = {
    .name = "arkanoid",
    .obs_n = ARKANOID_OBS_DIMS,
    .action_n = 4,
    .max_steps = ARKANOID_MAX_STEPS,
    .width = ARKANOID_W,
    .height = ARKANOID_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = ARKANOID_OBS_DIMS,
    .action_dims = 1,
    .obs_shape = {ARKANOID_OBS_DIMS, 0},
    .action_shape = {4, 0},
    .obs_dtype = TFRL_DTYPE_FLOAT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = 1.0,
    .action_low = 0.0,
    .action_high = 3.0,
    .agent_count = 1,
};

static const tfrl_env_spec ARKANOID_DISC_SPEC = {
    .name = "arkanoid_disc",
    .obs_n = ARKANOID_DISC_OBS_N,
    .action_n = 4,
    .max_steps = ARKANOID_MAX_STEPS,
    .width = ARKANOID_W,
    .height = ARKANOID_H,
    .obs_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {ARKANOID_DISC_OBS_N, 0},
    .action_shape = {4, 0},
    .obs_dtype = TFRL_DTYPE_INT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = (double)(ARKANOID_DISC_OBS_N - 1),
    .action_low = 0.0,
    .action_high = 3.0,
    .agent_count = 1,
};

const tfrl_env_spec *tfrl_env_spec_arkanoid(void) {
    return &ARKANOID_SPEC;
}

const tfrl_env_spec *tfrl_env_spec_arkanoid_disc(void) {
    return &ARKANOID_DISC_SPEC;
}

static void arkanoid_fill_obs(const tfrl_env *env, tfrl_obs *obs) {
    if (!obs) return;
    obs->data_len = ARKANOID_OBS_DIMS;
    obs->data[0] = env->arkanoid.player_x / (float)ARKANOID_W;
    obs->data[1] = env->arkanoid.ball_x / (float)ARKANOID_W;
    obs->data[2] = env->arkanoid.ball_y / (float)ARKANOID_H;
    float vx = clamp_float(env->arkanoid.ball_vx, -ARKANOID_BALL_SPEED, ARKANOID_BALL_SPEED);
    float vy = clamp_float(env->arkanoid.ball_vy, -ARKANOID_BALL_SPEED, ARKANOID_BALL_SPEED);
    obs->data[3] = (vx / ARKANOID_BALL_SPEED + 1.0f) * 0.5f;
    obs->data[4] = (vy / ARKANOID_BALL_SPEED + 1.0f) * 0.5f;
    obs->data[5] = (float)env->arkanoid.bricks_left / (float)(ARKANOID_LINES * ARKANOID_BRICKS_PER_LINE);
    obs->data[6] = env->arkanoid.ball_active ? 1.0f : 0.0f;
    float rel_x = (env->arkanoid.ball_x - env->arkanoid.player_x) / ((float)ARKANOID_W * 0.5f);
    rel_x = clamp_float(rel_x, -1.0f, 1.0f);
    obs->data[7] = (rel_x + 1.0f) * 0.5f;
    obs->data[8] = (float)env->arkanoid.life / (float)ARKANOID_PLAYER_MAX_LIFE;
}

tfrl_obs tfrl_env_reset_arkanoid(tfrl_env *env, uint64_t seed) {
    (void)seed;
    tfrl_env_reset_state(env);
    env->arkanoid.player_w = (float)ARKANOID_W / 10.0f;
    env->arkanoid.player_h = 20.0f;
    env->arkanoid.player_x = (float)ARKANOID_W / 2.0f;
    env->arkanoid.player_y = (float)ARKANOID_H * 7.0f / 8.0f;
    env->arkanoid.life = ARKANOID_PLAYER_MAX_LIFE;

    env->arkanoid.ball_radius = ARKANOID_BALL_RADIUS;
    env->arkanoid.ball_active = 0;
    env->arkanoid.ball_vx = 0.0f;
    env->arkanoid.ball_vy = 0.0f;
    env->arkanoid.ball_x = env->arkanoid.player_x;
    env->arkanoid.ball_y = env->arkanoid.player_y - env->arkanoid.player_h * 0.5f - env->arkanoid.ball_radius;

    env->arkanoid.brick_w = (float)ARKANOID_W / (float)ARKANOID_BRICKS_PER_LINE;
    env->arkanoid.brick_h = ARKANOID_BRICK_HEIGHT;
    env->arkanoid.bricks_left = 0;
    for (int i = 0; i < ARKANOID_LINES; i++) {
        for (int j = 0; j < ARKANOID_BRICKS_PER_LINE; j++) {
            env->arkanoid.bricks[i][j] = 1;
            env->arkanoid.bricks_left++;
        }
    }
    env->arkanoid.prev_phi = 0.0f;
    for (int i = 0; i < 4; i++) {
        env->arkanoid.reward_vec[i] = 0.0f;
    }

    tfrl_obs obs = {0};
    arkanoid_fill_obs(env, &obs);
    return obs;
}

tfrl_obs tfrl_env_reset_arkanoid_disc(tfrl_env *env, uint64_t seed) {
    tfrl_obs obs = tfrl_env_reset_arkanoid(env, seed);
    obs.index = arkanoid_hash_obs(env);
    obs.data_len = 0;
    return obs;
}

tfrl_step_result tfrl_env_step_arkanoid(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    if (act < 0 || act > 3) act = 0;

    // Player movement
    if (act == 1) env->arkanoid.player_x -= ARKANOID_PLAYER_SPEED;
    else if (act == 2) env->arkanoid.player_x += ARKANOID_PLAYER_SPEED;
    float half_w = env->arkanoid.player_w * 0.5f;
    if (env->arkanoid.player_x - half_w < 0.0f) env->arkanoid.player_x = half_w;
    if (env->arkanoid.player_x + half_w > (float)ARKANOID_W) env->arkanoid.player_x = (float)ARKANOID_W - half_w;

    // Ball launch / follow paddle
    if (!env->arkanoid.ball_active) {
        if (act == 3) {
            env->arkanoid.ball_active = 1;
            env->arkanoid.ball_vx = 0.0f;
            env->arkanoid.ball_vy = -ARKANOID_BALL_SPEED;
        }
        env->arkanoid.ball_x = env->arkanoid.player_x;
        env->arkanoid.ball_y = env->arkanoid.player_y - env->arkanoid.player_h * 0.5f - env->arkanoid.ball_radius;
    } else {
        env->arkanoid.ball_x += env->arkanoid.ball_vx;
        env->arkanoid.ball_y += env->arkanoid.ball_vy;
    }

    const float brick_reward = 1.0f;
    const float life_penalty = -1.0f;
    const float clear_bonus = 10.0f;
    const float track_weight = 0.2f;
    double reward = 0.0;
    int done = 0;
    float r = env->arkanoid.ball_radius;

    for (int i = 0; i < 4; i++) {
        env->arkanoid.reward_vec[i] = 0.0f;
    }

    // Wall collisions
    if (env->arkanoid.ball_active) {
        if ((env->arkanoid.ball_x + r) >= (float)ARKANOID_W || (env->arkanoid.ball_x - r) <= 0.0f) {
            env->arkanoid.ball_vx *= -1.0f;
            env->arkanoid.ball_x = clamp_float(env->arkanoid.ball_x, r, (float)ARKANOID_W - r);
        }
        if ((env->arkanoid.ball_y - r) <= 0.0f) {
            env->arkanoid.ball_vy *= -1.0f;
            env->arkanoid.ball_y = r;
        } else if ((env->arkanoid.ball_y + r) >= (float)ARKANOID_H) {
            env->arkanoid.ball_vx = 0.0f;
            env->arkanoid.ball_vy = 0.0f;
            env->arkanoid.ball_active = 0;
            env->arkanoid.life -= 1;
            env->arkanoid.reward_vec[1] += life_penalty;
            done = 1;
        }
    }

    // Paddle collision (only when ball moving downward)
    if (env->arkanoid.ball_active && env->arkanoid.ball_vy > 0.0f) {
        float rx = env->arkanoid.player_x - env->arkanoid.player_w * 0.5f;
        float ry = env->arkanoid.player_y - env->arkanoid.player_h * 0.5f;
        if (circle_rect_collision(env->arkanoid.ball_x, env->arkanoid.ball_y, r, rx, ry,
                                  env->arkanoid.player_w, env->arkanoid.player_h)) {
            env->arkanoid.ball_vy = -fabsf(env->arkanoid.ball_vy);
            float rel = (env->arkanoid.ball_x - env->arkanoid.player_x) / half_w;
            env->arkanoid.ball_vx = clamp_float(rel * ARKANOID_BALL_SPEED, -ARKANOID_BALL_SPEED, ARKANOID_BALL_SPEED);
        }
    }

    // Brick collisions
    if (env->arkanoid.ball_active) {
        float bw = env->arkanoid.brick_w;
        float bh = env->arkanoid.brick_h;
        for (int i = 0; i < ARKANOID_LINES; i++) {
            for (int j = 0; j < ARKANOID_BRICKS_PER_LINE; j++) {
                if (!env->arkanoid.bricks[i][j]) continue;
                float bx = (float)j * bw + bw * 0.5f;
                float by = (float)i * bh + ARKANOID_BRICK_OFFSET_Y;
                int hit = 0;
                if (((env->arkanoid.ball_y - r) <= (by + bh * 0.5f)) &&
                    ((env->arkanoid.ball_y - r) > (by + bh * 0.5f + env->arkanoid.ball_vy)) &&
                    (fabsf(env->arkanoid.ball_x - bx) < (bw * 0.5f + r * 0.6667f)) && (env->arkanoid.ball_vy < 0.0f)) {
                    env->arkanoid.ball_vy *= -1.0f;
                    hit = 1;
                } else if (((env->arkanoid.ball_y + r) >= (by - bh * 0.5f)) &&
                           ((env->arkanoid.ball_y + r) < (by - bh * 0.5f + env->arkanoid.ball_vy)) &&
                           (fabsf(env->arkanoid.ball_x - bx) < (bw * 0.5f + r * 0.6667f)) && (env->arkanoid.ball_vy > 0.0f)) {
                    env->arkanoid.ball_vy *= -1.0f;
                    hit = 1;
                } else if (((env->arkanoid.ball_x + r) >= (bx - bw * 0.5f)) &&
                           ((env->arkanoid.ball_x + r) < (bx - bw * 0.5f + env->arkanoid.ball_vx)) &&
                           (fabsf(env->arkanoid.ball_y - by) < (bh * 0.5f + r * 0.6667f)) && (env->arkanoid.ball_vx > 0.0f)) {
                    env->arkanoid.ball_vx *= -1.0f;
                    hit = 1;
                } else if (((env->arkanoid.ball_x - r) <= (bx + bw * 0.5f)) &&
                           ((env->arkanoid.ball_x - r) > (bx + bw * 0.5f + env->arkanoid.ball_vx)) &&
                           (fabsf(env->arkanoid.ball_y - by) < (bh * 0.5f + r * 0.6667f)) && (env->arkanoid.ball_vx < 0.0f)) {
                    env->arkanoid.ball_vx *= -1.0f;
                    hit = 1;
                }
                if (hit) {
                    env->arkanoid.bricks[i][j] = 0;
                    env->arkanoid.bricks_left -= 1;
                    env->arkanoid.reward_vec[0] += brick_reward;
                }
            }
        }
    }

    if (env->arkanoid.bricks_left <= 0 && !done) {
        done = 1;
        env->arkanoid.reward_vec[2] += clear_bonus;
    }

    if (env->arkanoid.ball_active && env->arkanoid.ball_vy > 0.0f) {
        float half_w = env->arkanoid.player_w * 0.5f;
        if (half_w > 0.0f) {
            float target_x = arkanoid_predict_landing_x(env);
            float rel = (target_x - env->arkanoid.player_x) / half_w;
            rel = clamp_float(rel, -1.0f, 1.0f);
            float align = 1.0f - fabsf(rel);
            env->arkanoid.reward_vec[3] += track_weight * align;
            env->arkanoid.prev_phi = align;
        }
    } else {
        env->arkanoid.prev_phi = 0.0f;
    }
    reward = env->arkanoid.reward_vec[0] + env->arkanoid.reward_vec[1] +
             env->arkanoid.reward_vec[2] + env->arkanoid.reward_vec[3];

    env->steps += 1;
    if (env->steps >= ARKANOID_MAX_STEPS) done = 1;

    tfrl_step_result out = (tfrl_step_result){0};
    arkanoid_fill_obs(env, &out.observation);
    out.reward = reward;
    out.done = done;
    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}

tfrl_step_result tfrl_env_step_arkanoid_disc(tfrl_env *env, tfrl_action action) {
    tfrl_step_result out = tfrl_env_step_arkanoid(env, action);
    out.observation.index = arkanoid_hash_obs(env);
    out.observation.data_len = 0;
    return out;
}
