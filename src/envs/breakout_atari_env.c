#include "envs/envs_internal.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#define BREAKOUT_ATARI_PLAYER_SPEED (BREAKOUT_ATARI_PLAYER_SPEED_BASE / (float)BREAKOUT_ATARI_ACTION_REPEAT)

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

static int circle_rect_collision(float cx, float cy, float r,
                                 float rx, float ry, float rw, float rh) {
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

static void breakout_atari_predict_intercept(const tfrl_env *env, float *x_out, float *t_norm_out) {
    float intercept_x = env->breakout_atari.ball_x;
    float t_norm = 0.0f;

    if (env->breakout_atari.ball_active && env->breakout_atari.ball_vy > 0.0f) {
        float r = env->breakout_atari.ball_radius;
        float paddle_y = env->breakout_atari.player_y - env->breakout_atari.player_h * 0.5f - r;
        float dy = paddle_y - env->breakout_atari.ball_y;
        if (dy > 0.0f) {
            float t = dy / env->breakout_atari.ball_vy;
            float x = env->breakout_atari.ball_x + env->breakout_atari.ball_vx * t;
            intercept_x = reflect_x(x, r, (float)BREAKOUT_ATARI_W - r);
            float max_t = (float)BREAKOUT_ATARI_H / BREAKOUT_ATARI_BALL_SPEED;
            if (max_t > 1e-6f) t_norm = clamp_float(t / max_t, 0.0f, 1.0f);
        }
    }

    if (x_out) *x_out = intercept_x;
    if (t_norm_out) *t_norm_out = t_norm;
}

static float breakout_atari_phi_from_distance(float paddle_x, float intercept_x, float k) {
    float denom = (float)BREAKOUT_ATARI_W * 0.5f;
    if (denom <= 1e-6f) return 0.0f;
    float d = fabsf(paddle_x - intercept_x) / denom;
    d = clamp_float(d, 0.0f, 1.0f);
    return -k * d;
}

static const tfrl_obs_field BREAKOUT_ATARI_OBS_FIELDS[] = {
    {.name = "paddle_x", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "ball_pos", .len = 2, .dims = 1, .shape = {2, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "ball_vel", .len = 2, .dims = 1, .shape = {2, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "dx", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "intercept_x", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "time_to_paddle", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "bricks_remaining", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "bricks_bitmask", .len = BREAKOUT_ATARI_BRICK_OBS, .dims = 1,
     .shape = {BREAKOUT_ATARI_BRICK_OBS, 0}, .dtype = TFRL_DTYPE_FLOAT32},
};

static const tfrl_obs_layout BREAKOUT_ATARI_OBS_LAYOUT = {
    .field_count = (int)(sizeof(BREAKOUT_ATARI_OBS_FIELDS) / sizeof(BREAKOUT_ATARI_OBS_FIELDS[0])),
    .fields = BREAKOUT_ATARI_OBS_FIELDS,
};

static int breakout_atari_quantize(float v, float max, int bins) {
    if (max <= 0.0f) return 0;
    float t = v / max;
    if (t < 0.0f) t = 0.0f;
    if (t > 0.9999f) t = 0.9999f;
    int b = (int)(t * (float)bins);
    return clamp_int(b, 0, bins - 1);
}

static int breakout_atari_hash_obs(const tfrl_env *env) {
    const int pos_bins = 16;
    const int bricks_bins = 16;
    const int intercept_bins = 16;

    int px = breakout_atari_quantize(env->breakout_atari.player_x, (float)BREAKOUT_ATARI_W, pos_bins);
    int bx = breakout_atari_quantize(env->breakout_atari.ball_x, (float)BREAKOUT_ATARI_W, pos_bins);
    int by = breakout_atari_quantize(env->breakout_atari.ball_y, (float)BREAKOUT_ATARI_H, pos_bins);
    int vx = (env->breakout_atari.ball_vx > 0.5f) ? 2 : (env->breakout_atari.ball_vx < -0.5f ? 0 : 1);
    int vy = (env->breakout_atari.ball_vy > 0.5f) ? 2 : (env->breakout_atari.ball_vy < -0.5f ? 0 : 1);
    int active = env->breakout_atari.ball_active ? 1 : 0;
    int life = clamp_int(env->breakout_atari.life, 0, BREAKOUT_ATARI_PLAYER_MAX_LIFE);
    int bricks = breakout_atari_quantize((float)env->breakout_atari.bricks_left,
                                         (float)(BREAKOUT_ATARI_LINES * BREAKOUT_ATARI_BRICKS_PER_LINE),
                                         bricks_bins);
    float intercept_x = env->breakout_atari.ball_x;
    breakout_atari_predict_intercept(env, &intercept_x, NULL);
    int ix = breakout_atari_quantize(intercept_x, (float)BREAKOUT_ATARI_W, intercept_bins);

    uint32_t h = 2166136261u;
    h ^= (uint32_t)px; h *= 16777619u;
    h ^= (uint32_t)bx; h *= 16777619u;
    h ^= (uint32_t)by; h *= 16777619u;
    h ^= (uint32_t)vx; h *= 16777619u;
    h ^= (uint32_t)vy; h *= 16777619u;
    h ^= (uint32_t)active; h *= 16777619u;
    h ^= (uint32_t)life; h *= 16777619u;
    h ^= (uint32_t)bricks; h *= 16777619u;
    h ^= (uint32_t)ix; h *= 16777619u;
    return (int)(h % (uint32_t)BREAKOUT_ATARI_DISC_OBS_N);
}

static const tfrl_env_spec BREAKOUT_ATARI_SPEC = {
    .name = "breakout_atari",
    .obs_n = BREAKOUT_ATARI_OBS_DIMS,
    .action_n = 3,
    .max_steps = BREAKOUT_ATARI_MAX_STEPS,
    .width = BREAKOUT_ATARI_W,
    .height = BREAKOUT_ATARI_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = BREAKOUT_ATARI_OBS_DIMS,
    .action_dims = 1,
    .obs_shape = {BREAKOUT_ATARI_OBS_DIMS, 0},
    .action_shape = {3, 0},
    .obs_dtype = TFRL_DTYPE_FLOAT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = 1.0,
    .action_low = 0.0,
    .action_high = 2.0,
    .agent_count = 1,
    .obs_layout = &BREAKOUT_ATARI_OBS_LAYOUT,
};

static const tfrl_env_spec BREAKOUT_ATARI_DISC_SPEC = {
    .name = "breakout_atari_disc",
    .obs_n = BREAKOUT_ATARI_DISC_OBS_N,
    .action_n = 3,
    .max_steps = BREAKOUT_ATARI_MAX_STEPS,
    .width = BREAKOUT_ATARI_W,
    .height = BREAKOUT_ATARI_H,
    .obs_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {BREAKOUT_ATARI_DISC_OBS_N, 0},
    .action_shape = {3, 0},
    .obs_dtype = TFRL_DTYPE_INT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = (double)(BREAKOUT_ATARI_DISC_OBS_N - 1),
    .action_low = 0.0,
    .action_high = 2.0,
    .agent_count = 1,
    .obs_layout = NULL,
};

const tfrl_env_spec *tfrl_env_spec_breakout_atari(void) { return &BREAKOUT_ATARI_SPEC; }
const tfrl_env_spec *tfrl_env_spec_breakout_atari_disc(void) { return &BREAKOUT_ATARI_DISC_SPEC; }

static void breakout_atari_fill_obs(const tfrl_env *env, tfrl_obs *obs) {
    if (!obs) return;
    float vx = clamp_float(env->breakout_atari.ball_vx, -BREAKOUT_ATARI_BALL_SPEED, BREAKOUT_ATARI_BALL_SPEED);
    float vy = clamp_float(env->breakout_atari.ball_vy, -BREAKOUT_ATARI_BALL_SPEED, BREAKOUT_ATARI_BALL_SPEED);

    float intercept_x = env->breakout_atari.ball_x;
    float time_to_paddle = 0.0f;
    breakout_atari_predict_intercept(env, &intercept_x, &time_to_paddle);

    float dx = (env->breakout_atari.ball_x - env->breakout_atari.player_x) / ((float)BREAKOUT_ATARI_W * 0.5f);
    dx = clamp_float(dx, -1.0f, 1.0f);

    float paddle_x[1] = {env->breakout_atari.player_x / (float)BREAKOUT_ATARI_W};
    float ball_pos[2] = {env->breakout_atari.ball_x / (float)BREAKOUT_ATARI_W,
                         env->breakout_atari.ball_y / (float)BREAKOUT_ATARI_H};
    float ball_vel[2] = {(vx / BREAKOUT_ATARI_BALL_SPEED + 1.0f) * 0.5f,
                         (vy / BREAKOUT_ATARI_BALL_SPEED + 1.0f) * 0.5f};
    float dx_arr[1] = {(dx + 1.0f) * 0.5f};
    float intercept_arr[1] = {intercept_x / (float)BREAKOUT_ATARI_W};
    float time_arr[1] = {time_to_paddle};
    float bricks_remaining[1] = {(float)env->breakout_atari.bricks_left /
                                 (float)(BREAKOUT_ATARI_LINES * BREAKOUT_ATARI_BRICKS_PER_LINE)};

    float bricks_bitmask[BREAKOUT_ATARI_BRICK_OBS];
    int idx = 0;
    for (int i = 0; i < BREAKOUT_ATARI_LINES; i++) {
        for (int j = 0; j < BREAKOUT_ATARI_BRICKS_PER_LINE; j++) {
            bricks_bitmask[idx++] = env->breakout_atari.bricks[i][j] ? 1.0f : 0.0f;
        }
    }

    const float *fields[] = {
        paddle_x, ball_pos, ball_vel, dx_arr, intercept_arr, time_arr, bricks_remaining,
        bricks_bitmask
    };

    if (!tfrl_obs_flatten(&BREAKOUT_ATARI_OBS_LAYOUT, fields, obs)) {
        obs->data_len = 0;
    }
}

tfrl_obs tfrl_env_reset_breakout_atari(tfrl_env *env, uint64_t seed) {
    tfrl_env_reset_state(env);
    env->rng_state = (uint32_t)(seed ? seed : 1u);

    env->breakout_atari.player_w = (float)BREAKOUT_ATARI_W / 6.0f;
    env->breakout_atari.player_h = 8.0f;
    env->breakout_atari.player_x = (float)BREAKOUT_ATARI_W / 2.0f;
    env->breakout_atari.player_y = (float)BREAKOUT_ATARI_H * 9.0f / 10.0f;
    env->breakout_atari.life = BREAKOUT_ATARI_PLAYER_MAX_LIFE;
    env->breakout_atari.score = 0.0f;

    env->breakout_atari.ball_radius = BREAKOUT_ATARI_BALL_RADIUS;
    env->breakout_atari.ball_active = 1;

    const float pi = 3.14159265358979323846f;
    float angle = (20.0f + 140.0f * tfrl_env_rand_uniform(env)) * pi / 180.0f;
    env->breakout_atari.ball_vx = BREAKOUT_ATARI_BALL_SPEED * cosf(angle);
    env->breakout_atari.ball_vy = -BREAKOUT_ATARI_BALL_SPEED * sinf(angle);
    env->breakout_atari.ball_x = env->breakout_atari.player_x;
    env->breakout_atari.ball_y =
        env->breakout_atari.player_y - env->breakout_atari.player_h * 0.5f - env->breakout_atari.ball_radius;

    env->breakout_atari.brick_w = (float)BREAKOUT_ATARI_W / (float)BREAKOUT_ATARI_BRICKS_PER_LINE;
    env->breakout_atari.brick_h = BREAKOUT_ATARI_BRICK_HEIGHT;

    env->breakout_atari.bricks_left = 0;
    for (int i = 0; i < BREAKOUT_ATARI_LINES; i++) {
        for (int j = 0; j < BREAKOUT_ATARI_BRICKS_PER_LINE; j++) {
            env->breakout_atari.bricks[i][j] = 1;
            env->breakout_atari.bricks_left++;
        }
    }

    tfrl_obs obs = {0};
    breakout_atari_fill_obs(env, &obs);
    return obs;
}

tfrl_obs tfrl_env_reset_breakout_atari_disc(tfrl_env *env, uint64_t seed) {
    tfrl_obs obs = tfrl_env_reset_breakout_atari(env, seed);
    obs.index = breakout_atari_hash_obs(env);
    obs.data_len = 0;
    return obs;
}

static tfrl_step_result breakout_atari_step_frame(tfrl_env *env, int act) {
    const float gamma = 0.99f;
    const float shaping_k = 0.04f;

    float paddle_x_prev = env->breakout_atari.player_x;
    float intercept_prev = env->breakout_atari.ball_x;
    breakout_atari_predict_intercept(env, &intercept_prev, NULL);
    float phi_prev = 0.0f;
    if (env->breakout_atari.ball_active && env->breakout_atari.ball_vy > 0.0f) {
        phi_prev = breakout_atari_phi_from_distance(paddle_x_prev, intercept_prev, shaping_k);
    }

    if (act == 1) env->breakout_atari.player_x -= BREAKOUT_ATARI_PLAYER_SPEED;
    else if (act == 2) env->breakout_atari.player_x += BREAKOUT_ATARI_PLAYER_SPEED;

    float half_w = env->breakout_atari.player_w * 0.5f;
    if (env->breakout_atari.player_x - half_w < 0.0f) env->breakout_atari.player_x = half_w;
    if (env->breakout_atari.player_x + half_w > (float)BREAKOUT_ATARI_W) {
        env->breakout_atari.player_x = (float)BREAKOUT_ATARI_W - half_w;
    }

    if (!env->breakout_atari.ball_active) {
        env->breakout_atari.ball_active = 1;
        env->breakout_atari.ball_vx = 0.0f;
        env->breakout_atari.ball_vy = -BREAKOUT_ATARI_BALL_SPEED;
        env->breakout_atari.ball_x = env->breakout_atari.player_x;
        env->breakout_atari.ball_y =
            env->breakout_atari.player_y - env->breakout_atari.player_h * 0.5f - env->breakout_atari.ball_radius;
    } else {
        env->breakout_atari.ball_x += env->breakout_atari.ball_vx;
        env->breakout_atari.ball_y += env->breakout_atari.ball_vy;
    }

    const float brick_reward = 1.0f;
    const float paddle_reward = 0.5f;
    const float life_penalty = -10.0f;
    const float clear_bonus = 10.0f;
    const float move_penalty = -0.0005f / (float)BREAKOUT_ATARI_ACTION_REPEAT;
    const float step_penalty = -0.002f / (float)BREAKOUT_ATARI_ACTION_REPEAT;

    double reward = 0.0;
    int done = 0;

    float r = env->breakout_atari.ball_radius;
    const float push_eps = 0.25f;

    if (act == 1 || act == 2) reward += move_penalty;

    if (env->breakout_atari.ball_active) {
        if ((env->breakout_atari.ball_x + r) >= (float)BREAKOUT_ATARI_W ||
            (env->breakout_atari.ball_x - r) <= 0.0f) {
            env->breakout_atari.ball_vx *= -1.0f;
            env->breakout_atari.ball_x = clamp_float(env->breakout_atari.ball_x, r, (float)BREAKOUT_ATARI_W - r);
        }

        if ((env->breakout_atari.ball_y - r) <= 0.0f) {
            env->breakout_atari.ball_vy *= -1.0f;
            env->breakout_atari.ball_y = r;

            if (fabsf(env->breakout_atari.ball_vx) < 0.2f) {
                env->breakout_atari.ball_vx =
                    (env->breakout_atari.ball_x < (float)BREAKOUT_ATARI_W * 0.5f) ? 1.0f : -1.0f;
            }
        } else if ((env->breakout_atari.ball_y + r) >= (float)BREAKOUT_ATARI_H) {
            env->breakout_atari.ball_vx = 0.0f;
            env->breakout_atari.ball_vy = 0.0f;
            env->breakout_atari.ball_active = 0;
            env->breakout_atari.life -= 1;
            reward += life_penalty;
            if (env->breakout_atari.life <= 0) done = 1;
        }
    }

    if (env->breakout_atari.ball_active && env->breakout_atari.ball_vy > 0.0f) {
        float rx = env->breakout_atari.player_x - env->breakout_atari.player_w * 0.5f;
        float ry = env->breakout_atari.player_y - env->breakout_atari.player_h * 0.5f;

        if (circle_rect_collision(env->breakout_atari.ball_x, env->breakout_atari.ball_y, r, rx, ry,
                                  env->breakout_atari.player_w, env->breakout_atari.player_h)) {
            env->breakout_atari.ball_vy = -fabsf(env->breakout_atari.ball_vy);

            float rel = (env->breakout_atari.ball_x - env->breakout_atari.player_x) / half_w;
            env->breakout_atari.ball_vx =
                clamp_float(rel * BREAKOUT_ATARI_BALL_SPEED, -BREAKOUT_ATARI_BALL_SPEED, BREAKOUT_ATARI_BALL_SPEED);

            if (fabsf(env->breakout_atari.ball_vx) < 0.2f) {
                env->breakout_atari.ball_vx = (rel < 0.0f) ? -0.2f : 0.2f;
            }
            reward += paddle_reward;
        }
    }

    if (env->breakout_atari.ball_active) {
        float bw = env->breakout_atari.brick_w;
        float bh = env->breakout_atari.brick_h;

        int brick_hit = 0;
        for (int i = 0; i < BREAKOUT_ATARI_LINES && !brick_hit; i++) {
            for (int j = 0; j < BREAKOUT_ATARI_BRICKS_PER_LINE; j++) {
                if (!env->breakout_atari.bricks[i][j]) continue;

                float bx = (float)j * bw + bw * 0.5f;
                float by = (float)i * bh + BREAKOUT_ATARI_BRICK_OFFSET_Y;

                int hit = 0;

                if (((env->breakout_atari.ball_y - r) <= (by + bh * 0.5f)) &&
                    ((env->breakout_atari.ball_y - r) > (by + bh * 0.5f + env->breakout_atari.ball_vy)) &&
                    (fabsf(env->breakout_atari.ball_x - bx) < (bw * 0.5f + r * 0.6667f)) &&
                    (env->breakout_atari.ball_vy < 0.0f)) {
                    env->breakout_atari.ball_vy *= -1.0f;
                    env->breakout_atari.ball_y = by + bh * 0.5f + r + push_eps;
                    hit = 1;
                } else if (((env->breakout_atari.ball_y + r) >= (by - bh * 0.5f)) &&
                           ((env->breakout_atari.ball_y + r) <
                            (by - bh * 0.5f + env->breakout_atari.ball_vy)) &&
                           (fabsf(env->breakout_atari.ball_x - bx) < (bw * 0.5f + r * 0.6667f)) &&
                           (env->breakout_atari.ball_vy > 0.0f)) {
                    env->breakout_atari.ball_vy *= -1.0f;
                    env->breakout_atari.ball_y = by - bh * 0.5f - r - push_eps;
                    hit = 1;
                } else if (((env->breakout_atari.ball_x + r) >= (bx - bw * 0.5f)) &&
                           ((env->breakout_atari.ball_x + r) <
                            (bx - bw * 0.5f + env->breakout_atari.ball_vx)) &&
                           (fabsf(env->breakout_atari.ball_y - by) < (bh * 0.5f + r * 0.6667f)) &&
                           (env->breakout_atari.ball_vx > 0.0f)) {
                    env->breakout_atari.ball_vx *= -1.0f;
                    env->breakout_atari.ball_x = bx - bw * 0.5f - r - push_eps;
                    hit = 1;
                } else if (((env->breakout_atari.ball_x - r) <= (bx + bw * 0.5f)) &&
                           ((env->breakout_atari.ball_x - r) >
                            (bx + bw * 0.5f + env->breakout_atari.ball_vx)) &&
                           (fabsf(env->breakout_atari.ball_y - by) < (bh * 0.5f + r * 0.6667f)) &&
                           (env->breakout_atari.ball_vx < 0.0f)) {
                    env->breakout_atari.ball_vx *= -1.0f;
                    env->breakout_atari.ball_x = bx + bw * 0.5f + r + push_eps;
                    hit = 1;
                }

                if (hit) {
                    env->breakout_atari.bricks[i][j] = 0;
                    env->breakout_atari.bricks_left -= 1;
                    reward += brick_reward;
                    env->breakout_atari.score += 1.0f;
                    brick_hit = 1;
                    break;
                }
            }
        }
    }

    if (env->breakout_atari.bricks_left <= 0 && !done) {
        done = 1;
        reward += clear_bonus;
    }

    float intercept_next = env->breakout_atari.ball_x;
    breakout_atari_predict_intercept(env, &intercept_next, NULL);
    float phi_next = 0.0f;
    if (env->breakout_atari.ball_active && env->breakout_atari.ball_vy > 0.0f) {
        phi_next = breakout_atari_phi_from_distance(env->breakout_atari.player_x, intercept_next, shaping_k);
    }
    reward += gamma * phi_next - phi_prev;

    reward += step_penalty;

    tfrl_step_result out = {0};
    breakout_atari_fill_obs(env, &out.observation);
    out.reward = reward;
    out.done = done;

    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}

tfrl_step_result tfrl_env_step_breakout_atari(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    if (act < 0 || act > 2) act = 0;

    tfrl_step_result out = {0};
    double reward_sum = 0.0;
    int done = 0;

    for (int i = 0; i < BREAKOUT_ATARI_ACTION_REPEAT; i++) {
        tfrl_step_result step = breakout_atari_step_frame(env, act);
        reward_sum += step.reward;

        env->steps += 1;
        if (env->steps >= BREAKOUT_ATARI_MAX_STEPS) done = 1;

        out = step;
        if (step.done || done) {
            done = 1;
            break;
        }
    }

    out.reward = reward_sum;
    out.done = done;
    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}

tfrl_step_result tfrl_env_step_breakout_atari_disc(tfrl_env *env, tfrl_action action) {
    tfrl_step_result out = tfrl_env_step_breakout_atari(env, action);
    out.observation.index = breakout_atari_hash_obs(env);
    out.observation.data_len = 0;
    return out;
}
