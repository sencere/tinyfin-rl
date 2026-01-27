#include "envs/envs_internal.h"

#include <math.h>
#include <stdint.h>

#define BREAKOUT_MAX_STEPS 6000
#define BREAKOUT_PLAYER_MAX_LIFE 1
#define BREAKOUT_LINES 4
#define BREAKOUT_BRICKS_PER_LINE 8
#define BREAKOUT_BRICK_FEATURES (BREAKOUT_BRICKS_PER_LINE + 1)
#define BREAKOUT_OBS_DIMS (12 + BREAKOUT_BRICK_FEATURES)
#define BREAKOUT_BRICK_HEIGHT 28.0f
#define BREAKOUT_BRICK_OFFSET_Y 40.0f
#define BREAKOUT_PLAYER_SPEED 9.0f
#define BREAKOUT_BALL_SPEED 4.0f
#define BREAKOUT_BALL_RADIUS 6.0f
#define BREAKOUT_ACTION_REPEAT 4

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

static int breakout_estimate_velocity(const tfrl_env *env, float *vx, float *vy) {
    if (!env->arkanoid.prev_ball_valid) return 0;
    *vx = env->arkanoid.ball_x - env->arkanoid.prev_ball_x;
    *vy = env->arkanoid.ball_y - env->arkanoid.prev_ball_y;
    return (fabsf(*vx) + fabsf(*vy)) > 1e-6f;
}

static void breakout_predict_landing(tfrl_env *env, float *x_out, float *t_out) {
    float vx = env->arkanoid.ball_vx;
    float vy = env->arkanoid.ball_vy;
    float est_vx = 0.0f;
    float est_vy = 0.0f;
    if (breakout_estimate_velocity(env, &est_vx, &est_vy)) {
        vx = est_vx;
        vy = est_vy;
    }

    float raw_x = env->arkanoid.ball_x;
    float raw_t = 0.0f;
    if (vy > 0.0f) {
        float r = env->arkanoid.ball_radius;
        float paddle_y = env->arkanoid.player_y - env->arkanoid.player_h * 0.5f - r;
        float dy = paddle_y - env->arkanoid.ball_y;
        if (dy > 0.0f) {
            float t = dy / vy;
            float max_t = (float)ARKANOID_H / BREAKOUT_BALL_SPEED;
            if (max_t > 0.0f) {
                raw_t = t / max_t;
                if (raw_t < 0.0f) raw_t = 0.0f;
                if (raw_t > 1.0f) raw_t = 1.0f;
            }
            float x = env->arkanoid.ball_x + vx * t;
            float min_x = r;
            float max_x = (float)ARKANOID_W - r;
            raw_x = reflect_x(x, min_x, max_x);
        }
    }

    const float alpha = 0.25f;
    if (!env->arkanoid.pred_ema_valid) {
        env->arkanoid.pred_x_ema = raw_x;
        env->arkanoid.pred_t_ema = raw_t;
        env->arkanoid.pred_ema_valid = 1;
    } else {
        env->arkanoid.pred_x_ema += alpha * (raw_x - env->arkanoid.pred_x_ema);
        env->arkanoid.pred_t_ema += alpha * (raw_t - env->arkanoid.pred_t_ema);
    }

    *x_out = env->arkanoid.pred_x_ema;
    *t_out = env->arkanoid.pred_t_ema;
}

static float breakout_phi_align_pred(const tfrl_env *env, float target_x) {
    float hw = env->arkanoid.player_w * 0.5f;
    if (hw <= 1e-6f) return 0.0f;

    float rel = (target_x - env->arkanoid.player_x) / hw;
    rel = clamp_float(rel, -1.0f, 1.0f);
    return -fabsf(rel);
}

static float breakout_urgency(float tn) {
    float u = 1.0f - tn;
    return clamp_float(0.25f + 0.75f * u, 0.25f, 1.0f);
}

static const tfrl_obs_field BREAKOUT_OBS_FIELDS[] = {
    {.name = "player_x", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "ball_pos", .len = 2, .dims = 1, .shape = {2, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "ball_vel", .len = 2, .dims = 1, .shape = {2, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "bricks_left", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "ball_active", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "rel_ball_x", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "life", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "landing_x", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "landing_t", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "score", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "brick_counts", .len = BREAKOUT_BRICK_FEATURES, .dims = 1,
     .shape = {BREAKOUT_BRICK_FEATURES, 0}, .dtype = TFRL_DTYPE_FLOAT32},
};

static const tfrl_obs_layout BREAKOUT_OBS_LAYOUT = {
    .field_count = (int)(sizeof(BREAKOUT_OBS_FIELDS) / sizeof(BREAKOUT_OBS_FIELDS[0])),
    .fields = BREAKOUT_OBS_FIELDS,
};

static int breakout_quantize(float v, float max, int bins) {
    if (max <= 0.0f) return 0;
    float t = v / max;
    if (t < 0.0f) t = 0.0f;
    if (t > 0.9999f) t = 0.9999f;
    int b = (int)(t * (float)bins);
    return clamp_int(b, 0, bins - 1);
}

static int breakout_hash_obs(const tfrl_env *env) {
    const int pos_bins = 16;
    const int bricks_bins = 16;

    int px = breakout_quantize(env->arkanoid.player_x, (float)ARKANOID_W, pos_bins);
    int bx = breakout_quantize(env->arkanoid.ball_x, (float)ARKANOID_W, pos_bins);
    int by = breakout_quantize(env->arkanoid.ball_y, (float)ARKANOID_H, pos_bins);
    int vx = (env->arkanoid.ball_vx > 0.5f) ? 2 : (env->arkanoid.ball_vx < -0.5f ? 0 : 1);
    int vy = (env->arkanoid.ball_vy > 0.5f) ? 2 : (env->arkanoid.ball_vy < -0.5f ? 0 : 1);
    int active = env->arkanoid.ball_active ? 1 : 0;
    int life = clamp_int(env->arkanoid.life, 0, BREAKOUT_PLAYER_MAX_LIFE);
    int bricks = breakout_quantize((float)env->arkanoid.bricks_left,
                                   (float)(BREAKOUT_LINES * BREAKOUT_BRICKS_PER_LINE),
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

static const tfrl_env_spec BREAKOUT_SPEC = {
    .name = "breakout",
    .obs_n = BREAKOUT_OBS_DIMS,
    .action_n = 3,
    .max_steps = BREAKOUT_MAX_STEPS,
    .width = ARKANOID_W,
    .height = ARKANOID_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = BREAKOUT_OBS_DIMS,
    .action_dims = 1,
    .obs_shape = {BREAKOUT_OBS_DIMS, 0},
    .action_shape = {3, 0},
    .obs_dtype = TFRL_DTYPE_FLOAT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = 1.0,
    .action_low = 0.0,
    .action_high = 2.0,
    .agent_count = 1,
    .obs_layout = &BREAKOUT_OBS_LAYOUT,
};

static const tfrl_env_spec BREAKOUT_DISC_SPEC = {
    .name = "breakout_disc",
    .obs_n = ARKANOID_DISC_OBS_N,
    .action_n = 3,
    .max_steps = BREAKOUT_MAX_STEPS,
    .width = ARKANOID_W,
    .height = ARKANOID_H,
    .obs_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {ARKANOID_DISC_OBS_N, 0},
    .action_shape = {3, 0},
    .obs_dtype = TFRL_DTYPE_INT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = (double)(ARKANOID_DISC_OBS_N - 1),
    .action_low = 0.0,
    .action_high = 2.0,
    .agent_count = 1,
    .obs_layout = NULL,
};

const tfrl_env_spec *tfrl_env_spec_breakout(void) {
    return &BREAKOUT_SPEC;
}

const tfrl_env_spec *tfrl_env_spec_breakout_disc(void) {
    return &BREAKOUT_DISC_SPEC;
}

static void breakout_fill_obs(const tfrl_env *env, tfrl_obs *obs) {
    if (!obs) return;

    float vx = clamp_float(env->arkanoid.ball_vx, -BREAKOUT_BALL_SPEED, BREAKOUT_BALL_SPEED);
    float vy = clamp_float(env->arkanoid.ball_vy, -BREAKOUT_BALL_SPEED, BREAKOUT_BALL_SPEED);

    float rel_x = (env->arkanoid.ball_x - env->arkanoid.player_x) / ((float)ARKANOID_W * 0.5f);
    rel_x = clamp_float(rel_x, -1.0f, 1.0f);

    float landing_x = env->arkanoid.pred_ema_valid ? env->arkanoid.pred_x_ema : env->arkanoid.ball_x;
    float landing_t = env->arkanoid.pred_ema_valid ? env->arkanoid.pred_t_ema : 0.0f;

    float player_x[1] = {env->arkanoid.player_x / (float)ARKANOID_W};
    float ball_pos[2] = {env->arkanoid.ball_x / (float)ARKANOID_W,
                         env->arkanoid.ball_y / (float)ARKANOID_H};
    float ball_vel[2] = {(vx / BREAKOUT_BALL_SPEED + 1.0f) * 0.5f,
                         (vy / BREAKOUT_BALL_SPEED + 1.0f) * 0.5f};
    float bricks_left[1] = {(float)env->arkanoid.bricks_left /
                            (float)(BREAKOUT_LINES * BREAKOUT_BRICKS_PER_LINE)};
    float ball_active[1] = {env->arkanoid.ball_active ? 1.0f : 0.0f};
    float rel_ball_x[1] = {(rel_x + 1.0f) * 0.5f};
    float life[1] = {(float)env->arkanoid.life / (float)BREAKOUT_PLAYER_MAX_LIFE};
    float landing[1] = {landing_x / (float)ARKANOID_W};
    float landing_t_arr[1] = {landing_t};
    float score[1] = {env->arkanoid.score / (float)(BREAKOUT_LINES * BREAKOUT_BRICKS_PER_LINE)};

    // Compact brick features: remaining bricks per column (normalized) + total remaining.
    float bricks[BREAKOUT_BRICK_FEATURES];
    for (int j = 0; j < BREAKOUT_BRICKS_PER_LINE; j++) {
        int count = 0;
        for (int i = 0; i < BREAKOUT_LINES; i++) {
            if (env->arkanoid.bricks[i][j]) count++;
        }
        bricks[j] = (float)count / (float)BREAKOUT_LINES;
    }
    bricks[BREAKOUT_BRICKS_PER_LINE] =
        (float)env->arkanoid.bricks_left / (float)(BREAKOUT_LINES * BREAKOUT_BRICKS_PER_LINE);

    const float *fields[] = {
        player_x, ball_pos, ball_vel, bricks_left, ball_active,
        rel_ball_x, life, landing, landing_t_arr, score, bricks
    };

    if (!tfrl_obs_flatten(&BREAKOUT_OBS_LAYOUT, fields, obs)) {
        obs->data_len = 0;
    }
}

static uint64_t arkanoid_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float arkanoid_rng_uniform(uint64_t *state) {
    return (float)(arkanoid_rng_next(state) & 0xFFFFFFu) / (float)0x1000000;
}

tfrl_obs tfrl_env_reset_breakout(tfrl_env *env, uint64_t seed) {
    tfrl_env_reset_state(env);

    env->arkanoid.player_w = (float)ARKANOID_W / 6.0f;
    env->arkanoid.player_h = 18.0f;
    env->arkanoid.player_x = (float)ARKANOID_W / 2.0f;
    env->arkanoid.player_y = (float)ARKANOID_H * 7.0f / 8.0f;
    env->arkanoid.life = BREAKOUT_PLAYER_MAX_LIFE;
    env->arkanoid.score = 0.0f;

    env->arkanoid.ball_radius = BREAKOUT_BALL_RADIUS;
    env->arkanoid.ball_active = 1;

    uint64_t rng = seed ? seed : 1u;
    const float pi = 3.14159265358979323846f;
    float angle = (30.0f + 120.0f * arkanoid_rng_uniform(&rng)) * pi / 180.0f;

    env->arkanoid.ball_vx = BREAKOUT_BALL_SPEED * cosf(angle);
    env->arkanoid.ball_vy = -BREAKOUT_BALL_SPEED * sinf(angle);
    env->arkanoid.ball_x = env->arkanoid.player_x;
    env->arkanoid.ball_y =
        env->arkanoid.player_y - env->arkanoid.player_h * 0.5f - env->arkanoid.ball_radius;
    env->arkanoid.prev_ball_x = env->arkanoid.ball_x;
    env->arkanoid.prev_ball_y = env->arkanoid.ball_y;
    env->arkanoid.prev_ball_valid = 0;
    env->arkanoid.pred_x_ema = 0.0f;
    env->arkanoid.pred_t_ema = 0.0f;
    env->arkanoid.pred_ema_valid = 0;

    env->arkanoid.brick_w = (float)ARKANOID_W / (float)BREAKOUT_BRICKS_PER_LINE;
    env->arkanoid.brick_h = BREAKOUT_BRICK_HEIGHT;

    env->arkanoid.bricks_left = 0;
    for (int i = 0; i < ARKANOID_LINES; i++) {
        for (int j = 0; j < ARKANOID_BRICKS_PER_LINE; j++) {
            env->arkanoid.bricks[i][j] = 0;
        }
    }
    for (int i = 0; i < BREAKOUT_LINES; i++) {
        for (int j = 0; j < BREAKOUT_BRICKS_PER_LINE; j++) {
            env->arkanoid.bricks[i][j] = 1;
            env->arkanoid.bricks_left++;
        }
    }
    env->arkanoid.prev_bricks_left = env->arkanoid.bricks_left;

    env->arkanoid.prev_phi = 0.0f;
    for (int i = 0; i < 5; i++) env->arkanoid.reward_vec[i] = 0.0f;
    for (int i = 0; i < BREAKOUT_REWARD_DELAY; i++) env->arkanoid.reward_delay[i] = 0.0f;
    env->arkanoid.reward_delay_idx = 0;
    env->arkanoid.reward_delay_filled = 0;

    tfrl_obs obs = {0};
    breakout_fill_obs(env, &obs);
    return obs;
}

tfrl_obs tfrl_env_reset_breakout_disc(tfrl_env *env, uint64_t seed) {
    tfrl_obs obs = tfrl_env_reset_breakout(env, seed);
    obs.index = breakout_hash_obs(env);
    obs.data_len = 0;
    return obs;
}

static tfrl_step_result breakout_step_frame(tfrl_env *env, int act) {
    if (act == 1) env->arkanoid.player_x -= BREAKOUT_PLAYER_SPEED;
    else if (act == 2) env->arkanoid.player_x += BREAKOUT_PLAYER_SPEED;

    float half_w = env->arkanoid.player_w * 0.5f;
    if (env->arkanoid.player_x - half_w < 0.0f) env->arkanoid.player_x = half_w;
    if (env->arkanoid.player_x + half_w > (float)ARKANOID_W) {
        env->arkanoid.player_x = (float)ARKANOID_W - half_w;
    }

    if (!env->arkanoid.ball_active) {
        env->arkanoid.ball_active = 1;
        env->arkanoid.ball_vx = 0.0f;
        env->arkanoid.ball_vy = -BREAKOUT_BALL_SPEED;
        env->arkanoid.ball_x = env->arkanoid.player_x;
        env->arkanoid.ball_y =
            env->arkanoid.player_y - env->arkanoid.player_h * 0.5f - env->arkanoid.ball_radius;
    } else {
        env->arkanoid.ball_x += env->arkanoid.ball_vx;
        env->arkanoid.ball_y += env->arkanoid.ball_vy;
    }

    const float brick_reward = 3.0f;
    const float paddle_reward = 0.05f;
    const float life_penalty = -2.0f;
    const float clear_bonus = 15.0f;
    const float shaping_k = 0.15f;
    const float phi_clamp = 0.15f;
    const float move_penalty = -0.00005f;
    const float step_penalty = -0.00005f;
    const float progress_k = 0.05f;
    const float center_k = 0.01f;
    const float center_phi_clamp = 0.10f;

    double reward = 0.0;
    int done = 0;

    float r = env->arkanoid.ball_radius;
    const float push_eps = 0.25f;

    for (int i = 0; i < 5; i++) env->arkanoid.reward_vec[i] = 0.0f;

    if (act == 1 || act == 2) env->arkanoid.reward_vec[4] += move_penalty;

    if (env->arkanoid.ball_active) {
        if ((env->arkanoid.ball_x + r) >= (float)ARKANOID_W ||
            (env->arkanoid.ball_x - r) <= 0.0f) {
            env->arkanoid.ball_vx *= -1.0f;
            env->arkanoid.ball_x = clamp_float(env->arkanoid.ball_x, r, (float)ARKANOID_W - r);
        }

        if ((env->arkanoid.ball_y - r) <= 0.0f) {
            env->arkanoid.ball_vy *= -1.0f;
            env->arkanoid.ball_y = r;

            if (fabsf(env->arkanoid.ball_vx) < 0.2f) {
                env->arkanoid.ball_vx =
                    (env->arkanoid.ball_x < (float)ARKANOID_W * 0.5f) ? 1.0f : -1.0f;
            }
        } else if ((env->arkanoid.ball_y + r) >= (float)ARKANOID_H) {
            env->arkanoid.ball_vx = 0.0f;
            env->arkanoid.ball_vy = 0.0f;
            env->arkanoid.ball_active = 0;
            env->arkanoid.life -= 1;
            env->arkanoid.reward_vec[1] += life_penalty;
            if (env->arkanoid.life <= 0) done = 1;
        }
    }

    if (env->arkanoid.ball_active && env->arkanoid.ball_vy > 0.0f) {
        float rx = env->arkanoid.player_x - env->arkanoid.player_w * 0.5f;
        float ry = env->arkanoid.player_y - env->arkanoid.player_h * 0.5f;

        if (circle_rect_collision(env->arkanoid.ball_x, env->arkanoid.ball_y, r, rx, ry,
                                  env->arkanoid.player_w, env->arkanoid.player_h)) {
            env->arkanoid.ball_vy = -fabsf(env->arkanoid.ball_vy);

            float rel = (env->arkanoid.ball_x - env->arkanoid.player_x) / half_w;
            env->arkanoid.ball_vx =
                clamp_float(rel * BREAKOUT_BALL_SPEED, -BREAKOUT_BALL_SPEED, BREAKOUT_BALL_SPEED);

            if (fabsf(env->arkanoid.ball_vx) < 0.2f) {
                env->arkanoid.ball_vx = (rel < 0.0f) ? -0.2f : 0.2f;
            }
            env->arkanoid.reward_vec[4] += paddle_reward;
        }
    }

    if (env->arkanoid.ball_active) {
        float bw = env->arkanoid.brick_w;
        float bh = env->arkanoid.brick_h;

        int brick_hit = 0;
        for (int i = 0; i < BREAKOUT_LINES && !brick_hit; i++) {
            for (int j = 0; j < BREAKOUT_BRICKS_PER_LINE; j++) {
                if (!env->arkanoid.bricks[i][j]) continue;

                float bx = (float)j * bw + bw * 0.5f;
                float by = (float)i * bh + BREAKOUT_BRICK_OFFSET_Y;

                int hit = 0;

                if (((env->arkanoid.ball_y - r) <= (by + bh * 0.5f)) &&
                    ((env->arkanoid.ball_y - r) > (by + bh * 0.5f + env->arkanoid.ball_vy)) &&
                    (fabsf(env->arkanoid.ball_x - bx) < (bw * 0.5f + r * 0.6667f)) &&
                    (env->arkanoid.ball_vy < 0.0f)) {
                    env->arkanoid.ball_vy *= -1.0f;
                    env->arkanoid.ball_y = by + bh * 0.5f + r + push_eps;
                    hit = 1;
                } else if (((env->arkanoid.ball_y + r) >= (by - bh * 0.5f)) &&
                           ((env->arkanoid.ball_y + r) <
                            (by - bh * 0.5f + env->arkanoid.ball_vy)) &&
                           (fabsf(env->arkanoid.ball_x - bx) < (bw * 0.5f + r * 0.6667f)) &&
                           (env->arkanoid.ball_vy > 0.0f)) {
                    env->arkanoid.ball_vy *= -1.0f;
                    env->arkanoid.ball_y = by - bh * 0.5f - r - push_eps;
                    hit = 1;
                } else if (((env->arkanoid.ball_x + r) >= (bx - bw * 0.5f)) &&
                           ((env->arkanoid.ball_x + r) <
                            (bx - bw * 0.5f + env->arkanoid.ball_vx)) &&
                           (fabsf(env->arkanoid.ball_y - by) < (bh * 0.5f + r * 0.6667f)) &&
                           (env->arkanoid.ball_vx > 0.0f)) {
                    env->arkanoid.ball_vx *= -1.0f;
                    env->arkanoid.ball_x = bx - bw * 0.5f - r - push_eps;
                    hit = 1;
                } else if (((env->arkanoid.ball_x - r) <= (bx + bw * 0.5f)) &&
                           ((env->arkanoid.ball_x - r) >
                            (bx + bw * 0.5f + env->arkanoid.ball_vx)) &&
                           (fabsf(env->arkanoid.ball_y - by) < (bh * 0.5f + r * 0.6667f)) &&
                           (env->arkanoid.ball_vx < 0.0f)) {
                    env->arkanoid.ball_vx *= -1.0f;
                    env->arkanoid.ball_x = bx + bw * 0.5f + r + push_eps;
                    hit = 1;
                }

                if (hit) {
                    env->arkanoid.bricks[i][j] = 0;
                    env->arkanoid.bricks_left -= 1;
                    env->arkanoid.reward_vec[0] += brick_reward;
                    env->arkanoid.score += 1.0f;
                    brick_hit = 1;
                    break;
                }
            }
        }
    }

    if (env->arkanoid.bricks_left <= 0 && !done) {
        done = 1;
        env->arkanoid.reward_vec[2] += clear_bonus;
    }

    if (env->arkanoid.ball_active && env->arkanoid.ball_vy > 0.5f) {
        float target_x = 0.0f;
        float landing_t = 0.0f;
        breakout_predict_landing(env, &target_x, &landing_t);
        if (landing_t < 0.95f) {
            float phi = breakout_phi_align_pred(env, target_x);
            float dphi = phi - env->arkanoid.prev_phi;
            float u = breakout_urgency(landing_t);
            float shaped = clamp_float(dphi, -phi_clamp, phi_clamp) * shaping_k * u;
            env->arkanoid.reward_vec[3] += shaped;
            env->arkanoid.prev_phi = phi;
        } else {
            env->arkanoid.prev_phi = 0.0f;
        }
    } else if (env->arkanoid.ball_active && env->arkanoid.ball_vy < -0.5f) {
        float center = (float)ARKANOID_W * 0.5f;
        float hw = env->arkanoid.player_w * 0.5f;
        float denom = center - hw;
        if (denom > 1e-6f) {
            float rel = (env->arkanoid.player_x - center) / denom;
            rel = clamp_float(rel, -1.0f, 1.0f);
            float phi = -fabsf(rel);
            float dphi = phi - env->arkanoid.prev_phi;
            dphi = clamp_float(dphi, -center_phi_clamp, center_phi_clamp);
            env->arkanoid.reward_vec[3] += center_k * dphi;
            env->arkanoid.prev_phi = phi;
        } else {
            env->arkanoid.prev_phi = 0.0f;
        }
    } else {
        env->arkanoid.prev_phi = 0.0f;
    }

    env->arkanoid.reward_vec[4] += step_penalty;
    if (env->arkanoid.prev_bricks_left > env->arkanoid.bricks_left) {
        int delta = env->arkanoid.prev_bricks_left - env->arkanoid.bricks_left;
        env->arkanoid.reward_vec[3] += progress_k * (float)delta;
    }
    env->arkanoid.prev_bricks_left = env->arkanoid.bricks_left;

    reward = env->arkanoid.reward_vec[0] + env->arkanoid.reward_vec[1] +
             env->arkanoid.reward_vec[2] + env->arkanoid.reward_vec[3] +
             env->arkanoid.reward_vec[4];

    tfrl_step_result out = {0};
    breakout_fill_obs(env, &out.observation);
    out.reward = reward;
    out.done = done;

    if (env->arkanoid.ball_active) {
        env->arkanoid.prev_ball_x = env->arkanoid.ball_x;
        env->arkanoid.prev_ball_y = env->arkanoid.ball_y;
        env->arkanoid.prev_ball_valid = 1;
    } else {
        env->arkanoid.prev_ball_valid = 0;
        env->arkanoid.pred_ema_valid = 0;
    }

    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}

tfrl_step_result tfrl_env_step_breakout(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    if (act < 0 || act > 2) act = 0;

    tfrl_step_result out = {0};
    double reward_sum = 0.0;
    int done = 0;

    for (int i = 0; i < BREAKOUT_ACTION_REPEAT; i++) {
        tfrl_step_result step = breakout_step_frame(env, act);
        reward_sum += step.reward;

        env->steps += 1;
        if (env->steps >= BREAKOUT_MAX_STEPS) {
            done = 1;
        }

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

tfrl_step_result tfrl_env_step_breakout_disc(tfrl_env *env, tfrl_action action) {
    tfrl_step_result out = tfrl_env_step_breakout(env, action);
    out.observation.index = breakout_hash_obs(env);
    out.observation.data_len = 0;
    return out;
}
