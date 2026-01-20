#include "envs/envs_internal.h"

#include <stdlib.h>

#define FLOPPY_TUBE_SPACING 280.0f
#define FLOPPY_TUBE_TOP_H 255.0f
#define FLOPPY_TUBE_GAP 90.0f

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

static const tfrl_env_spec FLOPPY_SPEC = {
    .name = "floppy",
    .obs_n = 4,
    .action_n = 2,
    .max_steps = FLOPPY_MAX_STEPS,
    .width = FLOPPY_W,
    .height = FLOPPY_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 4,
    .action_dims = 1,
    .obs_shape = {4, 0},
    .action_shape = {2, 0},
    .obs_dtype = TFRL_DTYPE_FLOAT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = 1.0,
    .action_low = 0.0,
    .action_high = 1.0,
    .agent_count = 1,
};

const tfrl_env_spec *tfrl_env_spec_floppy(void) {
    return &FLOPPY_SPEC;
}

tfrl_obs tfrl_env_reset_floppy(tfrl_env *env, uint64_t seed) {
    (void)seed;
    tfrl_env_reset_state(env);
    env->floppy.x = 80.0f;
    env->floppy.y = (float)FLOPPY_H / 2.0f - (float)FLOPPY_RADIUS;
    env->floppy.score = 0;
    env->floppy.last_action = 0;

    for (int i = 0; i < FLOPPY_MAX_TUBES; i++) {
        env->floppy.tubes_x[i] = 400.0f + FLOPPY_TUBE_SPACING * (float)i;
        env->floppy.tubes_offset_y[i] = -(float)tfrl_env_rand_range_int(env, 0, 120);
        env->floppy.tubes_scored[i] = 1;
    }

    tfrl_obs obs = {0};
    obs.data_len = 4;
    obs.data[0] = env->floppy.y / (float)FLOPPY_H;
    obs.data[1] = 1.0f;
    obs.data[2] = 0.5f;
    obs.data[3] = 0.0f;
    return obs;
}

tfrl_step_result tfrl_env_step_floppy(tfrl_env *env, tfrl_action action) {
    int act = action.index == 1 ? 1 : 0;
    env->floppy.last_action = act;

    if (act == 1) {
        env->floppy.y -= 3.0f;
    } else {
        env->floppy.y += 1.0f;
    }

    float max_x = env->floppy.tubes_x[0];
    for (int i = 0; i < FLOPPY_MAX_TUBES; i++) {
        env->floppy.tubes_x[i] -= 2.0f;
        if (env->floppy.tubes_x[i] > max_x) max_x = env->floppy.tubes_x[i];
    }
    for (int i = 0; i < FLOPPY_MAX_TUBES; i++) {
        if (env->floppy.tubes_x[i] + FLOPPY_TUBES_WIDTH < 0.0f) {
            env->floppy.tubes_x[i] = max_x + FLOPPY_TUBE_SPACING;
            env->floppy.tubes_offset_y[i] = -(float)tfrl_env_rand_range_int(env, 0, 120);
            env->floppy.tubes_scored[i] = 1;
            max_x = env->floppy.tubes_x[i];
        }
    }

    int done = 0;
    double reward = 0.0;

    if (env->floppy.y - FLOPPY_RADIUS < 0.0f || env->floppy.y + FLOPPY_RADIUS > (float)FLOPPY_H) {
        done = 1;
        reward = -1.0;
    }

    if (!done) {
        for (int i = 0; i < FLOPPY_MAX_TUBES; i++) {
            float rx = env->floppy.tubes_x[i];
            float ry_top = env->floppy.tubes_offset_y[i];
            float ry_bot = 600.0f + env->floppy.tubes_offset_y[i] - FLOPPY_TUBE_TOP_H;
            if (circle_rect_collision(env->floppy.x, env->floppy.y, FLOPPY_RADIUS, rx, ry_top,
                                      FLOPPY_TUBES_WIDTH, FLOPPY_TUBE_TOP_H) ||
                circle_rect_collision(env->floppy.x, env->floppy.y, FLOPPY_RADIUS, rx, ry_bot,
                                      FLOPPY_TUBES_WIDTH, FLOPPY_TUBE_TOP_H)) {
                done = 1;
                reward = -1.0;
                break;
            }
        }
    }

    if (!done) {
        for (int i = 0; i < FLOPPY_MAX_TUBES; i++) {
            if (env->floppy.tubes_scored[i] &&
                env->floppy.tubes_x[i] + FLOPPY_TUBES_WIDTH < env->floppy.x) {
                env->floppy.tubes_scored[i] = 0;
                env->floppy.score += 1;
                reward += 1.0;
            }
        }
    }

    env->steps += 1;
    if (env->steps >= FLOPPY_MAX_STEPS) done = 1;

    int next_idx = -1;
    float best_dx = 1e9f;
    for (int i = 0; i < FLOPPY_MAX_TUBES; i++) {
        float dx = env->floppy.tubes_x[i] - env->floppy.x;
        if (dx + FLOPPY_TUBES_WIDTH >= 0.0f && dx < best_dx) {
            best_dx = dx;
            next_idx = i;
        }
    }

    float next_dx_norm = 0.0f;
    float gap_center_norm = 0.5f;
    if (next_idx >= 0) {
        float next_dx = env->floppy.tubes_x[next_idx] - env->floppy.x;
        if (next_dx < 0.0f) next_dx = 0.0f;
        next_dx_norm = next_dx / (float)FLOPPY_W;
        float gap_center = env->floppy.tubes_offset_y[next_idx] + FLOPPY_TUBE_TOP_H + FLOPPY_TUBE_GAP * 0.5f;
        if (gap_center < 0.0f) gap_center = 0.0f;
        if (gap_center > (float)FLOPPY_H) gap_center = (float)FLOPPY_H;
        gap_center_norm = gap_center / (float)FLOPPY_H;
    }

    tfrl_step_result out = {0};
    out.observation.data_len = 4;
    out.observation.data[0] = env->floppy.y / (float)FLOPPY_H;
    out.observation.data[1] = next_dx_norm;
    out.observation.data[2] = gap_center_norm;
    out.observation.data[3] = (float)env->floppy.last_action;
    out.reward = reward;
    out.done = done;
    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}
