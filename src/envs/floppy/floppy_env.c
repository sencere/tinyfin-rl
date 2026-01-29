#include "envs/envs_internal.h"

#include <string.h>

#define FLOPPY_GRAVITY 0.35f
#define FLOPPY_FLAP_VY -5.0f
#define FLOPPY_TUBE_GAP 70.0f
#define FLOPPY_TUBE_SPACING 120.0f

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
    .obs_low = -1.0,
    .obs_high = 1.0,
    .action_low = 0.0,
    .action_high = 1.0,
    .agent_count = 1,
    .obs_layout = NULL,
};

const tfrl_env_spec *tfrl_env_spec_floppy(void) {
    return &FLOPPY_SPEC;
}

static void floppy_init_tubes(tfrl_env *env) {
    for (int i = 0; i < FLOPPY_MAX_TUBES; i++) {
        env->floppy.tubes_x[i] = (float)FLOPPY_W + i * FLOPPY_TUBE_SPACING;
        env->floppy.tubes_offset_y[i] = (float)tfrl_env_rand_range_int(env, -40, 40);
        env->floppy.tubes_scored[i] = 0;
    }
}

tfrl_obs tfrl_env_reset_floppy(tfrl_env *env, uint64_t seed) {
    tfrl_env_reset_state(env);
    env->rng_state = (uint32_t)(seed ? seed : 1u);
    env->floppy.x = (float)FLOPPY_W * 0.25f;
    env->floppy.y = (float)FLOPPY_H * 0.5f;
    env->floppy.last_action = 0;
    env->floppy.score = 0;
    floppy_init_tubes(env);
    tfrl_obs obs = {0};
    obs.data_len = 4;
    obs.data[0] = env->floppy.y / (float)FLOPPY_H;
    obs.data[1] = 0.0f;
    obs.data[2] = (env->floppy.tubes_x[0] - env->floppy.x) / (float)FLOPPY_W;
    obs.data[3] = (env->floppy.tubes_offset_y[0]) / (float)FLOPPY_H;
    return obs;
}

tfrl_step_result tfrl_env_step_floppy(tfrl_env *env, tfrl_action action) {
    int flap = action.index == 1;
    if (flap) env->pos = FLOPPY_FLAP_VY;
    env->pos += FLOPPY_GRAVITY;
    env->floppy.y += env->pos;

    for (int i = 0; i < FLOPPY_MAX_TUBES; i++) {
        env->floppy.tubes_x[i] -= 2.0f;
        if (env->floppy.tubes_x[i] < -FLOPPY_TUBES_WIDTH) {
            env->floppy.tubes_x[i] = (float)FLOPPY_W + FLOPPY_TUBE_SPACING;
            env->floppy.tubes_offset_y[i] = (float)tfrl_env_rand_range_int(env, -40, 40);
            env->floppy.tubes_scored[i] = 0;
        }
    }

    int done = 0;
    double reward = 0.01;
    if (env->floppy.y < 0 || env->floppy.y > FLOPPY_H) {
        done = 1;
        reward = -1.0;
    }

    for (int i = 0; i < FLOPPY_MAX_TUBES; i++) {
        float tx = env->floppy.tubes_x[i];
        float gap_y = (float)FLOPPY_H * 0.5f + env->floppy.tubes_offset_y[i];
        if (env->floppy.x + FLOPPY_RADIUS > tx && env->floppy.x - FLOPPY_RADIUS < tx + FLOPPY_TUBES_WIDTH) {
            if (env->floppy.y < gap_y - FLOPPY_TUBE_GAP * 0.5f ||
                env->floppy.y > gap_y + FLOPPY_TUBE_GAP * 0.5f) {
                done = 1;
                reward = -1.0;
            }
        }
        if (!env->floppy.tubes_scored[i] && env->floppy.x > tx + FLOPPY_TUBES_WIDTH) {
            env->floppy.tubes_scored[i] = 1;
            env->floppy.score += 1;
            reward += 1.0;
        }
    }

    env->steps += 1;
    if (env->steps >= FLOPPY_MAX_STEPS) done = 1;

    int next = 0;
    for (int i = 0; i < FLOPPY_MAX_TUBES; i++) {
        if (env->floppy.tubes_x[i] + FLOPPY_TUBES_WIDTH >= env->floppy.x) {
            next = i;
            break;
        }
    }

    tfrl_step_result out = {0};
    out.observation.data_len = 4;
    out.observation.data[0] = env->floppy.y / (float)FLOPPY_H;
    out.observation.data[1] = env->pos / 10.0f;
    out.observation.data[2] = (env->floppy.tubes_x[next] - env->floppy.x) / (float)FLOPPY_W;
    out.observation.data[3] = env->floppy.tubes_offset_y[next] / (float)FLOPPY_H;
    out.reward = reward;
    out.done = done;
    env->last_reward = (float)reward;
    env->last_done = done;
    return out;
}
