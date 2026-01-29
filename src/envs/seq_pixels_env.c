#include "envs/envs_internal.h"
#include "envs/render_grid.h"

static const tfrl_env_spec SEQ_PIXELS_SPEC = {
    .name = "seq_pixels",
    .obs_n = SEQ_PIXELS_W,
    .action_n = SEQ_PIXELS_W,
    .max_steps = SEQ_PIXELS_MAX_STEPS,
    .width = SEQ_PIXELS_W,
    .height = SEQ_PIXELS_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = SEQ_PIXELS_W,
    .action_dims = 1,
    .obs_shape = {SEQ_PIXELS_W, 0},
    .action_shape = {SEQ_PIXELS_W, 0},
    .obs_dtype = TFRL_DTYPE_FLOAT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = 1.0,
    .action_low = 0.0,
    .action_high = (double)(SEQ_PIXELS_W - 1),
    .agent_count = 1,
    .obs_layout = NULL,
};

const tfrl_env_spec *tfrl_env_spec_seq_pixels(void) {
    return &SEQ_PIXELS_SPEC;
}

static void seq_pixels_fill_obs(tfrl_env *env, tfrl_obs *obs) {
    if (!env || !obs) return;
    obs->data_len = SEQ_PIXELS_W;
    for (int i = 0; i < SEQ_PIXELS_W; i++) {
        obs->data[i] = 0.0f;
    }

    if (env->steps < SEQ_PIXELS_SEQ_LEN) {
        int idx = env->seq_pixels[env->steps];
        if (idx < 1) idx = 1;
        if (idx >= SEQ_PIXELS_W) idx = SEQ_PIXELS_W - 1;
        obs->data[idx] = 1.0f;
    } else {
        obs->data[0] = 1.0f;
    }
}

tfrl_obs tfrl_env_reset_seq_pixels(tfrl_env *env, uint64_t seed) {
    tfrl_obs obs = {0};
    if (!env) return obs;
    tfrl_env_reset_state(env);
    env->rng_state = (uint32_t)(seed ? seed : 1u);
    env->steps = 0;
    env->seq_cursor = 0;
    for (int i = 0; i < SEQ_PIXELS_SEQ_LEN; i++) {
        env->seq_pixels[i] = tfrl_env_rand_range_int(env, 1, SEQ_PIXELS_W - 1);
    }
    seq_pixels_fill_obs(env, &obs);
    return obs;
}

tfrl_step_result tfrl_env_step_seq_pixels(tfrl_env *env, tfrl_action action) {
    tfrl_step_result out = {0};
    if (!env) return out;

    int act = action.index;
    if (act < 0) act = 0;
    if (act >= SEQ_PIXELS_W) act = SEQ_PIXELS_W - 1;

    double reward = 0.0;
    if (env->steps >= SEQ_PIXELS_SEQ_LEN) {
        int target_idx = env->steps - SEQ_PIXELS_SEQ_LEN;
        if (target_idx >= 0 && target_idx < SEQ_PIXELS_SEQ_LEN) {
            if (act == env->seq_pixels[target_idx]) reward = 1.0;
        }
    }

    env->steps += 1;
    int done = env->steps >= SEQ_PIXELS_MAX_STEPS;

    seq_pixels_fill_obs(env, &out.observation);
    out.reward = reward;
    out.done = done;
    env->last_reward = (float)reward;
    env->last_done = done;
    return out;
}

size_t tfrl_env_render_bytes_seq_pixels(const tfrl_env *env) {
    (void)env;
    return tfrl_render_grid_bytes(SEQ_PIXELS_W, SEQ_PIXELS_H, 0);
}

size_t tfrl_env_render_write_seq_pixels(tfrl_env *env, void *buffer, size_t buffer_len) {
    unsigned char cells[SEQ_PIXELS_W * SEQ_PIXELS_H];
    for (int i = 0; i < SEQ_PIXELS_W * SEQ_PIXELS_H; i++) {
        cells[i] = 0;
    }
    int lit = 0;
    if (env->steps < SEQ_PIXELS_SEQ_LEN) {
        lit = env->seq_pixels[env->steps];
    } else {
        lit = 0;
    }
    if (lit < 0) lit = 0;
    if (lit >= SEQ_PIXELS_W) lit = SEQ_PIXELS_W - 1;
    cells[lit] = 255;
    return tfrl_render_grid_write(env, buffer, buffer_len, SEQ_PIXELS_W, SEQ_PIXELS_H,
                                  (uint32_t)lit, 0,
                                  (uint32_t)(SEQ_PIXELS_W - 1), 0,
                                  (uint32_t)SEQ_PIXELS_MAX_STEPS, (uint32_t)env->kind,
                                  cells, 0, NULL);
}
