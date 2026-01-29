#include "envs/envs_internal.h"

static const tfrl_env_spec POINT1D_SPEC = {
    .name = "point1d",
    .obs_n = 1,
    .action_n = 2,
    .max_steps = POINT1D_MAX_STEPS,
    .width = POINT1D_W,
    .height = POINT1D_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {1, 0},
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

const tfrl_env_spec *tfrl_env_spec_point1d(void) {
    return &POINT1D_SPEC;
}

tfrl_obs tfrl_env_reset_point1d(tfrl_env *env, uint64_t seed) {
    tfrl_env_reset_state(env);
    env->rng_state = (uint32_t)(seed ? seed : 1u);
    env->pos = -1.0f + 2.0f * tfrl_env_rand_uniform(env);
    tfrl_obs obs = {0};
    obs.data_len = 1;
    obs.data[0] = env->pos;
    return obs;
}

tfrl_step_result tfrl_env_step_point1d(tfrl_env *env, tfrl_action action) {
    int act = action.index == 1 ? 1 : 0;
    float delta = (act == 1) ? POINT1D_STEP_SCALE : -POINT1D_STEP_SCALE;
    env->pos += delta;
    if (env->pos > 1.0f) env->pos = 1.0f;
    if (env->pos < -1.0f) env->pos = -1.0f;
    env->steps += 1;
    float dist = 1.0f - env->pos;
    int done = (dist < 0.05f) || (env->steps >= POINT1D_MAX_STEPS);

    tfrl_step_result out = {0};
    out.observation.data_len = 1;
    out.observation.data[0] = env->pos;
    out.reward = (dist < 0.05f) ? 1.0 : -0.01;
    out.done = done;
    env->last_reward = (float)out.reward;
    env->last_done = done;
    return out;
}
