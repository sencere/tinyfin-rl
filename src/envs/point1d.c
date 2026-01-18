#include "envs_internal.h"

static const tfrl_env_spec POINT1D_SPEC = {
    .name = "point1d",
    .obs_n = 1,
    .action_n = 1,
    .max_steps = POINT1D_MAX_STEPS,
    .width = POINT1D_W,
    .height = POINT1D_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_BOX,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {1, 0},
    .action_shape = {1, 0},
    .obs_dtype = TFRL_DTYPE_FLOAT32,
    .action_dtype = TFRL_DTYPE_FLOAT32,
    .obs_low = -1.0,
    .obs_high = 1.0,
    .action_low = -1.0,
    .action_high = 1.0,
    .agent_count = 1,
};

const tfrl_env_spec *tfrl_env_spec_point1d(void) {
    return &POINT1D_SPEC;
}

tfrl_obs tfrl_env_reset_point1d(tfrl_env *env, uint64_t seed) {
    (void)seed;
    tfrl_env_reset_state(env);
    tfrl_obs obs = {0};
    obs.data_len = 1;
    obs.data[0] = env->pos;
    return obs;
}

tfrl_step_result tfrl_env_step_point1d(tfrl_env *env, tfrl_action action) {
    float a = 0.0f;
    if (action.data_len > 0) {
        a = action.data[0];
    }
    if (a < -1.0f) a = -1.0f;
    if (a > 1.0f) a = 1.0f;
    env->pos += a * POINT1D_STEP_SCALE;
    if (env->pos < -1.0f) env->pos = -1.0f;
    if (env->pos > 1.0f) env->pos = 1.0f;
    env->steps += 1;

    int reached = (env->pos >= 0.95f);
    int done = reached || (env->steps >= POINT1D_MAX_STEPS);

    tfrl_step_result out = {0};
    out.observation.data_len = 1;
    out.observation.data[0] = env->pos;
    float dist = 1.0f - env->pos;
    if (dist < 0.0f) dist = -dist;
    out.reward = reached ? 1.0 : -dist;
    out.done = done;
    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}
