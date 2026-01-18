#include "envs_internal.h"

static const tfrl_env_spec LINEWORLD_SPEC = {
    .name = "lineworld",
    .obs_n = LINEWORLD_W * LINEWORLD_H,
    .action_n = 2,
    .max_steps = LINEWORLD_MAX_STEPS,
    .width = LINEWORLD_W,
    .height = LINEWORLD_H,
    .obs_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {LINEWORLD_W * LINEWORLD_H, 0},
    .action_shape = {2, 0},
    .obs_dtype = TFRL_DTYPE_INT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = (double)(LINEWORLD_W * LINEWORLD_H - 1),
    .action_low = 0.0,
    .action_high = 1.0,
    .agent_count = 1,
};

static const tfrl_env_spec LINEWORLD_CONT_SPEC = {
    .name = "lineworld_cont",
    .obs_n = 1,
    .action_n = 2,
    .max_steps = LINEWORLD_MAX_STEPS,
    .width = LINEWORLD_W,
    .height = LINEWORLD_H,
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
};

static const tfrl_env_spec LINEWORLD_DUO_SPEC = {
    .name = "lineworld_duo",
    .obs_n = LINEWORLD_W * LINEWORLD_H,
    .action_n = 2,
    .max_steps = LINEWORLD_MAX_STEPS,
    .width = LINEWORLD_W,
    .height = LINEWORLD_H,
    .obs_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {LINEWORLD_W * LINEWORLD_H, 0},
    .action_shape = {2, 0},
    .obs_dtype = TFRL_DTYPE_INT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = (double)(LINEWORLD_W * LINEWORLD_H - 1),
    .action_low = 0.0,
    .action_high = 1.0,
    .agent_count = LINEWORLD_DUO_AGENTS,
};

const tfrl_env_spec *tfrl_env_spec_lineworld(void) {
    return &LINEWORLD_SPEC;
}

const tfrl_env_spec *tfrl_env_spec_lineworld_cont(void) {
    return &LINEWORLD_CONT_SPEC;
}

const tfrl_env_spec *tfrl_env_spec_lineworld_duo(void) {
    return &LINEWORLD_DUO_SPEC;
}

tfrl_obs tfrl_env_reset_lineworld(tfrl_env *env, uint64_t seed) {
    (void)seed;
    tfrl_env_reset_state(env);
    tfrl_obs obs = {0};
    obs.index = env->x;
    return obs;
}

tfrl_obs tfrl_env_reset_lineworld_cont(tfrl_env *env, uint64_t seed) {
    (void)seed;
    tfrl_env_reset_state(env);
    tfrl_obs obs = {0};
    obs.data_len = 1;
    obs.data[0] = (float)env->x / (float)(LINEWORLD_W - 1) * 2.0f - 1.0f;
    return obs;
}

tfrl_obs tfrl_env_reset_lineworld_duo(tfrl_env *env, uint64_t seed) {
    return tfrl_env_reset_lineworld(env, seed);
}

tfrl_step_result tfrl_env_step_lineworld(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    int nx = env->x;
    if (act == 0) nx -= 1;
    else if (act == 1) nx += 1;
    if (nx >= 0 && nx < LINEWORLD_W) {
        env->x = nx;
    }
    env->steps += 1;
    int reached = (env->x == LINEWORLD_W - 1);
    int done = reached || (env->steps >= LINEWORLD_MAX_STEPS);

    tfrl_step_result out = {0};
    out.observation.index = env->x;
    out.reward = reached ? 1.0 : -0.01;
    out.done = done;
    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}

tfrl_step_result tfrl_env_step_lineworld_cont(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    int nx = env->x;
    if (act == 0) nx -= 1;
    else if (act == 1) nx += 1;
    if (nx >= 0 && nx < LINEWORLD_W) {
        env->x = nx;
    }
    env->steps += 1;
    int reached = (env->x == LINEWORLD_W - 1);
    int done = reached || (env->steps >= LINEWORLD_MAX_STEPS);

    tfrl_step_result out = {0};
    out.observation.data_len = 1;
    out.observation.data[0] = (float)env->x / (float)(LINEWORLD_W - 1) * 2.0f - 1.0f;
    out.reward = reached ? 1.0 : -0.01;
    out.done = done;
    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}

tfrl_step_result tfrl_env_step_lineworld_duo(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    int nx = env->x;
    if (act == 0) nx -= 1;
    else if (act == 1) nx += 1;
    if (nx >= 0 && nx < LINEWORLD_W) {
        env->x = nx;
    }
    env->steps += 1;
    int reached = (env->x == LINEWORLD_W - 1) && (env->x2 == LINEWORLD_W - 1);
    int done = reached || (env->steps >= LINEWORLD_MAX_STEPS);

    tfrl_step_result out = {0};
    out.observation.index = env->x;
    out.reward = reached ? 1.0 : -0.01;
    out.done = done;
    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}

int tfrl_env_step_multi_lineworld_duo(tfrl_env *env, const tfrl_action *actions, int action_count,
                                     tfrl_step_result *out_steps, int max_agents) {
    if (!env || !actions || !out_steps || max_agents < LINEWORLD_DUO_AGENTS || action_count < LINEWORLD_DUO_AGENTS) {
        return 0;
    }
    int nx0 = env->x;
    int nx1 = env->x2;
    if (actions[0].index == 0) nx0 -= 1;
    else if (actions[0].index == 1) nx0 += 1;
    if (actions[1].index == 0) nx1 -= 1;
    else if (actions[1].index == 1) nx1 += 1;
    if (nx0 >= 0 && nx0 < LINEWORLD_W) env->x = nx0;
    if (nx1 >= 0 && nx1 < LINEWORLD_W) env->x2 = nx1;
    env->steps += 1;

    int reached = (env->x == LINEWORLD_W - 1) && (env->x2 == LINEWORLD_W - 1);
    int done = reached || (env->steps >= LINEWORLD_MAX_STEPS);
    for (int i = 0; i < LINEWORLD_DUO_AGENTS; i++) {
        out_steps[i].observation.index = (i == 0) ? env->x : env->x2;
        out_steps[i].reward = reached ? 1.0 : -0.01;
        out_steps[i].done = done;
    }
    env->last_reward = (float)out_steps[0].reward;
    env->last_done = done;
    return LINEWORLD_DUO_AGENTS;
}
