#include "tinyfin_rl/rl.h"

typedef struct {
    int step_count;
} dummy_env;

typedef struct {
    int last_action;
} dummy_agent;

static int dummy_env_reset(void *ctx, tfrl_value *out_obs) {
    dummy_env *env = (dummy_env *)ctx;
    env->step_count = 0;
    out_obs->type = TFRL_VALUE_I64;
    out_obs->as.i64 = 0;
    return TFRL_OK;
}

static int dummy_env_step(void *ctx, const tfrl_value *action, tfrl_step *out_step) {
    dummy_env *env = (dummy_env *)ctx;
    env->step_count += 1;
    (void)action;
    out_step->observation.type = TFRL_VALUE_I64;
    out_step->observation.as.i64 = env->step_count;
    out_step->reward = 1.0;
    out_step->terminated = env->step_count >= 1;
    out_step->truncated = 0;
    return TFRL_OK;
}

static int dummy_env_seed(void *ctx, uint64_t seed) {
    (void)ctx;
    (void)seed;
    return TFRL_OK;
}

static int dummy_agent_act(void *ctx, const tfrl_value *obs, tfrl_value *out_action) {
    dummy_agent *agent = (dummy_agent *)ctx;
    (void)obs;
    agent->last_action = 0;
    out_action->type = TFRL_VALUE_I64;
    out_action->as.i64 = agent->last_action;
    return TFRL_OK;
}

static int dummy_agent_observe(void *ctx, const tfrl_transition *transition) {
    (void)ctx;
    (void)transition;
    return TFRL_OK;
}

int main(void) {
    dummy_env env_state = {0};
    dummy_agent agent_state = {0};

    const tfrl_env_vtable env_vtable = {
        .reset = dummy_env_reset,
        .step = dummy_env_step,
        .seed = dummy_env_seed,
        .destroy = NULL
    };

    const tfrl_agent_vtable agent_vtable = {
        .act = dummy_agent_act,
        .observe = dummy_agent_observe,
        .destroy = NULL
    };

    tfrl_env env = {&env_state, &env_vtable};
    tfrl_agent agent = {&agent_state, &agent_vtable};

    tfrl_value obs = {0};
    tfrl_value action = {0};
    tfrl_step step = {0};

    if (tfrl_env_reset(&env, &obs) != TFRL_OK) {
        return 1;
    }
    if (tfrl_agent_act(&agent, &obs, &action) != TFRL_OK) {
        return 2;
    }
    if (tfrl_env_step(&env, &action, &step) != TFRL_OK) {
        return 3;
    }

    tfrl_transition transition = {
        .observation = obs,
        .action = action,
        .reward = step.reward,
        .next_observation = step.observation,
        .terminated = step.terminated,
        .truncated = step.truncated
    };

    if (tfrl_agent_observe(&agent, &transition) != TFRL_OK) {
        return 4;
    }

    return 0;
}
