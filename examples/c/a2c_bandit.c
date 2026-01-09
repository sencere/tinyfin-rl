#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "tinyfin_rl/rl.h"

#define ACTIONS 3

static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static double rng_uniform(uint64_t *state) {
    uint64_t v = rng_next(state) >> 11;
    return (double)v * (1.0 / 9007199254740992.0);
}

typedef struct {
    double probs[ACTIONS];
    uint64_t rng;
} bandit_env;

static int bandit_reset(void *ctx, tfrl_value *out_obs) {
    (void)ctx;
    out_obs->type = TFRL_VALUE_I64;
    out_obs->as.i64 = 0;
    return TFRL_OK;
}

static int bandit_step(void *ctx, const tfrl_value *action, tfrl_step *out_step) {
    bandit_env *env = (bandit_env *)ctx;
    int idx = (int)action->as.i64;
    double r = rng_uniform(&env->rng) < env->probs[idx] ? 1.0 : 0.0;
    out_step->observation.type = TFRL_VALUE_I64;
    out_step->observation.as.i64 = 0;
    out_step->reward = r;
    out_step->terminated = 1;
    out_step->truncated = 0;
    return TFRL_OK;
}

static int bandit_seed(void *ctx, uint64_t seed) {
    bandit_env *env = (bandit_env *)ctx;
    env->rng = seed ? seed : 1;
    return TFRL_OK;
}

typedef struct {
    double logits[ACTIONS];
    double probs[ACTIONS];
    double value;
    double lr_policy;
    double lr_value;
    uint64_t rng;
} a2c_agent;

static int a2c_act(void *ctx, const tfrl_value *obs, tfrl_value *out_action) {
    a2c_agent *agent = (a2c_agent *)ctx;
    (void)obs;
    double max_logit = agent->logits[0];
    for (int i = 1; i < ACTIONS; i++) {
        if (agent->logits[i] > max_logit) {
            max_logit = agent->logits[i];
        }
    }
    double sum = 0.0;
    for (int i = 0; i < ACTIONS; i++) {
        double e = exp(agent->logits[i] - max_logit);
        agent->probs[i] = e;
        sum += e;
    }
    for (int i = 0; i < ACTIONS; i++) {
        agent->probs[i] /= sum;
    }
    double r = rng_uniform(&agent->rng);
    double cdf = 0.0;
    int action = ACTIONS - 1;
    for (int i = 0; i < ACTIONS; i++) {
        cdf += agent->probs[i];
        if (r <= cdf) {
            action = i;
            break;
        }
    }
    out_action->type = TFRL_VALUE_I64;
    out_action->as.i64 = action;
    return TFRL_OK;
}

static int a2c_observe(void *ctx, const tfrl_transition *transition) {
    a2c_agent *agent = (a2c_agent *)ctx;
    int action = (int)transition->action.as.i64;
    double reward = transition->reward;
    double advantage = reward - agent->value;
    agent->value += agent->lr_value * advantage;
    for (int i = 0; i < ACTIONS; i++) {
        double grad = (i == action ? 1.0 : 0.0) - agent->probs[i];
        agent->logits[i] += agent->lr_policy * advantage * grad;
    }
    return TFRL_OK;
}

typedef struct {
    tfrl_env *env;
    tfrl_agent *agent;
    int episodes;
    double return_sum;
} bandit_trainer;

static int bandit_train(void *ctx, int episodes) {
    bandit_trainer *trainer = (bandit_trainer *)ctx;
    trainer->return_sum = 0.0;
    for (int ep = 0; ep < episodes; ep++) {
        tfrl_value obs = {0};
        tfrl_value action = {0};
        tfrl_step step = {0};
        if (tfrl_env_reset(trainer->env, &obs) != TFRL_OK) {
            return TFRL_ERR;
        }
        if (tfrl_agent_act(trainer->agent, &obs, &action) != TFRL_OK) {
            return TFRL_ERR;
        }
        if (tfrl_env_step(trainer->env, &action, &step) != TFRL_OK) {
            return TFRL_ERR;
        }
        tfrl_transition transition = {
            .observation = obs,
            .action = action,
            .reward = step.reward,
            .next_observation = step.observation,
            .terminated = step.terminated,
            .truncated = step.truncated
        };
        if (tfrl_agent_observe(trainer->agent, &transition) != TFRL_OK) {
            return TFRL_ERR;
        }
        if (tfrl_step_done(&step)) {
            trainer->return_sum += step.reward;
        }
    }
    return TFRL_OK;
}

int main(void) {
    bandit_env env_state = {.probs = {0.2, 0.5, 0.8}, .rng = 1};
    const tfrl_env_vtable env_vtable = {
        .reset = bandit_reset,
        .step = bandit_step,
        .seed = bandit_seed,
        .destroy = NULL
    };
    tfrl_env env = {&env_state, &env_vtable};
    tfrl_env_seed(&env, 1234);

    a2c_agent agent_state = {
        .logits = {0.0, 0.0, 0.0},
        .probs = {0.0, 0.0, 0.0},
        .value = 0.0,
        .lr_policy = 0.1,
        .lr_value = 0.05,
        .rng = 1234
    };
    const tfrl_agent_vtable agent_vtable = {
        .act = a2c_act,
        .observe = a2c_observe,
        .destroy = NULL
    };
    tfrl_agent agent = {&agent_state, &agent_vtable};

    bandit_trainer trainer_state = {.env = &env, .agent = &agent, .episodes = 2000, .return_sum = 0.0};
    const tfrl_trainer_vtable trainer_vtable = {
        .train = bandit_train,
        .destroy = NULL
    };
    tfrl_trainer trainer = {&trainer_state, &trainer_vtable};

    if (tfrl_trainer_train(&trainer, trainer_state.episodes) != TFRL_OK) {
        fprintf(stderr, "training failed\n");
        return 1;
    }

    printf("episodes=%d return_mean=%.3f\n", trainer_state.episodes, trainer_state.return_sum / trainer_state.episodes);
    return 0;
}
