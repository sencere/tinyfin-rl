#include <stdint.h>
#include <stdio.h>

#include "tinyfin_rl/rl.h"

#define ACTIONS 2

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
    uint64_t rng;
} random_agent;

static int random_act(void *ctx, const tfrl_value *obs, tfrl_value *out_action) {
    random_agent *agent = (random_agent *)ctx;
    (void)obs;
    int action = (int)(rng_uniform(&agent->rng) * ACTIONS) % ACTIONS;
    out_action->type = TFRL_VALUE_I64;
    out_action->as.i64 = action;
    return TFRL_OK;
}

static int random_observe(void *ctx, const tfrl_transition *transition) {
    (void)ctx;
    (void)transition;
    return TFRL_OK;
}

typedef struct {
    tfrl_env *env;
    tfrl_agent *agent;
    int episodes;
    double return_sum;
} simple_trainer;

static int trainer_train(void *ctx, int episodes) {
    simple_trainer *trainer = (simple_trainer *)ctx;
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
        trainer->return_sum += step.reward;
    }
    return TFRL_OK;
}

static double run_once(uint64_t seed) {
    bandit_env env_state = {.probs = {0.3, 0.7}, .rng = seed};
    const tfrl_env_vtable env_vtable = {
        .reset = bandit_reset,
        .step = bandit_step,
        .seed = bandit_seed,
        .destroy = NULL
    };
    tfrl_env env = {&env_state, &env_vtable};
    tfrl_env_seed(&env, seed);

    random_agent agent_state = {.rng = seed};
    const tfrl_agent_vtable agent_vtable = {
        .act = random_act,
        .observe = random_observe,
        .destroy = NULL
    };
    tfrl_agent agent = {&agent_state, &agent_vtable};

    simple_trainer trainer_state = {.env = &env, .agent = &agent, .episodes = 200, .return_sum = 0.0};
    const tfrl_trainer_vtable trainer_vtable = {
        .train = trainer_train,
        .destroy = NULL
    };
    tfrl_trainer trainer = {&trainer_state, &trainer_vtable};

    if (tfrl_trainer_train(&trainer, trainer_state.episodes) != TFRL_OK) {
        return -1.0;
    }

    return trainer_state.return_sum / trainer_state.episodes;
}

int main(void) {
    double run1 = run_once(1234);
    double run2 = run_once(1234);
    printf("run1=%.4f run2=%.4f\n", run1, run2);
    if (run1 < 0.0 || run2 < 0.0) {
        return 1;
    }
    if (run1 != run2) {
        fprintf(stderr, "determinism check failed\n");
        return 2;
    }
    return 0;
}
