#include <stdio.h>

#include "tinyfin_rl/rl.h"

typedef struct {
    int state;
    int steps;
} smoke_env;

static int smoke_reset(void *ctx, tfrl_value *out_obs) {
    smoke_env *env = (smoke_env *)ctx;
    env->state = 0;
    env->steps = 0;
    out_obs->type = TFRL_VALUE_I64;
    out_obs->as.i64 = env->state;
    return TFRL_OK;
}

static int smoke_step(void *ctx, const tfrl_value *action, tfrl_step *out_step) {
    smoke_env *env = (smoke_env *)ctx;
    env->state += (int)action->as.i64;
    env->steps += 1;
    out_step->observation.type = TFRL_VALUE_I64;
    out_step->observation.as.i64 = env->state;
    out_step->reward = 1.0;
    out_step->terminated = env->steps >= 5;
    out_step->truncated = 0;
    return TFRL_OK;
}

int main(void) {
    smoke_env env_state = {0};
    const tfrl_env_vtable vtable = {
        .reset = smoke_reset,
        .step = smoke_step,
        .seed = NULL,
        .destroy = NULL
    };
    tfrl_env env = {&env_state, &vtable};
    tfrl_value obs = {0};
    tfrl_step step = {0};
    tfrl_env_reset(&env, &obs);
    for (int i = 0; i < 5; i++) {
        tfrl_value act = {.type = TFRL_VALUE_I64, .as.i64 = 1};
        tfrl_env_step(&env, &act, &step);
    }
    printf("ok\n");
    return 0;
}
