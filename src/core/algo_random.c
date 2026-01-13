#include <stdlib.h>

#include "algo_api.h"

typedef struct {
    int action_n;
} tfrl_random_algo;

static tfrl_action random_act(void *ctx, tfrl_obs obs) {
    (void)obs;
    tfrl_random_algo *algo = (tfrl_random_algo *)ctx;
    tfrl_action action = {0};
    int n = algo->action_n > 0 ? algo->action_n : 1;
    action.index = rand() % n;
    return action;
}

static void random_update(void *ctx, const tfrl_transition *transition) {
    (void)ctx;
    (void)transition;
}

static void random_destroy(void *ctx) {
    free(ctx);
}

static const tfrl_algo_vtable RANDOM_VTABLE = {
    .act = random_act,
    .act_batch = NULL,
    .update = random_update,
    .save = NULL,
    .destroy = random_destroy,
};

tfrl_algo tfrl_algo_random_create(int action_n) {
    tfrl_random_algo *algo = (tfrl_random_algo *)calloc(1, sizeof(tfrl_random_algo));
    if (!algo) {
        tfrl_algo out = {0};
        return out;
    }
    algo->action_n = action_n;
    tfrl_algo out = {algo, &RANDOM_VTABLE};
    return out;
}
