#include <stdlib.h>
#include <string.h>

#include "algo_api.h"

typedef struct {
    int action_n;
    tfrl_space_type action_type;
    int action_dims;
    int action_shape[2];
    double action_low;
    double action_high;
} tfrl_random_algo;

static tfrl_action random_act(void *ctx, tfrl_obs obs) {
    (void)obs;
    tfrl_random_algo *algo = (tfrl_random_algo *)ctx;
    tfrl_action action = {0};
    if (!algo) return action;

    if (algo->action_type == TFRL_SPACE_DISCRETE) {
        int n = algo->action_n > 0 ? algo->action_n : 1;
        action.index = rand() % n;
    } else {
        int dims = algo->action_dims > 0 ? algo->action_dims : 1;
        if (dims > TFRL_MAX_BOX_DIMS) dims = TFRL_MAX_BOX_DIMS;
        action.data_len = dims;
        double lo = algo->action_low;
        double hi = algo->action_high;
        if (hi < lo) {
            double tmp = hi;
            hi = lo;
            lo = tmp;
        }
        for (int i = 0; i < dims; i++) {
            double r = (double)rand() / (double)RAND_MAX;
            action.data[i] = (float)(lo + (hi - lo) * r);
        }
    }
    return action;
}

static void random_act_batch(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions) {
    if (!ctx || !out_actions || count <= 0) return;
    for (int i = 0; i < count; i++) {
        out_actions[i] = random_act(ctx, obs ? obs[i] : (tfrl_obs){0});
    }
}

static void random_update(void *ctx, const tfrl_transition *transition) {
    (void)ctx;
    (void)transition;
}

static void random_destroy(void *ctx) {
    free(ctx);
}

static int random_diagnostics(void *ctx, char *buffer, size_t buffer_len) {
    (void)ctx;
    if (!buffer || buffer_len == 0) return 0;
    buffer[0] = '\0';
    return 1;
}

static const tfrl_algo_vtable RANDOM_VTABLE = {
    .act = random_act,
    .act_env = NULL,
    .act_batch = random_act_batch,
    .update = random_update,
    .save = NULL,
    .destroy = random_destroy,
    .diagnostics = random_diagnostics,
};

tfrl_algo tfrl_algo_random_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->action_n <= 0) return out;
    tfrl_random_algo *algo = (tfrl_random_algo *)calloc(1, sizeof(tfrl_random_algo));
    if (!algo) return out;
    algo->action_n = cfg->action_n;
    algo->action_type = cfg->action_type;
    algo->action_dims = cfg->action_dims;
    algo->action_shape[0] = cfg->action_shape[0];
    algo->action_shape[1] = cfg->action_shape[1];
    algo->action_low = cfg->action_low;
    algo->action_high = cfg->action_high;
    out.ctx = algo;
    out.vtable = &RANDOM_VTABLE;
    return out;
}
