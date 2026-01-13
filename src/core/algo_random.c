#include <stdlib.h>

#include "algo_api.h"

typedef struct {
    int action_n;
    tfrl_space_type action_type;
    int action_dims;
    float action_low;
    float action_high;
} tfrl_random_algo;

static tfrl_action random_act(void *ctx, tfrl_obs obs) {
    (void)obs;
    tfrl_random_algo *algo = (tfrl_random_algo *)ctx;
    tfrl_action action = {0};
    if (algo->action_type == TFRL_SPACE_BOX) {
        int n = algo->action_dims > 0 ? algo->action_dims : 1;
        action.data_len = n;
        float low = algo->action_low;
        float high = algo->action_high;
        if (high <= low) {
            low = -1.0f;
            high = 1.0f;
        }
        for (int i = 0; i < n && i < TFRL_MAX_BOX_DIMS; i++) {
            float r = (float)rand() / (float)RAND_MAX;
            action.data[i] = low + r * (high - low);
        }
    } else {
        int n = algo->action_n > 0 ? algo->action_n : 1;
        action.index = rand() % n;
    }
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

tfrl_algo tfrl_algo_random_create(const tfrl_algo_config *cfg) {
    tfrl_random_algo *algo = (tfrl_random_algo *)calloc(1, sizeof(tfrl_random_algo));
    if (!algo) {
        tfrl_algo out = {0};
        return out;
    }
    algo->action_n = cfg ? cfg->action_n : 0;
    algo->action_type = cfg ? cfg->action_type : TFRL_SPACE_DISCRETE;
    algo->action_dims = cfg ? cfg->action_dims : 0;
    algo->action_low = cfg ? (float)cfg->action_low : -1.0f;
    algo->action_high = cfg ? (float)cfg->action_high : 1.0f;
    tfrl_algo out = {algo, &RANDOM_VTABLE};
    return out;
}
