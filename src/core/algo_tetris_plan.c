#include <stdlib.h>

#include "algo_api.h"

typedef struct {
    int action_n;
} tfrl_tetris_plan_algo;

static tfrl_action tetris_plan_act(void *ctx, tfrl_obs obs) {
    (void)ctx;
    (void)obs;
    tfrl_action action = {0};
    action.index = -1;
    return action;
}

static void tetris_plan_update(void *ctx, const tfrl_transition *transition) {
    (void)ctx;
    (void)transition;
}

static void tetris_plan_destroy(void *ctx) {
    free(ctx);
}

static const tfrl_algo_vtable TETRIS_PLAN_VTABLE = {
    .act = tetris_plan_act,
    .act_batch = NULL,
    .update = tetris_plan_update,
    .save = NULL,
    .destroy = tetris_plan_destroy,
    .diagnostics = NULL,
};

tfrl_algo tfrl_algo_tetris_plan_create(const tfrl_algo_config *cfg) {
    tfrl_tetris_plan_algo *algo = (tfrl_tetris_plan_algo *)calloc(1, sizeof(tfrl_tetris_plan_algo));
    if (!algo) {
        tfrl_algo out = {0};
        return out;
    }
    algo->action_n = cfg ? cfg->action_n : 0;
    tfrl_algo out = {algo, &TETRIS_PLAN_VTABLE};
    return out;
}
