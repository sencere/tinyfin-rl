#include <stdio.h>
#include <string.h>

#include "algo_api.h"

tfrl_algo tfrl_algo_random_create(int action_n);
tfrl_algo tfrl_algo_dqn_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_reinforce_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_ppo_create(const tfrl_algo_config *cfg);

tfrl_algo tfrl_algo_create(const tfrl_algo_config *cfg, const tfrl_env_spec *spec) {
    tfrl_algo out = {0};
    if (!cfg || !spec) return out;
    if (!cfg->name) {
        return tfrl_algo_random_create(spec->action_n);
    }
    if (strcmp(cfg->name, "dqn") == 0) {
        return tfrl_algo_dqn_create(cfg);
    }
    if (strcmp(cfg->name, "reinforce") == 0) {
        return tfrl_algo_reinforce_create(cfg);
    }
    if (strcmp(cfg->name, "ppo") == 0) {
        return tfrl_algo_ppo_create(cfg);
    }
    fprintf(stderr, "unknown algo '%s'; using random policy\n", cfg->name);
    return tfrl_algo_random_create(spec->action_n);
}

void tfrl_algo_destroy(tfrl_algo *algo) {
    if (!algo || !algo->vtable || !algo->vtable->destroy) return;
    algo->vtable->destroy(algo->ctx);
    algo->ctx = NULL;
    algo->vtable = NULL;
}

void tfrl_algo_save(tfrl_algo *algo, const char *path) {
    if (!algo || !algo->vtable || !algo->vtable->save) return;
    algo->vtable->save(algo->ctx, path);
}

void tfrl_algo_act_batch(tfrl_algo *algo, const tfrl_obs *obs, int count, tfrl_action *out_actions) {
    if (!algo || !algo->vtable || count <= 0 || !out_actions) return;
    if (algo->vtable->act_batch) {
        algo->vtable->act_batch(algo->ctx, obs, count, out_actions);
        return;
    }
    for (int i = 0; i < count; i++) {
        out_actions[i] = algo->vtable->act(algo->ctx, obs[i]);
    }
}
