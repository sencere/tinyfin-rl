#include <stdio.h>
#include <string.h>

#include "algo_api.h"

tfrl_algo tfrl_algo_random_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_dqn_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_rainbow_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_qrdqn_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_a2c_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_trpo_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_sac_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_td3_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_impala_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_mcts_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_reinforce_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_ppo_create(const tfrl_algo_config *cfg);

tfrl_algo tfrl_algo_create(const tfrl_algo_config *cfg, const tfrl_env_spec *spec) {
    tfrl_algo out = {0};
    if (!cfg || !spec) return out;
    if (!cfg->name) {
        return tfrl_algo_random_create(cfg);
    }
    if (strcmp(cfg->name, "random") == 0) {
        return tfrl_algo_random_create(cfg);
    }
    if (cfg->action_type == TFRL_SPACE_BOX) {
        if (strcmp(cfg->name, "sac") == 0 || strcmp(cfg->name, "td3") == 0) {
            return strcmp(cfg->name, "sac") == 0 ? tfrl_algo_sac_create(cfg) : tfrl_algo_td3_create(cfg);
        }
        fprintf(stderr, "algo '%s' does not support continuous actions yet; using random policy\n", cfg->name);
        return tfrl_algo_random_create(cfg);
    }
    if (strcmp(cfg->name, "mcts") == 0) {
        tfrl_algo algo = tfrl_algo_mcts_create(cfg);
        if (!algo.ctx) {
            fprintf(stderr, "mcts requires maze_rooms or lineworld; using random policy\n");
            return tfrl_algo_random_create(cfg);
        }
        return algo;
    }
    if (strcmp(cfg->name, "dqn") == 0) {
        return tfrl_algo_dqn_create(cfg);
    }
    if (strcmp(cfg->name, "rainbow") == 0) {
        return tfrl_algo_rainbow_create(cfg);
    }
    if (strcmp(cfg->name, "qrdqn") == 0) {
        return tfrl_algo_qrdqn_create(cfg);
    }
    if (strcmp(cfg->name, "a2c") == 0) {
        return tfrl_algo_a2c_create(cfg);
    }
    if (strcmp(cfg->name, "trpo") == 0) {
        return tfrl_algo_trpo_create(cfg);
    }
    if (strcmp(cfg->name, "sac") == 0) {
        return tfrl_algo_sac_create(cfg);
    }
    if (strcmp(cfg->name, "td3") == 0) {
        return tfrl_algo_td3_create(cfg);
    }
    if (strcmp(cfg->name, "impala") == 0) {
        return tfrl_algo_impala_create(cfg);
    }
    if (strcmp(cfg->name, "reinforce") == 0) {
        return tfrl_algo_reinforce_create(cfg);
    }
    if (strcmp(cfg->name, "ppo") == 0) {
        return tfrl_algo_ppo_create(cfg);
    }
    fprintf(stderr, "unknown algo '%s'; using random policy\n", cfg->name);
    return tfrl_algo_random_create(cfg);
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
