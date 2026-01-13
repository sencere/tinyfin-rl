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

static int require_positive(const char *name, int v) {
    if (v > 0) return 1;
    fprintf(stderr, "invalid %s: %d\n", name, v);
    return 0;
}

static int require_nonneg(const char *name, int v) {
    if (v >= 0) return 1;
    fprintf(stderr, "invalid %s: %d\n", name, v);
    return 0;
}

static int require_float_positive(const char *name, float v) {
    if (v > 0.0f) return 1;
    fprintf(stderr, "invalid %s: %.6f\n", name, v);
    return 0;
}

static int require_discrete(const char *algo, const tfrl_algo_config *cfg) {
    if (cfg->obs_type == TFRL_SPACE_DISCRETE && cfg->action_type == TFRL_SPACE_DISCRETE) return 1;
    fprintf(stderr, "%s requires discrete obs/action spaces\n", algo);
    return 0;
}

static int validate_algo_config(const tfrl_algo_config *cfg) {
    if (!cfg || !cfg->name) return 1;
    if (!require_positive("obs_n", cfg->obs_n)) return 0;
    if (!require_positive("action_n", cfg->action_n)) return 0;
    if (!require_float_positive("gamma", cfg->gamma)) return 0;
    if (!require_nonneg("replay_size", cfg->replay_size)) return 0;
    if (!require_nonneg("batch_size", cfg->batch_size)) return 0;

    if (strcmp(cfg->name, "ppo") == 0) {
        if (!require_discrete("ppo", cfg)) return 0;
        if (!require_positive("steps_per_batch", cfg->steps_per_batch)) return 0;
        if (!require_positive("epochs", cfg->epochs)) return 0;
        if (!require_float_positive("clip_eps", cfg->clip_eps)) return 0;
    } else if (strcmp(cfg->name, "reinforce") == 0) {
        if (!require_discrete("reinforce", cfg)) return 0;
        if (!require_positive("steps_per_batch", cfg->steps_per_batch)) return 0;
    } else if (strcmp(cfg->name, "a2c") == 0 || strcmp(cfg->name, "trpo") == 0 || strcmp(cfg->name, "impala") == 0) {
        if (!require_discrete(cfg->name, cfg)) return 0;
        if (!require_positive("steps_per_batch", cfg->steps_per_batch)) return 0;
    } else if (strcmp(cfg->name, "dqn") == 0 || strcmp(cfg->name, "rainbow") == 0 || strcmp(cfg->name, "qrdqn") == 0) {
        if (cfg->action_type != TFRL_SPACE_DISCRETE) {
            fprintf(stderr, "%s requires discrete action space\n", cfg->name);
            return 0;
        }
        if (!require_positive("batch_size", cfg->batch_size)) return 0;
    } else if (strcmp(cfg->name, "sac") == 0 || strcmp(cfg->name, "td3") == 0) {
        if (!require_positive("batch_size", cfg->batch_size)) return 0;
    } else if (strcmp(cfg->name, "mcts") == 0) {
        if (!require_discrete("mcts", cfg)) return 0;
        if (!require_positive("mcts_sims", cfg->mcts_sims)) return 0;
        if (!require_positive("mcts_depth", cfg->mcts_depth)) return 0;
        if (!cfg->env_name || (strcmp(cfg->env_name, "maze_rooms") != 0 && strcmp(cfg->env_name, "lineworld") != 0)) {
            fprintf(stderr, "mcts requires env maze_rooms or lineworld\n");
            return 0;
        }
    }
    return 1;
}

tfrl_algo tfrl_algo_create(const tfrl_algo_config *cfg, const tfrl_env_spec *spec) {
    tfrl_algo out = {0};
    if (!cfg || !spec) return out;
    if (!validate_algo_config(cfg)) return out;
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
