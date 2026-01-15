#include <stdio.h>
#include <string.h>

#include "algo_api.h"

tfrl_algo tfrl_algo_random_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_dqn_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_rainbow_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_qrdqn_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_a2c_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_a3c_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_trpo_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_sac_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_td3_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_impala_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_mcts_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_reinforce_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_ppo_create(const tfrl_algo_config *cfg);
tfrl_algo tfrl_algo_iqn_create(const tfrl_algo_config *cfg);

static void apply_algo_defaults(tfrl_algo_config *cfg);

typedef struct {
    const char *name;
    int defaults_version;
    float gamma;
    float lr;
    float epsilon;
    float entropy_coef;
    float clip_eps;
    float kl_target;
    float gae_lambda;
    int replay_size;
    int batch_size;
    int train_every;
    int learning_starts;
    int grad_steps;
    float per_alpha;
    float per_beta;
    int mcts_sims;
    int mcts_depth;
    int steps_per_batch;
    int epochs;
    int c51_atoms;
    float c51_vmin;
    float c51_vmax;
    int iqn_quantiles;
    int iqn_tau_samples;
} tfrl_algo_schema;

static const tfrl_algo_schema ALGO_SCHEMA_DEFAULT = {
    .name = "default",
    .defaults_version = 1,
    .gamma = 0.99f,
    .lr = 0.05f,
    .epsilon = 0.1f,
    .entropy_coef = 0.0f,
    .clip_eps = 0.2f,
    .kl_target = 0.01f,
    .gae_lambda = 0.95f,
    .replay_size = 1000,
    .batch_size = 32,
    .train_every = 1,
    .learning_starts = 0,
    .grad_steps = 1,
    .per_alpha = 0.0f,
    .per_beta = 0.4f,
    .mcts_sims = 200,
    .mcts_depth = 50,
    .steps_per_batch = 64,
    .epochs = 2,
    .c51_atoms = 51,
    .c51_vmin = -10.0f,
    .c51_vmax = 10.0f,
    .iqn_quantiles = 32,
    .iqn_tau_samples = 32,
};

static const tfrl_algo_schema ALGO_SCHEMAS[] = {
    {.name = "dqn", .defaults_version = 1, .gamma = 0.99f, .lr = 0.05f, .epsilon = 0.1f, .replay_size = 1000, .batch_size = 32, .per_alpha = 0.0f, .per_beta = 0.4f},
    {.name = "rainbow", .defaults_version = 1, .gamma = 0.99f, .lr = 0.05f, .epsilon = 0.1f, .replay_size = 1000, .batch_size = 32, .per_alpha = 0.0f, .per_beta = 0.4f, .c51_atoms = 51, .c51_vmin = -10.0f, .c51_vmax = 10.0f},
    {.name = "qrdqn", .defaults_version = 1, .gamma = 0.99f, .lr = 0.05f, .epsilon = 0.1f, .replay_size = 1000, .batch_size = 32, .per_alpha = 0.0f, .per_beta = 0.4f},
    {.name = "iqn", .defaults_version = 1, .gamma = 0.99f, .lr = 0.05f, .epsilon = 0.1f, .replay_size = 1000, .batch_size = 32, .per_alpha = 0.0f, .per_beta = 0.4f, .iqn_quantiles = 32, .iqn_tau_samples = 32},
    {.name = "ppo", .defaults_version = 1, .gamma = 0.99f, .lr = 0.0003f, .entropy_coef = 0.01f, .clip_eps = 0.2f, .kl_target = 0.01f, .gae_lambda = 0.95f, .steps_per_batch = 64, .epochs = 2},
    {.name = "reinforce", .defaults_version = 1, .gamma = 0.99f, .lr = 0.01f, .steps_per_batch = 64},
    {.name = "a2c", .defaults_version = 1, .gamma = 0.99f, .lr = 0.001f, .entropy_coef = 0.01f, .steps_per_batch = 64},
    {.name = "a3c", .defaults_version = 1, .gamma = 0.99f, .lr = 0.001f, .entropy_coef = 0.01f, .steps_per_batch = 64},
    {.name = "trpo", .defaults_version = 1, .gamma = 0.99f, .lr = 0.0003f, .clip_eps = 0.01f, .steps_per_batch = 64},
    {.name = "impala", .defaults_version = 1, .gamma = 0.99f, .lr = 0.0003f, .clip_eps = 1.0f, .entropy_coef = 0.01f, .steps_per_batch = 64},
    {.name = "sac", .defaults_version = 1, .gamma = 0.99f, .lr = 0.0003f, .entropy_coef = 0.2f, .replay_size = 1000, .batch_size = 32, .per_alpha = 0.0f, .per_beta = 0.4f},
    {.name = "td3", .defaults_version = 1, .gamma = 0.99f, .lr = 0.0003f, .replay_size = 1000, .batch_size = 32, .per_alpha = 0.0f, .per_beta = 0.4f},
    {.name = "mcts", .defaults_version = 1, .gamma = 0.99f, .mcts_sims = 200, .mcts_depth = 50},
    {.name = "random", .defaults_version = 1, .gamma = 0.99f},
};

static const tfrl_algo_schema *algo_schema_for(const char *name) {
    if (!name) return &ALGO_SCHEMA_DEFAULT;
    size_t count = sizeof(ALGO_SCHEMAS) / sizeof(ALGO_SCHEMAS[0]);
    for (size_t i = 0; i < count; i++) {
        if (strcmp(ALGO_SCHEMAS[i].name, name) == 0) return &ALGO_SCHEMAS[i];
    }
    return &ALGO_SCHEMA_DEFAULT;
}

void tfrl_algo_config_apply_defaults(tfrl_algo_config *cfg) {
    apply_algo_defaults(cfg);
}

static int float_unset_positive(float v) {
    return v <= 0.0f;
}

static int float_unset_nonneg(float v) {
    return v < 0.0f;
}

static int float_unset_sentinel(float v) {
    return v < -1.0e20f;
}

static int int_unset_positive(int v) {
    return v <= 0;
}

static int int_unset_nonneg(int v) {
    return v < 0;
}

static void apply_algo_defaults(tfrl_algo_config *cfg) {
    if (!cfg) return;
    const tfrl_algo_schema *schema = algo_schema_for(cfg->name);
    cfg->defaults_version = schema->defaults_version;
    if (float_unset_positive(cfg->gamma)) cfg->gamma = schema->gamma;
    if (float_unset_positive(cfg->lr)) cfg->lr = schema->lr;
    if (float_unset_nonneg(cfg->epsilon)) cfg->epsilon = schema->epsilon;
    if (float_unset_nonneg(cfg->entropy_coef)) cfg->entropy_coef = schema->entropy_coef;
    if (float_unset_positive(cfg->clip_eps)) cfg->clip_eps = schema->clip_eps;
    if (float_unset_nonneg(cfg->kl_target)) cfg->kl_target = schema->kl_target;
    if (float_unset_positive(cfg->gae_lambda)) cfg->gae_lambda = schema->gae_lambda;
    if (int_unset_nonneg(cfg->replay_size)) cfg->replay_size = schema->replay_size;
    if (int_unset_nonneg(cfg->batch_size)) cfg->batch_size = schema->batch_size;
    if (int_unset_positive(cfg->train_every)) {
        cfg->train_every = schema->train_every > 0 ? schema->train_every : ALGO_SCHEMA_DEFAULT.train_every;
    }
    if (int_unset_nonneg(cfg->learning_starts)) {
        cfg->learning_starts = schema->learning_starts >= 0 ? schema->learning_starts : ALGO_SCHEMA_DEFAULT.learning_starts;
    }
    if (int_unset_positive(cfg->grad_steps)) {
        cfg->grad_steps = schema->grad_steps > 0 ? schema->grad_steps : ALGO_SCHEMA_DEFAULT.grad_steps;
    }
    if (float_unset_nonneg(cfg->per_alpha)) cfg->per_alpha = schema->per_alpha;
    if (float_unset_nonneg(cfg->per_beta)) cfg->per_beta = schema->per_beta;
    if (int_unset_positive(cfg->mcts_sims)) cfg->mcts_sims = schema->mcts_sims;
    if (int_unset_positive(cfg->mcts_depth)) cfg->mcts_depth = schema->mcts_depth;
    if (int_unset_positive(cfg->steps_per_batch)) cfg->steps_per_batch = schema->steps_per_batch;
    if (int_unset_positive(cfg->epochs)) cfg->epochs = schema->epochs;
    if (int_unset_positive(cfg->c51_atoms)) cfg->c51_atoms = schema->c51_atoms;
    if (float_unset_sentinel(cfg->c51_vmin) || float_unset_sentinel(cfg->c51_vmax)) {
        cfg->c51_vmin = schema->c51_vmin;
        cfg->c51_vmax = schema->c51_vmax;
    }
    if (int_unset_positive(cfg->iqn_quantiles)) cfg->iqn_quantiles = schema->iqn_quantiles;
    if (int_unset_positive(cfg->iqn_tau_samples)) cfg->iqn_tau_samples = schema->iqn_tau_samples;
}

static void log_algo_config(const tfrl_algo_config *cfg) {
    if (!cfg) return;
    fprintf(stdout,
            "algo_config name=%s defaults=v%d gamma=%.3f lr=%.6f epsilon=%.3f entropy=%.3f clip_eps=%.3f kl_target=%.3f "
            "gae_lambda=%.3f replay=%d batch=%d train_every=%d learning_starts=%d grad_steps=%d per_alpha=%.3f per_beta=%.3f steps_per_batch=%d epochs=%d c51_atoms=%d c51_vmin=%.3f c51_vmax=%.3f iqn_quantiles=%d iqn_tau_samples=%d mcts_sims=%d mcts_depth=%d seed=%d deterministic=%d\n",
            cfg->name ? cfg->name : "null",
            cfg->defaults_version,
            cfg->gamma,
            cfg->lr,
            cfg->epsilon,
            cfg->entropy_coef,
            cfg->clip_eps,
            cfg->kl_target,
            cfg->gae_lambda,
            cfg->replay_size,
            cfg->batch_size,
            cfg->train_every,
            cfg->learning_starts,
            cfg->grad_steps,
            cfg->per_alpha,
            cfg->per_beta,
            cfg->steps_per_batch,
            cfg->epochs,
            cfg->c51_atoms,
            cfg->c51_vmin,
            cfg->c51_vmax,
            cfg->iqn_quantiles,
            cfg->iqn_tau_samples,
            cfg->mcts_sims,
            cfg->mcts_depth,
            cfg->seed,
            cfg->deterministic);
}

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

static int require_float_unit(const char *name, float v) {
    if (v > 0.0f && v <= 1.0f) return 1;
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
        if (!require_float_unit("gae_lambda", cfg->gae_lambda)) return 0;
    } else if (strcmp(cfg->name, "reinforce") == 0) {
        if (!require_discrete("reinforce", cfg)) return 0;
        if (!require_positive("steps_per_batch", cfg->steps_per_batch)) return 0;
    } else if (strcmp(cfg->name, "a2c") == 0 || strcmp(cfg->name, "a3c") == 0 || strcmp(cfg->name, "trpo") == 0 || strcmp(cfg->name, "impala") == 0) {
        if (!require_discrete(cfg->name, cfg)) return 0;
        if (!require_positive("steps_per_batch", cfg->steps_per_batch)) return 0;
    } else if (strcmp(cfg->name, "dqn") == 0 || strcmp(cfg->name, "rainbow") == 0 || strcmp(cfg->name, "qrdqn") == 0 || strcmp(cfg->name, "iqn") == 0) {
        if (cfg->action_type != TFRL_SPACE_DISCRETE) {
            fprintf(stderr, "%s requires discrete action space\n", cfg->name);
            return 0;
        }
        if (!require_positive("batch_size", cfg->batch_size)) return 0;
        if (!require_positive("train_every", cfg->train_every)) return 0;
        if (!require_nonneg("learning_starts", cfg->learning_starts)) return 0;
        if (!require_positive("grad_steps", cfg->grad_steps)) return 0;
        if ((strcmp(cfg->name, "rainbow") == 0) && !require_positive("c51_atoms", cfg->c51_atoms)) return 0;
        if ((strcmp(cfg->name, "rainbow") == 0) && !(cfg->c51_vmax > cfg->c51_vmin)) {
            fprintf(stderr, "invalid c51 support: vmin=%.3f vmax=%.3f\n", cfg->c51_vmin, cfg->c51_vmax);
            return 0;
        }
        if ((strcmp(cfg->name, "iqn") == 0) && !require_positive("iqn_quantiles", cfg->iqn_quantiles)) return 0;
        if ((strcmp(cfg->name, "iqn") == 0) && !require_positive("iqn_tau_samples", cfg->iqn_tau_samples)) return 0;
    } else if (strcmp(cfg->name, "sac") == 0 || strcmp(cfg->name, "td3") == 0) {
        if (!require_positive("batch_size", cfg->batch_size)) return 0;
        if (!require_positive("train_every", cfg->train_every)) return 0;
        if (!require_nonneg("learning_starts", cfg->learning_starts)) return 0;
        if (!require_positive("grad_steps", cfg->grad_steps)) return 0;
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
    tfrl_algo_config merged = *cfg;
    if (merged.defaults_version <= 0) {
        apply_algo_defaults(&merged);
    }
    log_algo_config(&merged);
    if (!validate_algo_config(&merged)) return out;
    const tfrl_algo_config *cfg_use = &merged;
    if (!cfg_use->name) {
        return tfrl_algo_random_create(cfg_use);
    }
    if (strcmp(cfg_use->name, "random") == 0) {
        return tfrl_algo_random_create(cfg_use);
    }
    if (cfg_use->action_type == TFRL_SPACE_BOX) {
        if (strcmp(cfg_use->name, "sac") == 0 || strcmp(cfg_use->name, "td3") == 0) {
            return strcmp(cfg_use->name, "sac") == 0 ? tfrl_algo_sac_create(cfg_use) : tfrl_algo_td3_create(cfg_use);
        }
        fprintf(stderr, "algo '%s' does not support continuous actions yet; using random policy\n", cfg_use->name);
        return tfrl_algo_random_create(cfg_use);
    }
    if (strcmp(cfg_use->name, "mcts") == 0) {
        tfrl_algo algo = tfrl_algo_mcts_create(cfg_use);
        if (!algo.ctx) {
            fprintf(stderr, "mcts requires maze_rooms or lineworld; using random policy\n");
            return tfrl_algo_random_create(cfg_use);
        }
        return algo;
    }
    if (strcmp(cfg_use->name, "dqn") == 0) {
        return tfrl_algo_dqn_create(cfg_use);
    }
    if (strcmp(cfg_use->name, "rainbow") == 0) {
        return tfrl_algo_rainbow_create(cfg_use);
    }
    if (strcmp(cfg_use->name, "qrdqn") == 0) {
        return tfrl_algo_qrdqn_create(cfg_use);
    }
    if (strcmp(cfg_use->name, "a2c") == 0) {
        return tfrl_algo_a2c_create(cfg_use);
    }
    if (strcmp(cfg_use->name, "trpo") == 0) {
        return tfrl_algo_trpo_create(cfg_use);
    }
    if (strcmp(cfg_use->name, "sac") == 0) {
        return tfrl_algo_sac_create(cfg_use);
    }
    if (strcmp(cfg_use->name, "td3") == 0) {
        return tfrl_algo_td3_create(cfg_use);
    }
    if (strcmp(cfg_use->name, "impala") == 0) {
        return tfrl_algo_impala_create(cfg_use);
    }
    if (strcmp(cfg_use->name, "a3c") == 0) {
        return tfrl_algo_a3c_create(cfg_use);
    }
    if (strcmp(cfg_use->name, "reinforce") == 0) {
        return tfrl_algo_reinforce_create(cfg_use);
    }
    if (strcmp(cfg_use->name, "ppo") == 0) {
        return tfrl_algo_ppo_create(cfg_use);
    }
    if (strcmp(cfg_use->name, "iqn") == 0) {
        return tfrl_algo_iqn_create(cfg_use);
    }
    fprintf(stderr, "unknown algo '%s'; using random policy\n", cfg_use->name);
    return tfrl_algo_random_create(cfg_use);
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

int tfrl_algo_diagnostics(tfrl_algo *algo, char *buffer, size_t buffer_len) {
    if (!algo || !algo->vtable || !algo->vtable->diagnostics) return 0;
    return algo->vtable->diagnostics(algo->ctx, buffer, buffer_len);
}
