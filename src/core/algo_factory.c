#include <stdio.h>
#include <string.h>

#include "algo_api.h"

static void apply_algo_defaults(tfrl_algo_config *cfg);

static const int DEFAULTS_VERSION = 1;

tfrl_algo tfrl_algo_random_create(const tfrl_algo_config *cfg);

void tfrl_algo_config_apply_defaults(tfrl_algo_config *cfg) {
    apply_algo_defaults(cfg);
}

static void apply_algo_defaults(tfrl_algo_config *cfg) {
    if (!cfg) return;
    if (cfg->defaults_version <= 0) cfg->defaults_version = DEFAULTS_VERSION;
    if (cfg->gamma <= 0.0f) cfg->gamma = 0.99f;
    if (cfg->lr < 0.0f) cfg->lr = 0.0005f;
    if (cfg->epsilon < 0.0f) cfg->epsilon = 0.1f;
    if (cfg->entropy_coef < 0.0f) cfg->entropy_coef = 0.0f;
    if (cfg->clip_eps < 0.0f) cfg->clip_eps = 0.2f;
    if (cfg->gae_lambda < 0.0f) cfg->gae_lambda = 0.95f;
    if (cfg->replay_size < 0) cfg->replay_size = 1000;
    if (cfg->batch_size < 0) cfg->batch_size = 32;
    if (cfg->train_every < 0) cfg->train_every = 1;
    if (cfg->learning_starts < 0) cfg->learning_starts = 0;
    if (cfg->grad_steps < 0) cfg->grad_steps = 1;
    if (cfg->per_alpha < 0.0f) cfg->per_alpha = 0.0f;
    if (cfg->per_beta < 0.0f) cfg->per_beta = 0.4f;
    if (cfg->steps_per_batch < 0) cfg->steps_per_batch = 64;
    if (cfg->epochs < 0) cfg->epochs = 2;
    if (cfg->c51_atoms < 0) cfg->c51_atoms = 51;
    if (cfg->c51_vmin < -1.0e20f) cfg->c51_vmin = -10.0f;
    if (cfg->c51_vmax < -1.0e20f) cfg->c51_vmax = 10.0f;
    if (cfg->iqn_quantiles < 0) cfg->iqn_quantiles = 32;
    if (cfg->iqn_tau_samples < 0) cfg->iqn_tau_samples = 32;
}

tfrl_algo tfrl_algo_create(const tfrl_algo_config *cfg, const tfrl_env_spec *spec) {
    (void)spec;
    tfrl_algo out = {0};
    if (!cfg) return out;
    // Minimal factory: use random policy for all algorithms.
    out = tfrl_algo_random_create(cfg);
    if (!out.vtable) {
        fprintf(stderr, "algo_factory: failed to create algo '%s'\n", cfg->name ? cfg->name : "(null)");
    }
    return out;
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

void tfrl_algo_act_batch(tfrl_algo *algo, const tfrl_obs *obs, int count,
                         tfrl_action *out_actions) {
    if (!algo || !algo->vtable || !algo->vtable->act_batch || !out_actions) return;
    algo->vtable->act_batch(algo->ctx, obs, count, out_actions);
}

int tfrl_algo_diagnostics(tfrl_algo *algo, char *buffer, size_t buffer_len) {
    if (!algo || !algo->vtable || !algo->vtable->diagnostics) return 0;
    return algo->vtable->diagnostics(algo->ctx, buffer, buffer_len);
}
