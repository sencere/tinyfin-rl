#include "algo_api.h"

tfrl_algo tfrl_algo_random_create(const tfrl_algo_config *cfg);

static tfrl_algo create_fallback(const tfrl_algo_config *cfg) {
    return tfrl_algo_random_create(cfg);
}

// Entry point expected by the factory / runner.
// Minimal fallback: random policy.
tfrl_algo tfrl_algo_rainbow_create(const tfrl_algo_config *cfg) { return create_fallback(cfg); }
