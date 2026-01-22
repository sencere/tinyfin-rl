#ifndef TFRL_ALGO_API_H
#define TFRL_ALGO_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include "env_api.h"

typedef struct {
    const char *name;
    int obs_n;
    int action_n;
    tfrl_space_type obs_type;
    tfrl_space_type action_type;
    int obs_dims;
    int action_dims;
    int obs_shape[2];
    int action_shape[2];
    tfrl_dtype obs_dtype;
    tfrl_dtype action_dtype;
    double obs_low;
    double obs_high;
    double action_low;
    double action_high;
    int defaults_version;
    int seed;
    int deterministic;
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
    const char *save_path;
    const char *load_path;
    const char *env_name;
} tfrl_algo_config;

typedef struct {
    tfrl_obs obs;
    tfrl_action action;
    double reward;
    tfrl_obs next_obs;
    int done;
    int policy_version;
} tfrl_transition;

typedef struct tfrl_algo_vtable {
    tfrl_action (*act)(void *ctx, tfrl_obs obs);
    tfrl_action (*act_env)(void *ctx, tfrl_env *env, tfrl_obs obs);
    void (*act_batch)(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions);
    void (*update)(void *ctx, const tfrl_transition *transition);
    void (*save)(void *ctx, const char *path);
    void (*destroy)(void *ctx);
    int (*diagnostics)(void *ctx, char *buffer, size_t buffer_len);
} tfrl_algo_vtable;

typedef struct {
    void *ctx;
    const tfrl_algo_vtable *vtable;
} tfrl_algo;

tfrl_algo tfrl_algo_create(const tfrl_algo_config *cfg, const tfrl_env_spec *spec);
void tfrl_algo_config_apply_defaults(tfrl_algo_config *cfg);
void tfrl_algo_destroy(tfrl_algo *algo);
void tfrl_algo_save(tfrl_algo *algo, const char *path);
static inline tfrl_action tfrl_algo_act(tfrl_algo *algo, tfrl_env *env, tfrl_obs obs) {
    if (!algo || !algo->vtable) return (tfrl_action){0};
    if (env && algo->vtable->act_env) return algo->vtable->act_env(algo->ctx, env, obs);
    if (algo->vtable->act) return algo->vtable->act(algo->ctx, obs);
    return (tfrl_action){0};
}
void tfrl_algo_act_batch(tfrl_algo *algo, const tfrl_obs *obs, int count, tfrl_action *out_actions);
int tfrl_algo_diagnostics(tfrl_algo *algo, char *buffer, size_t buffer_len);

#ifdef __cplusplus
}
#endif

#endif
