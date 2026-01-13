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
    float gamma;
    float lr;
    float epsilon;
    float entropy_coef;
    float clip_eps;
    int replay_size;
    int batch_size;
    float per_alpha;
    float per_beta;
    int mcts_sims;
    int mcts_depth;
    int steps_per_batch;
    int epochs;
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
} tfrl_transition;

typedef struct tfrl_algo_vtable {
    tfrl_action (*act)(void *ctx, tfrl_obs obs);
    void (*act_batch)(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions);
    void (*update)(void *ctx, const tfrl_transition *transition);
    void (*save)(void *ctx, const char *path);
    void (*destroy)(void *ctx);
} tfrl_algo_vtable;

typedef struct {
    void *ctx;
    const tfrl_algo_vtable *vtable;
} tfrl_algo;

tfrl_algo tfrl_algo_create(const tfrl_algo_config *cfg, const tfrl_env_spec *spec);
void tfrl_algo_destroy(tfrl_algo *algo);
void tfrl_algo_save(tfrl_algo *algo, const char *path);
void tfrl_algo_act_batch(tfrl_algo *algo, const tfrl_obs *obs, int count, tfrl_action *out_actions);

#ifdef __cplusplus
}
#endif

#endif
