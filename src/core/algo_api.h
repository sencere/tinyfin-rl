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
    float gamma;
    float lr;
    float epsilon;
    float clip_eps;
    float entropy_coef;
    int steps_per_batch;
    int epochs;
    const char *save_path;
    const char *load_path;
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

#ifdef __cplusplus
}
#endif

#endif
