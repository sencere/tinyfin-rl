#ifndef TINYFIN_RL_TRAIN_H
#define TINYFIN_RL_TRAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "rl.h"

typedef struct {
    int32_t obs_n;
    int32_t action_n;
    int32_t steps;
    int32_t eval_steps;
    float gamma;
    float lr;
    float epsilon;
    const char *save_path;
    const char *load_path;
} tfrl_dqn_config;

typedef struct {
    int32_t obs_n;
    int32_t action_n;
    int32_t steps_per_batch;
    int32_t updates;
    int32_t epochs;
    int32_t eval_steps;
    float gamma;
    float lr;
    float clip_eps;
    const char *save_path;
    const char *load_path;
} tfrl_ppo_config;

typedef struct {
    float return_mean;
    float loss;
} tfrl_train_metrics;

int tfrl_train_dqn(tfrl_env *env, const tfrl_dqn_config *cfg, tfrl_train_metrics *out_metrics);
int tfrl_train_ppo(tfrl_env *env, const tfrl_ppo_config *cfg, tfrl_train_metrics *out_metrics);

#ifdef __cplusplus
}
#endif

#endif
