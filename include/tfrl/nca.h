#ifndef TFRL_PUBLIC_NCA_H
#define TFRL_PUBLIC_NCA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "tfrl/algo.h"

#define TFRL_NCA_VERSION 1
#define TFRL_NCA_MAX_CELLS 128
#define TFRL_NCA_DEFAULT_ROLLOUT_STEPS 4

typedef struct {
    int version;
    int cell_count;
    int action_count;
    int rollout_steps;
    float gamma;
    float learning_rate;
    float epsilon;
} tfrl_nca_policy_desc;

tfrl_algo tfrl_algo_nca_create(const tfrl_algo_config *cfg);

#ifdef __cplusplus
}
#endif

#endif
