#ifndef TINYFIN_RL_TRAINER_H
#define TINYFIN_RL_TRAINER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "rl.h"

static inline int tfrl_run_episodes(tfrl_env *env, tfrl_agent *agent, int episodes, double *out_return_mean) {
    double return_sum = 0.0;
    for (int ep = 0; ep < episodes; ep++) {
        tfrl_value obs = {0};
        tfrl_value action = {0};
        tfrl_step step = {0};
        if (tfrl_env_reset(env, &obs) != TFRL_OK) {
            return TFRL_ERR;
        }
        int done = 0;
        while (!done) {
            if (tfrl_agent_act(agent, &obs, &action) != TFRL_OK) {
                return TFRL_ERR;
            }
            if (tfrl_env_step(env, &action, &step) != TFRL_OK) {
                return TFRL_ERR;
            }
            tfrl_transition transition = {
                .observation = obs,
                .action = action,
                .reward = step.reward,
                .next_observation = step.observation,
                .terminated = step.terminated,
                .truncated = step.truncated
            };
            tfrl_agent_observe(agent, &transition);
            obs = step.observation;
            done = tfrl_step_done(&step);
            return_sum += step.reward;
        }
    }
    if (out_return_mean) {
        *out_return_mean = episodes > 0 ? return_sum / episodes : 0.0;
    }
    return TFRL_OK;
}

#ifdef __cplusplus
}
#endif

#endif
