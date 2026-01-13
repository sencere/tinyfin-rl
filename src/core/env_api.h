#ifndef TFRL_ENV_API_H
#define TFRL_ENV_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int index;
} tfrl_obs;

typedef struct {
    int index;
} tfrl_action;

typedef struct {
    tfrl_obs observation;
    double reward;
    int done;
} tfrl_step_result;

typedef struct {
    const char *name;
    int obs_n;
    int action_n;
    int max_steps;
    int width;
    int height;
} tfrl_env_spec;

typedef struct {
    const char *name;
    uint64_t seed;
} tfrl_env_config;

typedef struct tfrl_env tfrl_env;

tfrl_env *tfrl_env_create(const tfrl_env_config *cfg);
void tfrl_env_destroy(tfrl_env *env);
tfrl_obs tfrl_env_reset(tfrl_env *env, uint64_t seed);
tfrl_step_result tfrl_env_step(tfrl_env *env, tfrl_action action);
const tfrl_env_spec *tfrl_env_get_spec(const tfrl_env *env);

void tfrl_env_step_batch(tfrl_env **envs, int env_count, const tfrl_action *actions, tfrl_step_result *out_steps);

size_t tfrl_env_render_bytes_needed(const tfrl_env *env);
size_t tfrl_env_render_write(tfrl_env *env, void *buffer, size_t buffer_len);

#ifdef __cplusplus
}
#endif

#endif
