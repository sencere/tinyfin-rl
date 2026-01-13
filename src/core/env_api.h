#ifndef TFRL_ENV_API_H
#define TFRL_ENV_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define TFRL_MAX_BOX_DIMS 4

typedef struct {
    int index;
    int data_len;
    float data[TFRL_MAX_BOX_DIMS];
} tfrl_obs;

typedef struct {
    int index;
    int data_len;
    float data[TFRL_MAX_BOX_DIMS];
} tfrl_action;

typedef struct {
    tfrl_obs observation;
    double reward;
    int done;
} tfrl_step_result;

typedef enum {
    TFRL_SPACE_DISCRETE = 0,
    TFRL_SPACE_BOX = 1
} tfrl_space_type;

typedef enum {
    TFRL_DTYPE_INT32 = 0,
    TFRL_DTYPE_FLOAT32 = 1
} tfrl_dtype;

typedef struct {
    const char *name;
    int obs_n;
    int action_n;
    int max_steps;
    int width;
    int height;
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
    int agent_count;
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
int tfrl_env_agent_count(const tfrl_env *env);
int tfrl_env_reset_multi(tfrl_env *env, uint64_t seed, tfrl_obs *out_obs, int max_agents);
int tfrl_env_step_multi(tfrl_env *env, const tfrl_action *actions, int action_count, tfrl_step_result *out_steps, int max_agents);

void tfrl_env_step_batch(tfrl_env **envs, int env_count, const tfrl_action *actions, tfrl_step_result *out_steps);

size_t tfrl_env_render_bytes_needed(const tfrl_env *env);
size_t tfrl_env_render_write(tfrl_env *env, void *buffer, size_t buffer_len);

#ifdef __cplusplus
}
#endif

#endif
