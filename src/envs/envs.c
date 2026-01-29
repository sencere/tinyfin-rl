#include "envs/envs_internal.h"

#include <stdlib.h>
#include <string.h>

static int parse_py_env(const char *name, char *out_kind, size_t kind_len,
                        char *out_id, size_t id_len) {
    if (!name || strncmp(name, "py:", 3) != 0) return 0;
    const char *p = name + 3;
    const char *colon = strchr(p, ':');
    if (!colon) return 0;
    size_t klen = (size_t)(colon - p);
    size_t idlen = strlen(colon + 1);
    if (klen == 0 || idlen == 0) return 0;
    if (klen >= kind_len) klen = kind_len - 1;
    if (idlen >= id_len) idlen = id_len - 1;
    memcpy(out_kind, p, klen);
    out_kind[klen] = '\0';
    memcpy(out_id, colon + 1, idlen);
    out_id[idlen] = '\0';
    return 1;
}

void tfrl_env_reset_state(tfrl_env *env) {
    if (!env) return;
    env->x = 0;
    env->y = 0;
    env->x2 = 0;
    env->y2 = 0;
    env->pos = 0.0f;
    env->steps = 0;
    env->last_reward = 0.0f;
    env->last_done = 0;
    env->frame_index = 0;
    for (int i = 0; i < COIN_MAZE_COINS; i++) {
        env->coins_x[i] = 0;
        env->coins_y[i] = 0;
        env->coins_collected[i] = 0;
    }
}

int tfrl_env_is_wall_maze(int x, int y) {
    if (x < 0 || y < 0 || x >= MAZE_W || y >= MAZE_H) return 1;
    // Simple checkerboard walls for variety.
    return ((x + y) % 4) == 1;
}

tfrl_env *tfrl_env_create(const tfrl_env_config *cfg) {
    if (!cfg || !cfg->name) return NULL;
    tfrl_env *env = (tfrl_env *)calloc(1, sizeof(tfrl_env));
    if (!env) return NULL;
    env->rng_state = (uint32_t)(cfg->seed ? cfg->seed : 1u);

    char kind[64];
    char env_id[64];
    if (parse_py_env(cfg->name, kind, sizeof(kind), env_id, sizeof(env_id))) {
        env->kind = TFRL_ENV_PYBRIDGE;
        env->ops = NULL;
        env->py_fd = -1;
        env->py_fp = NULL;
        if (!tfrl_env_py_connect(env, "/tmp/tfrl_bridge.sock", kind, env_id, cfg->seed)) {
            free(env);
            return NULL;
        }
        return env;
    }

    const tfrl_env_ops *ops = tfrl_env_find_ops(cfg->name);
    if (!ops) {
        free(env);
        return NULL;
    }
    env->kind = ops->kind;
    env->ops = ops;
    env->py_fd = -1;
    env->py_fp = NULL;
    return env;
}

void tfrl_env_destroy(tfrl_env *env) {
    if (!env) return;
    if (env->py_fp) {
        tfrl_env_py_close(env);
    }
    free(env);
}

const tfrl_env_spec *tfrl_env_get_spec(const tfrl_env *env) {
    if (!env) return NULL;
    if (env->kind == TFRL_ENV_PYBRIDGE) return &env->py_spec;
    if (!env->ops || !env->ops->spec) return NULL;
    return env->ops->spec();
}

int tfrl_env_agent_count(const tfrl_env *env) {
    const tfrl_env_spec *spec = tfrl_env_get_spec(env);
    return spec ? spec->agent_count : 1;
}

size_t tfrl_env_state_size(const tfrl_env *env) {
    (void)env;
    return sizeof(tfrl_env);
}

int tfrl_env_state_save(const tfrl_env *env, void *buffer, size_t buffer_len) {
    if (!env || !buffer || buffer_len < sizeof(tfrl_env)) return 0;
    memcpy(buffer, env, sizeof(tfrl_env));
    return 1;
}

int tfrl_env_state_load(tfrl_env *env, const void *buffer, size_t buffer_len) {
    if (!env || !buffer || buffer_len < sizeof(tfrl_env)) return 0;
    memcpy(env, buffer, sizeof(tfrl_env));
    return 1;
}

tfrl_obs tfrl_env_reset(tfrl_env *env, uint64_t seed) {
    tfrl_obs obs = {0};
    if (!env) return obs;
    if (env->kind == TFRL_ENV_PYBRIDGE) {
        int got = tfrl_env_py_reset_multi(env, seed, &obs, 1);
        if (got <= 0) return (tfrl_obs){0};
        return obs;
    }
    if (!env->ops || !env->ops->reset) return obs;
    return env->ops->reset(env, seed);
}

tfrl_step_result tfrl_env_step(tfrl_env *env, tfrl_action action) {
    tfrl_step_result out = {0};
    if (!env) return out;
    if (env->kind == TFRL_ENV_PYBRIDGE) {
        tfrl_step_result steps[1] = {0};
        int got = tfrl_env_py_step_multi(env, &action, 1, steps, 1);
        if (got <= 0) return out;
        return steps[0];
    }
    if (!env->ops || !env->ops->step) return out;
    return env->ops->step(env, action);
}

int tfrl_env_reset_multi(tfrl_env *env, uint64_t seed, tfrl_obs *out_obs, int max_agents) {
    if (!env || !out_obs || max_agents <= 0) return 0;
    if (env->kind == TFRL_ENV_PYBRIDGE) {
        return tfrl_env_py_reset_multi(env, seed, out_obs, max_agents);
    }
    int agent_count = tfrl_env_agent_count(env);
    if (agent_count > max_agents) return 0;
    for (int i = 0; i < agent_count; i++) {
        out_obs[i] = tfrl_env_reset(env, seed + (uint64_t)i);
    }
    return agent_count;
}

int tfrl_env_step_multi(tfrl_env *env, const tfrl_action *actions, int action_count,
                        tfrl_step_result *out_steps, int max_agents) {
    if (!env || !actions || !out_steps || max_agents <= 0) return 0;
    if (env->kind == TFRL_ENV_PYBRIDGE) {
        return tfrl_env_py_step_multi(env, actions, action_count, out_steps, max_agents);
    }
    if (env->ops && env->ops->step_multi) {
        return env->ops->step_multi(env, actions, action_count, out_steps, max_agents);
    }
    int agent_count = tfrl_env_agent_count(env);
    if (agent_count > max_agents || action_count < agent_count) return 0;
    for (int i = 0; i < agent_count; i++) {
        out_steps[i] = tfrl_env_step(env, actions[i]);
    }
    return agent_count;
}

void tfrl_env_step_batch(tfrl_env **restrict envs, int env_count,
                         const tfrl_action *restrict actions,
                         tfrl_step_result *restrict out_steps) {
    if (!envs || !actions || !out_steps || env_count <= 0) return;
    for (int i = 0; i < env_count; i++) {
        out_steps[i] = tfrl_env_step(envs[i], actions[i]);
    }
}

void tfrl_env_reset_batch(tfrl_env **restrict envs, int env_count, uint64_t seed_base,
                          tfrl_obs *restrict out_obs) {
    if (!envs || !out_obs || env_count <= 0) return;
    for (int i = 0; i < env_count; i++) {
        out_obs[i] = tfrl_env_reset(envs[i], seed_base + (uint64_t)i);
    }
}

void tfrl_env_reset_batch_seeds(tfrl_env **restrict envs, int env_count,
                                const uint64_t *restrict seeds, tfrl_obs *restrict out_obs) {
    if (!envs || !out_obs || !seeds || env_count <= 0) return;
    for (int i = 0; i < env_count; i++) {
        out_obs[i] = tfrl_env_reset(envs[i], seeds[i]);
    }
}
