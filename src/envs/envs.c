#include "envs_internal.h"

#include <stdlib.h>
#include <string.h>

void tfrl_env_reset_state(tfrl_env *env) {
    if (!env) return;
    env->x = 0;
    env->y = 0;
    env->x2 = 1;
    env->y2 = 0;
    env->pos = -1.0f;
    env->steps = 0;
    for (int i = 0; i < COIN_MAZE_COINS; i++) {
        env->coins_collected[i] = 0;
    }
    env->coins_x[0] = 1;
    env->coins_y[0] = 1;
    env->coins_x[1] = 3;
    env->coins_y[1] = 2;
    env->coins_x[2] = 6;
    env->coins_y[2] = 5;
    env->coins_x[3] = 8;
    env->coins_y[3] = 7;
    env->last_reward = 0.0f;
    env->last_done = 0;
    env->frame_index = 0;
}

tfrl_env *tfrl_env_create(const tfrl_env_config *cfg) {
    tfrl_env *env = (tfrl_env *)calloc(1, sizeof(tfrl_env));
    if (!env) return NULL;
    if (cfg && cfg->name && strncmp(cfg->name, "py:", 3) == 0) {
        env->kind = TFRL_ENV_PYBRIDGE;
        const char *spec = cfg->name + 3;
        const char *kind = spec;
        const char *env_id = strchr(spec, ':');
        if (!env_id) {
            free(env);
            return NULL;
        }
        size_t kind_len = (size_t)(env_id - spec);
        char kind_buf[32];
        if (kind_len >= sizeof(kind_buf)) {
            free(env);
            return NULL;
        }
        memcpy(kind_buf, kind, kind_len);
        kind_buf[kind_len] = '\0';
        env_id += 1;
        const char *socket_path = getenv("TFRL_PY_BRIDGE");
        if (!socket_path) socket_path = "/tmp/tfrl_py_bridge.sock";
        if (!tfrl_env_py_connect(env, socket_path, kind_buf, env_id, cfg ? cfg->seed : 0)) {
            free(env);
            return NULL;
        }
        return env;
    }
    env->ops = tfrl_env_find_ops(cfg ? cfg->name : NULL);
    env->kind = env->ops ? env->ops->kind : TFRL_ENV_MAZE;
    tfrl_env_reset(env, cfg ? cfg->seed : 0);
    return env;
}

void tfrl_env_destroy(tfrl_env *env) {
    if (env && env->kind == TFRL_ENV_PYBRIDGE) {
        tfrl_env_py_close(env);
    }
    free(env);
}

tfrl_obs tfrl_env_reset(tfrl_env *env, uint64_t seed) {
    if (!env) return (tfrl_obs){0};
    if (env->kind == TFRL_ENV_PYBRIDGE) {
        int count = tfrl_env_agent_count(env);
        if (count < 1) count = 1;
        tfrl_obs out[count];
        for (int i = 0; i < count; i++) out[i] = (tfrl_obs){0};
        if (tfrl_env_reset_multi(env, seed, out, count) >= 1) {
            return out[0];
        }
        return (tfrl_obs){0};
    }
    if (!env->ops || !env->ops->reset) return (tfrl_obs){0};
    env->rng_state = (uint32_t)(seed ^ 0xA5A5A5A5u);
    return env->ops->reset(env, seed);
}

tfrl_step_result tfrl_env_step(tfrl_env *env, tfrl_action action) {
    if (!env) return (tfrl_step_result){0};
    if (env->kind == TFRL_ENV_PYBRIDGE) {
        tfrl_step_result out = {0};
        tfrl_step_result steps[1] = {0};
        if (tfrl_env_step_multi(env, &action, 1, steps, 1) == 1) {
            out = steps[0];
        }
        return out;
    }
    if (!env->ops || !env->ops->step) return (tfrl_step_result){0};
    return env->ops->step(env, action);
}

const tfrl_env_spec *tfrl_env_get_spec(const tfrl_env *env) {
    if (!env) return NULL;
    if (env->kind == TFRL_ENV_PYBRIDGE) {
        return &env->py_spec;
    }
    if (!env->ops || !env->ops->spec) return NULL;
    return env->ops->spec();
}

int tfrl_env_agent_count(const tfrl_env *env) {
    if (!env) return 0;
    if (env->kind == TFRL_ENV_PYBRIDGE) return env->py_agent_count;
    if (!env->ops) return 0;
    return env->ops->agent_count;
}

int tfrl_env_reset_multi(tfrl_env *env, uint64_t seed, tfrl_obs *out_obs, int max_agents) {
    if (!env || !out_obs || max_agents <= 0) return 0;
    if (env->kind == TFRL_ENV_PYBRIDGE) {
        return tfrl_env_py_reset_multi(env, seed, out_obs, max_agents);
    }
    int count = tfrl_env_agent_count(env);
    if (max_agents < count) return 0;
    if (count == 1) {
        out_obs[0] = tfrl_env_reset(env, seed);
        return 1;
    }
    out_obs[0] = tfrl_env_reset(env, seed);
    if (env->kind == TFRL_ENV_COIN_MAZE_DUO) {
        out_obs[1].index = env->y2 * COIN_MAZE_W + env->x2;
    } else {
        out_obs[1].index = env->x2;
    }
    return count;
}

int tfrl_env_step_multi(tfrl_env *env, const tfrl_action *actions, int action_count,
                        tfrl_step_result *out_steps, int max_agents) {
    if (!env || !actions || !out_steps || max_agents <= 0) return 0;
    if (env->kind == TFRL_ENV_PYBRIDGE) {
        return tfrl_env_py_step_multi(env, actions, action_count, out_steps, max_agents);
    }
    int count = tfrl_env_agent_count(env);
    if (action_count < count || max_agents < count) return 0;
    if (count == 1) {
        out_steps[0] = tfrl_env_step(env, actions[0]);
        return 1;
    }
    if (!env->ops || !env->ops->step_multi) return 0;
    return env->ops->step_multi(env, actions, action_count, out_steps, max_agents);
}

void tfrl_env_step_batch(tfrl_env **restrict envs, int env_count,
                         const tfrl_action *restrict actions, tfrl_step_result *restrict out_steps) {
    if (!envs || !actions || !out_steps || env_count <= 0) return;
    int i = 0;
    while (i < env_count) {
        tfrl_env *env = envs[i];
        if (!env || !env->ops || env->kind == TFRL_ENV_PYBRIDGE || !env->ops->step) {
            out_steps[i] = tfrl_env_step(env, actions[i]);
            i++;
            continue;
        }
        const tfrl_env_ops *ops = env->ops;
        tfrl_env_kind kind = env->kind;
        int j = i;
        while (j < env_count) {
            tfrl_env *cur = envs[j];
            if (!cur || cur->ops != ops || cur->kind != kind || cur->kind == TFRL_ENV_PYBRIDGE || !ops->step) {
                break;
            }
            out_steps[j] = ops->step(cur, actions[j]);
            j++;
        }
        if (j == i) {
            out_steps[i] = tfrl_env_step(env, actions[i]);
            i++;
            continue;
        }
        i = j;
    }
}

void tfrl_env_reset_batch(tfrl_env **restrict envs, int env_count, uint64_t seed_base,
                          tfrl_obs *restrict out_obs) {
    if (!envs || !out_obs || env_count <= 0) return;
    int i = 0;
    while (i < env_count) {
        tfrl_env *env = envs[i];
        if (!env || !env->ops || env->kind == TFRL_ENV_PYBRIDGE || !env->ops->reset) {
            out_obs[i] = tfrl_env_reset(env, seed_base + (uint64_t)i);
            i++;
            continue;
        }
        const tfrl_env_ops *ops = env->ops;
        tfrl_env_kind kind = env->kind;
        int j = i;
        while (j < env_count) {
            tfrl_env *cur = envs[j];
            if (!cur || cur->ops != ops || cur->kind != kind || cur->kind == TFRL_ENV_PYBRIDGE || !ops->reset) {
                break;
            }
            out_obs[j] = ops->reset(cur, seed_base + (uint64_t)j);
            j++;
        }
        if (j == i) {
            out_obs[i] = tfrl_env_reset(env, seed_base + (uint64_t)i);
            i++;
            continue;
        }
        i = j;
    }
}

void tfrl_env_reset_batch_seeds(tfrl_env **restrict envs, int env_count,
                                const uint64_t *restrict seeds, tfrl_obs *restrict out_obs) {
    if (!envs || !seeds || !out_obs || env_count <= 0) return;
    int i = 0;
    while (i < env_count) {
        tfrl_env *env = envs[i];
        if (!env || !env->ops || env->kind == TFRL_ENV_PYBRIDGE || !env->ops->reset) {
            out_obs[i] = tfrl_env_reset(env, seeds[i]);
            i++;
            continue;
        }
        const tfrl_env_ops *ops = env->ops;
        tfrl_env_kind kind = env->kind;
        int j = i;
        while (j < env_count) {
            tfrl_env *cur = envs[j];
            if (!cur || cur->ops != ops || cur->kind != kind || cur->kind == TFRL_ENV_PYBRIDGE || !ops->reset) {
                break;
            }
            out_obs[j] = ops->reset(cur, seeds[j]);
            j++;
        }
        if (j == i) {
            out_obs[i] = tfrl_env_reset(env, seeds[i]);
            i++;
            continue;
        }
        i = j;
    }
}
