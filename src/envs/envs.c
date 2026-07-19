#include "envs/envs_internal.h"

#ifndef TFRL_ENV_NO_DYNAMIC
#include <dlfcn.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TFRL_ENV_NO_PYBRIDGE
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
#endif

#ifndef TFRL_ENV_NO_DYNAMIC
typedef tfrl_env *(*tfrl_env_create_dyn_fn)(const tfrl_env_config *cfg);

typedef struct {
    char name[128];
    char library[256];
} tfrl_env_manifest;

static int looks_like_env_library_path(const char *name) {
    if (!name || !name[0]) return 0;
    if (strchr(name, '/')) return 1;
    size_t len = strlen(name);
    return len > 3 && strcmp(name + len - 3, ".so") == 0;
}

static int str_ends_with(const char *s, const char *suffix) {
    if (!s || !suffix) return 0;
    size_t slen = strlen(s);
    size_t suffix_len = strlen(suffix);
    return slen >= suffix_len && strcmp(s + slen - suffix_len, suffix) == 0;
}

static int read_toml_string(const char *line, const char *key, char *out, size_t out_len) {
    size_t key_len = strlen(key);
    const char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (strncmp(p, key, key_len) != 0) return 0;
    p += key_len;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '=') return 0;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return 0;
    p++;
    const char *end = strchr(p, '"');
    if (!end) return 0;
    size_t len = (size_t)(end - p);
    if (len >= out_len) len = out_len - 1;
    memcpy(out, p, len);
    out[len] = '\0';
    return len > 0;
}

static int read_env_manifest(const char *path, tfrl_env_manifest *out) {
    if (!path || !out) return 0;
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;
    memset(out, 0, sizeof(*out));
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        (void)read_toml_string(line, "name", out->name, sizeof(out->name));
        (void)read_toml_string(line, "library", out->library, sizeof(out->library));
    }
    fclose(fp);
    return out->name[0] && out->library[0];
}

static void path_dirname(const char *path, char *out, size_t out_len) {
    if (!out || out_len == 0) return;
    out[0] = '\0';
    if (!path) return;
    const char *slash = strrchr(path, '/');
    if (!slash) {
        snprintf(out, out_len, ".");
        return;
    }
    size_t len = (size_t)(slash - path);
    if (len >= out_len) len = out_len - 1;
    memcpy(out, path, len);
    out[len] = '\0';
}

static int file_exists(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;
    fclose(fp);
    return 1;
}

static void resolve_manifest_library(const char *manifest_path, const char *library,
                                     char *out, size_t out_len) {
    if (!out || out_len == 0) return;
    out[0] = '\0';
    if (!library || !library[0]) return;
    if (strchr(library, '/')) {
        snprintf(out, out_len, "%s", library);
        return;
    }

    char candidate[512];
    snprintf(candidate, sizeof(candidate), "build/%s", library);
    if (file_exists(candidate)) {
        snprintf(out, out_len, "%s", candidate);
        return;
    }

    char dir[256];
    path_dirname(manifest_path, dir, sizeof(dir));
    snprintf(candidate, sizeof(candidate), "%s/%s", dir, library);
    if (file_exists(candidate)) {
        snprintf(out, out_len, "%s", candidate);
        return;
    }

    snprintf(out, out_len, "%s", library);
}

static tfrl_env *tfrl_env_create_from_library_path(const char *library_path,
                                                   const tfrl_env_config *cfg,
                                                   const char *env_name) {
    void *handle = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) return NULL;
    dlerror();
    tfrl_env_create_dyn_fn create_env =
        (tfrl_env_create_dyn_fn)dlsym(handle, "tfrl_env_create");
    const char *err = dlerror();
    if (err || !create_env) {
        dlclose(handle);
        return NULL;
    }

    tfrl_env_config module_cfg = *cfg;
    if (env_name && env_name[0]) module_cfg.name = env_name;
    tfrl_env *env = create_env(&module_cfg);
    if (!env) {
        dlclose(handle);
        return NULL;
    }
    env->module_handle = handle;
    return env;
}

static tfrl_env *tfrl_env_create_from_library(const tfrl_env_config *cfg) {
    const char *base = strrchr(cfg->name, '/');
    base = base ? base + 1 : cfg->name;
    char inferred_name[128] = {0};
    const char *prefix = "libtfrl_env_";
    const char *suffix = ".so";
    size_t prefix_len = strlen(prefix);
    size_t base_len = strlen(base);
    size_t suffix_len = strlen(suffix);
    if (base_len > prefix_len + suffix_len &&
        strncmp(base, prefix, prefix_len) == 0 &&
        strcmp(base + base_len - suffix_len, suffix) == 0) {
        size_t name_len = base_len - prefix_len - suffix_len;
        if (name_len >= sizeof(inferred_name)) name_len = sizeof(inferred_name) - 1;
        memcpy(inferred_name, base + prefix_len, name_len);
        inferred_name[name_len] = '\0';
    }
    return tfrl_env_create_from_library_path(cfg->name, cfg,
                                             inferred_name[0] ? inferred_name : NULL);
}

static tfrl_env *tfrl_env_create_from_manifest(const char *manifest_path,
                                               const tfrl_env_config *cfg) {
    tfrl_env_manifest manifest;
    if (!read_env_manifest(manifest_path, &manifest)) return NULL;
    char library_path[512];
    resolve_manifest_library(manifest_path, manifest.library, library_path, sizeof(library_path));
    if (!library_path[0]) return NULL;
    return tfrl_env_create_from_library_path(library_path, cfg, manifest.name);
}

static tfrl_env *tfrl_env_create_from_manifest_name(const tfrl_env_config *cfg) {
    char manifest_path[512];
    snprintf(manifest_path, sizeof(manifest_path), "envs/%s/env.toml", cfg->name);
    if (!file_exists(manifest_path)) return NULL;
    return tfrl_env_create_from_manifest(manifest_path, cfg);
}
#endif

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
    static const char *ROWS[MAZE_H] = {
        "########",
        "#......#",
        "#.####.#",
        "#.#....#",
        "#.#.##.#",
        "#...#..#",
        "###....#",
        "########",
    };
    if (x < 0 || y < 0 || x >= MAZE_W || y >= MAZE_H) return 1;
    return ROWS[y][x] == '#';
}

tfrl_env *tfrl_env_create(const tfrl_env_config *cfg) {
    if (!cfg || !cfg->name) return NULL;
#ifndef TFRL_ENV_NO_DYNAMIC
    if (str_ends_with(cfg->name, ".toml")) {
        tfrl_env *env = tfrl_env_create_from_manifest(cfg->name, cfg);
        if (env) return env;
    }
    if (looks_like_env_library_path(cfg->name)) {
        return tfrl_env_create_from_library(cfg);
    }
    {
        tfrl_env *env = tfrl_env_create_from_manifest_name(cfg);
        if (env) return env;
    }
#endif
    tfrl_env *env = (tfrl_env *)calloc(1, sizeof(tfrl_env));
    if (!env) return NULL;
    env->rng_state = (uint32_t)(cfg->seed ? cfg->seed : 1u);

#ifndef TFRL_ENV_NO_PYBRIDGE
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
#else
    if (strncmp(cfg->name, "py:", 3) == 0) {
        free(env);
        return NULL;
    }
#endif

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
#ifndef TFRL_ENV_NO_PYBRIDGE
    if (env->py_fp) {
        tfrl_env_py_close(env);
    }
#endif
#ifndef TFRL_ENV_NO_DYNAMIC
    void *module_handle = env->module_handle;
    free(env);
    if (module_handle) {
        dlclose(module_handle);
    }
#else
    free(env);
#endif
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
#ifndef TFRL_ENV_NO_PYBRIDGE
    if (env->kind == TFRL_ENV_PYBRIDGE) {
        int got = tfrl_env_py_reset_multi(env, seed, &obs, 1);
        if (got <= 0) return (tfrl_obs){0};
        return obs;
    }
#endif
    if (!env->ops || !env->ops->reset) return obs;
    return env->ops->reset(env, seed);
}

tfrl_step_result tfrl_env_step(tfrl_env *env, tfrl_action action) {
    tfrl_step_result out = {0};
    if (!env) return out;
#ifndef TFRL_ENV_NO_PYBRIDGE
    if (env->kind == TFRL_ENV_PYBRIDGE) {
        tfrl_step_result steps[1] = {0};
        int got = tfrl_env_py_step_multi(env, &action, 1, steps, 1);
        if (got <= 0) return out;
        return steps[0];
    }
#endif
    if (!env->ops || !env->ops->step) return out;
    return env->ops->step(env, action);
}

int tfrl_env_reset_multi(tfrl_env *env, uint64_t seed, tfrl_obs *out_obs, int max_agents) {
    if (!env || !out_obs || max_agents <= 0) return 0;
#ifndef TFRL_ENV_NO_PYBRIDGE
    if (env->kind == TFRL_ENV_PYBRIDGE) {
        return tfrl_env_py_reset_multi(env, seed, out_obs, max_agents);
    }
#endif
    int agent_count = tfrl_env_agent_count(env);
    if (agent_count > max_agents) return 0;
    tfrl_obs obs = tfrl_env_reset(env, seed);
    for (int i = 0; i < agent_count; i++) {
        out_obs[i] = obs;
    }
    if (env->kind == TFRL_ENV_LINEWORLD_DUO && agent_count >= 2) {
        out_obs[0].index = env->x;
        out_obs[1].index = env->x2;
    }
    return agent_count;
}

int tfrl_env_step_multi(tfrl_env *env, const tfrl_action *actions, int action_count,
                        tfrl_step_result *out_steps, int max_agents) {
    if (!env || !actions || !out_steps || max_agents <= 0) return 0;
#ifndef TFRL_ENV_NO_PYBRIDGE
    if (env->kind == TFRL_ENV_PYBRIDGE) {
        return tfrl_env_py_step_multi(env, actions, action_count, out_steps, max_agents);
    }
#endif
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
