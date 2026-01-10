#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyfin_rl/rl.h"
#include "tinyfin_rl/rl_plugin.h"
#include "tinyfin_rl/rl_trainer.h"

static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static double rng_uniform(uint64_t *state) {
    uint64_t v = rng_next(state) >> 11;
    return (double)v * (1.0 / 9007199254740992.0);
}

static int arg_int(int argc, char **argv, const char *name, int def) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0) {
            return atoi(argv[i + 1]);
        }
    }
    return def;
}

typedef struct {
    uint64_t rng;
    int actions;
} random_agent;

static int random_act(void *ctx, const tfrl_value *obs, tfrl_value *out_action) {
    random_agent *agent = (random_agent *)ctx;
    (void)obs;
    int action = (int)(rng_uniform(&agent->rng) * agent->actions) % agent->actions;
    out_action->type = TFRL_VALUE_I64;
    out_action->as.i64 = action;
    return TFRL_OK;
}

static int random_observe(void *ctx, const tfrl_transition *transition) {
    (void)ctx;
    (void)transition;
    return TFRL_OK;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s /path/to/plugin.so [--episodes N] [--actions N]\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    int episodes = arg_int(argc, argv, "--episodes", 5);
    int actions = arg_int(argc, argv, "--actions", 4);

    void *handle = dlopen(path, RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    tfrl_env_plugin_get_fn get_plugin = (tfrl_env_plugin_get_fn)dlsym(handle, "tfrl_env_plugin_get");
    if (!get_plugin) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }
    const tfrl_env_plugin *plugin = get_plugin();
    if (!plugin || plugin->api_version != TFRL_ENV_PLUGIN_API_VERSION) {
        fprintf(stderr, "invalid plugin api\n");
        dlclose(handle);
        return 1;
    }

    tfrl_env env = plugin->create();
    random_agent agent_state = {.rng = 1234, .actions = actions};
    const tfrl_agent_vtable agent_vtable = {
        .act = random_act,
        .observe = random_observe,
        .destroy = NULL
    };
    tfrl_agent agent = {&agent_state, &agent_vtable};
    double mean_return = 0.0;
    if (tfrl_run_episodes(&env, &agent, episodes, &mean_return) != TFRL_OK) {
        fprintf(stderr, "train loop failed\n");
        if (plugin->destroy) {
            plugin->destroy(&env);
        }
        dlclose(handle);
        return 1;
    }
    if (plugin->destroy) {
        plugin->destroy(&env);
    }
    printf("plugin=%s episodes=%d return_mean=%.3f\n", plugin->name, episodes, mean_return);
    dlclose(handle);
    return 0;
}
