#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyfin_rl/rl.h"
#include "tinyfin_rl/rl_plugin.h"

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

    uint64_t rng = 1234;
    double return_sum = 0.0;
    for (int ep = 0; ep < episodes; ep++) {
        tfrl_env env = plugin->create();
        tfrl_value obs = {0};
        tfrl_step step = {0};
        if (tfrl_env_reset(&env, &obs) != TFRL_OK) {
            fprintf(stderr, "env reset failed\n");
            if (plugin->destroy) {
                plugin->destroy(&env);
            }
            dlclose(handle);
            return 1;
        }
        int done = 0;
        double ep_return = 0.0;
        while (!done) {
            int action = (int)(rng_uniform(&rng) * actions) % actions;
            tfrl_value act = {.type = TFRL_VALUE_I64, .as.i64 = action};
            if (tfrl_env_step(&env, &act, &step) != TFRL_OK) {
                fprintf(stderr, "env step failed\n");
                if (plugin->destroy) {
                    plugin->destroy(&env);
                }
                dlclose(handle);
                return 1;
            }
            done = tfrl_step_done(&step);
            ep_return += step.reward;
        }
        return_sum += ep_return;
        if (plugin->destroy) {
            plugin->destroy(&env);
        }
    }

    printf("plugin=%s episodes=%d return_mean=%.3f\n", plugin->name, episodes, return_sum / episodes);
    dlclose(handle);
    return 0;
}
