#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyfin_rl/rl.h"
#include "tinyfin_rl/rl_env_info.h"
#include "tinyfin_rl/rl_plugin.h"
#include "tinyfin_rl/rl_train.h"

typedef const tfrl_env_plugin *(*plugin_get_fn)(void);
typedef const tfrl_env_metadata *(*meta_get_fn)(void);

static const char *arg_str(int argc, char **argv, const char *name, const char *def) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return def;
}

static int arg_int(int argc, char **argv, const char *name, int def) {
    const char *v = arg_str(argc, argv, name, NULL);
    return v ? atoi(v) : def;
}

static float arg_float(int argc, char **argv, const char *name, float def) {
    const char *v = arg_str(argc, argv, name, NULL);
    return v ? (float)atof(v) : def;
}

int main(int argc, char **argv) {
    const char *env_path = arg_str(argc, argv, "--env", NULL);
    if (!env_path) {
        fprintf(stderr, "usage: %s --env <env_plugin.so> [--obs-n N] [--action-n N] [--save PATH] [--load PATH] [--eval-steps N]\n", argv[0]);
        return 1;
    }
    void *handle = dlopen(env_path, RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    plugin_get_fn plugin_get = (plugin_get_fn)dlsym(handle, "tfrl_env_plugin_get");
    if (!plugin_get) {
        fprintf(stderr, "missing tfrl_env_plugin_get\n");
        dlclose(handle);
        return 1;
    }
    const tfrl_env_plugin *plugin = plugin_get();
    tfrl_env env = plugin->create();

    int obs_n = arg_int(argc, argv, "--obs-n", 0);
    int action_n = arg_int(argc, argv, "--action-n", 0);
    meta_get_fn meta_get = (meta_get_fn)dlsym(handle, "tfrl_env_metadata_get");
    if (meta_get) {
        const tfrl_env_metadata *meta = meta_get();
        if (meta) {
            if (obs_n <= 0) obs_n = (int)meta->observation_n;
            if (action_n <= 0) action_n = (int)meta->action_n;
        }
    }
    if (obs_n <= 0 || action_n <= 0) {
        fprintf(stderr, "obs_n/action_n required\n");
        return 1;
    }

    tfrl_ppo_config cfg = {
        .obs_n = obs_n,
        .action_n = action_n,
        .steps_per_batch = arg_int(argc, argv, "--steps-per-batch", 64),
        .updates = arg_int(argc, argv, "--updates", 20),
        .epochs = arg_int(argc, argv, "--epochs", 2),
        .eval_steps = arg_int(argc, argv, "--eval-steps", 0),
        .gamma = arg_float(argc, argv, "--gamma", 0.99f),
        .lr = arg_float(argc, argv, "--lr", 0.03f),
        .clip_eps = arg_float(argc, argv, "--clip-eps", 0.2f),
        .save_path = arg_str(argc, argv, "--save", NULL),
        .load_path = arg_str(argc, argv, "--load", NULL),
    };
    tfrl_train_metrics metrics = {0};
    int status = tfrl_train_ppo(&env, &cfg, &metrics);
    printf("status=%d return_mean=%.4f loss=%.4f\n", status, metrics.return_mean, metrics.loss);

    if (plugin->destroy) {
        plugin->destroy(&env);
    } else {
        tfrl_env_destroy(&env);
    }
    dlclose(handle);
    return 0;
}
