#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyfin_rl/rl.h"
#include "tinyfin_rl/rl_plugin.h"

typedef const tfrl_env_plugin *(*plugin_get_fn)(void);

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <env_plugin.so>\n", argv[0]);
        return 1;
    }
    void *handle = dlopen(argv[1], RTLD_NOW);
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

    tfrl_value obs = {0};
    tfrl_step step = {0};
    tfrl_env_reset(&env, &obs);
    for (int i = 0; i < 20; i++) {
        tfrl_value act = {.type = TFRL_VALUE_I64, .as.i64 = 0};
        tfrl_env_step(&env, &act, &step);
        if (tfrl_step_done(&step)) {
            tfrl_env_reset(&env, &obs);
        }
    }
    if (plugin->destroy) {
        plugin->destroy(&env);
    } else {
        tfrl_env_destroy(&env);
    }
    dlclose(handle);
    return 0;
}
