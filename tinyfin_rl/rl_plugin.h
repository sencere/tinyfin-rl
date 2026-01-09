#ifndef TINYFIN_RL_PLUGIN_H
#define TINYFIN_RL_PLUGIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "rl.h"

#define TFRL_ENV_PLUGIN_API_VERSION 0x0001

typedef struct {
    uint32_t api_version;
    const char *name;
    tfrl_env (*create)(void);
    void (*destroy)(tfrl_env *env);
} tfrl_env_plugin;

typedef const tfrl_env_plugin *(*tfrl_env_plugin_get_fn)(void);

#ifdef __cplusplus
}
#endif

#endif
