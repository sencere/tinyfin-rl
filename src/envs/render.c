#include "envs/envs_internal.h"

size_t tfrl_env_render_bytes_needed(const tfrl_env *env) {
    if (!env || !env->ops || !env->ops->render_bytes) return 0;
    return env->ops->render_bytes(env);
}

size_t tfrl_env_render_write(tfrl_env *env, void *buffer, size_t buffer_len) {
    if (!env || !buffer || !env->ops || !env->ops->render_write) return 0;
    return env->ops->render_write(env, buffer, buffer_len);
}
