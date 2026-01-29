#include "envs/render_grid.h"
#include "envs/envs_internal.h"

size_t tfrl_env_render_bytes_lineworld(const tfrl_env *env) {
    (void)env;
    return tfrl_render_grid_bytes(LINEWORLD_W, LINEWORLD_H, 0);
}

size_t tfrl_env_render_write_lineworld(tfrl_env *env, void *buffer, size_t buffer_len) {
    unsigned char cells[LINEWORLD_W * LINEWORLD_H];
    for (int i = 0; i < LINEWORLD_W * LINEWORLD_H; i++) cells[i] = 0;
    return tfrl_render_grid_write(env, buffer, buffer_len, LINEWORLD_W, LINEWORLD_H,
                                  (uint32_t)env->x, 0,
                                  (uint32_t)(LINEWORLD_W - 1), 0,
                                  (uint32_t)LINEWORLD_MAX_STEPS, (uint32_t)env->kind,
                                  cells, 0, NULL);
}
