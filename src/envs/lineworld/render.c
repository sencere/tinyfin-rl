#include "envs/envs_internal.h"
#include "envs/render_grid.h"

size_t tfrl_env_render_bytes_lineworld(const tfrl_env *env) {
    if (!env) return 0;
    int entity_count = (env->kind == TFRL_ENV_LINEWORLD_DUO) ? 1 : 0;
    return tfrl_render_grid_bytes(LINEWORLD_W, LINEWORLD_H, entity_count);
}

size_t tfrl_env_render_write_lineworld(tfrl_env *env, void *buffer, size_t buffer_len) {
    if (!env || !buffer) return 0;
    unsigned char cells[LINEWORLD_W * LINEWORLD_H];
    for (int i = 0; i < LINEWORLD_W * LINEWORLD_H; i++) {
        cells[i] = 0;
    }
    tfrl_entity_snapshot entities[1];
    int entity_count = 0;
    if (env->kind == TFRL_ENV_LINEWORLD_DUO) {
        entities[0].type = 2;
        entities[0].x = (float)env->x2;
        entities[0].y = (float)env->y2;
        entities[0].value = 1.0f;
        entity_count = 1;
    }
    return tfrl_render_grid_write(env, buffer, buffer_len, LINEWORLD_W, LINEWORLD_H,
                                  (uint32_t)env->x, (uint32_t)env->y,
                                  (uint32_t)(LINEWORLD_W - 1), (uint32_t)(LINEWORLD_H - 1),
                                  (uint32_t)LINEWORLD_MAX_STEPS, (uint32_t)env->kind,
                                  cells, entity_count, entity_count ? entities : NULL);
}
