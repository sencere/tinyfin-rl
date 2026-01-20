#include "envs/envs_internal.h"
#include "envs/render_grid.h"

size_t tfrl_env_render_bytes_coin_maze(const tfrl_env *env) {
    if (!env) return 0;
    int entity_count = 0;
    for (int i = 0; i < COIN_MAZE_COINS; i++) {
        if (!env->coins_collected[i]) entity_count++;
    }
    if (env->kind == TFRL_ENV_COIN_MAZE_DUO) {
        entity_count += 1;
    }
    return tfrl_render_grid_bytes(COIN_MAZE_W, COIN_MAZE_H, entity_count);
}

size_t tfrl_env_render_write_coin_maze(tfrl_env *env, void *buffer, size_t buffer_len) {
    if (!env || !buffer) return 0;
    unsigned char cells[COIN_MAZE_W * COIN_MAZE_H];
    for (int i = 0; i < COIN_MAZE_W * COIN_MAZE_H; i++) {
        cells[i] = 0;
    }
    tfrl_entity_snapshot entities[COIN_MAZE_COINS + 1];
    int entity_count = 0;
    for (int i = 0; i < COIN_MAZE_COINS; i++) {
        if (env->coins_collected[i]) continue;
        entities[entity_count].type = 1;
        entities[entity_count].x = (float)env->coins_x[i];
        entities[entity_count].y = (float)env->coins_y[i];
        entities[entity_count].value = 1.0f;
        entity_count++;
    }
    if (env->kind == TFRL_ENV_COIN_MAZE_DUO) {
        entities[entity_count].type = 2;
        entities[entity_count].x = (float)env->x2;
        entities[entity_count].y = (float)env->y2;
        entities[entity_count].value = 1.0f;
        entity_count++;
    }
    return tfrl_render_grid_write(env, buffer, buffer_len, COIN_MAZE_W, COIN_MAZE_H,
                                  (uint32_t)env->x, (uint32_t)env->y,
                                  (uint32_t)(COIN_MAZE_W - 1), (uint32_t)(COIN_MAZE_H - 1),
                                  (uint32_t)COIN_MAZE_MAX_STEPS, (uint32_t)env->kind,
                                  cells, entity_count, entity_count ? entities : NULL);
}
