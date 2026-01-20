#include "envs/envs_internal.h"
#include "envs/render_grid.h"

size_t tfrl_env_render_bytes_snake(const tfrl_env *env) {
    if (!env) return 0;
    int body_count = env->snake.len > 1 ? env->snake.len - 1 : 0;
    int entity_count = body_count + 1;
    return tfrl_render_grid_bytes(SNAKE_W, SNAKE_H, entity_count);
}

size_t tfrl_env_render_write_snake(tfrl_env *env, void *buffer, size_t buffer_len) {
    if (!env || !buffer) return 0;
    unsigned char cells[SNAKE_W * SNAKE_H];
    for (int i = 0; i < SNAKE_W * SNAKE_H; i++) {
        cells[i] = 0;
    }

    int body_count = env->snake.len > 1 ? env->snake.len - 1 : 0;
    int entity_count = body_count + 1;
    tfrl_entity_snapshot entities[SNAKE_MAX_LEN + 1];
    int idx = 0;
    for (int i = 1; i < env->snake.len; i++) {
        entities[idx].type = 2;
        entities[idx].x = (float)env->snake.xs[i];
        entities[idx].y = (float)env->snake.ys[i];
        entities[idx].value = 1.0f;
        idx++;
    }
    entities[idx].type = 1;
    entities[idx].x = (float)env->snake.fruit_x;
    entities[idx].y = (float)env->snake.fruit_y;
    entities[idx].value = 1.0f;

    return tfrl_render_grid_write(env, buffer, buffer_len, SNAKE_W, SNAKE_H,
                                  (uint32_t)env->snake.xs[0], (uint32_t)env->snake.ys[0],
                                  (uint32_t)env->snake.fruit_x, (uint32_t)env->snake.fruit_y,
                                  (uint32_t)SNAKE_MAX_STEPS, (uint32_t)env->kind,
                                  cells, entity_count, entities);
}
