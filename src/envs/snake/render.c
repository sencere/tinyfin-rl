#include "envs/render_grid.h"
#include "envs/envs_internal.h"

size_t tfrl_env_render_bytes_snake(const tfrl_env *env) {
    (void)env;
    return tfrl_render_grid_bytes(SNAKE_W, SNAKE_H, 0);
}

size_t tfrl_env_render_write_snake(tfrl_env *env, void *buffer, size_t buffer_len) {
    unsigned char cells[SNAKE_W * SNAKE_H];
    for (int i = 0; i < SNAKE_W * SNAKE_H; i++) cells[i] = 0;
    for (int i = 0; i < env->snake.len; i++) {
        int x = env->snake.xs[i];
        int y = env->snake.ys[i];
        if (x >= 0 && x < SNAKE_W && y >= 0 && y < SNAKE_H) {
            cells[y * SNAKE_W + x] = 1;
        }
    }
    if (env->snake.fruit_x >= 0 && env->snake.fruit_x < SNAKE_W &&
        env->snake.fruit_y >= 0 && env->snake.fruit_y < SNAKE_H) {
        cells[env->snake.fruit_y * SNAKE_W + env->snake.fruit_x] = 2;
    }
    return tfrl_render_grid_write(env, buffer, buffer_len, SNAKE_W, SNAKE_H,
                                  (uint32_t)env->snake.xs[0], (uint32_t)env->snake.ys[0],
                                  (uint32_t)env->snake.fruit_x, (uint32_t)env->snake.fruit_y,
                                  (uint32_t)SNAKE_MAX_STEPS, (uint32_t)env->kind,
                                  cells, 0, NULL);
}
