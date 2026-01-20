#include "envs/envs_internal.h"
#include "envs/render_grid.h"

size_t tfrl_env_render_bytes_maze(const tfrl_env *env) {
    if (!env) return 0;
    return tfrl_render_grid_bytes(MAZE_W, MAZE_H, 0);
}

size_t tfrl_env_render_write_maze(tfrl_env *env, void *buffer, size_t buffer_len) {
    if (!env || !buffer) return 0;
    unsigned char cells[MAZE_W * MAZE_H];
    for (int y = 0; y < MAZE_H; y++) {
        for (int x = 0; x < MAZE_W; x++) {
            cells[y * MAZE_W + x] = (unsigned char)(tfrl_env_is_wall_maze(x, y) ? 1 : 0);
        }
    }
    return tfrl_render_grid_write(env, buffer, buffer_len, MAZE_W, MAZE_H,
                                  (uint32_t)env->x, (uint32_t)env->y,
                                  (uint32_t)(MAZE_W - 1), (uint32_t)(MAZE_H - 1),
                                  (uint32_t)MAZE_MAX_STEPS, (uint32_t)env->kind,
                                  cells, 0, NULL);
}
