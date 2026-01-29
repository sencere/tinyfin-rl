#include "envs/render_grid.h"
#include "envs/envs_internal.h"

static int coin_maze_is_wall_render(int x, int y) {
    if (x < 0 || y < 0 || x >= COIN_MAZE_W || y >= COIN_MAZE_H) return 1;
    if (x == 0 || y == 0 || x == COIN_MAZE_W - 1 || y == COIN_MAZE_H - 1) return 1;
    return ((x + y) % 5) == 2;
}

size_t tfrl_env_render_bytes_coin_maze(const tfrl_env *env) {
    (void)env;
    return tfrl_render_grid_bytes(COIN_MAZE_W, COIN_MAZE_H, 0);
}

size_t tfrl_env_render_write_coin_maze(tfrl_env *env, void *buffer, size_t buffer_len) {
    unsigned char cells[COIN_MAZE_W * COIN_MAZE_H];
    for (int y = 0; y < COIN_MAZE_H; y++) {
        for (int x = 0; x < COIN_MAZE_W; x++) {
            unsigned char v = coin_maze_is_wall_render(x, y) ? 1 : 0;
            for (int i = 0; i < COIN_MAZE_COINS; i++) {
                if (!env->coins_collected[i] &&
                    env->coins_x[i] == x && env->coins_y[i] == y) {
                    v = 2;
                    break;
                }
            }
            cells[y * COIN_MAZE_W + x] = v;
        }
    }
    uint32_t goal_x = (uint32_t)(COIN_MAZE_W - 2);
    uint32_t goal_y = (uint32_t)(COIN_MAZE_H - 2);
    for (int i = 0; i < COIN_MAZE_COINS; i++) {
        if (!env->coins_collected[i]) {
            goal_x = (uint32_t)env->coins_x[i];
            goal_y = (uint32_t)env->coins_y[i];
            break;
        }
    }
    return tfrl_render_grid_write(env, buffer, buffer_len, COIN_MAZE_W, COIN_MAZE_H,
                                  (uint32_t)env->x, (uint32_t)env->y,
                                  goal_x, goal_y,
                                  (uint32_t)COIN_MAZE_MAX_STEPS, (uint32_t)env->kind,
                                  cells, 0, NULL);
}
