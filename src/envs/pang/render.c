#include "envs/render_grid.h"
#include "envs/envs_internal.h"

size_t tfrl_env_render_bytes_pang(const tfrl_env *env) {
    (void)env;
    return tfrl_render_grid_bytes(20, 12, 0);
}

size_t tfrl_env_render_write_pang(tfrl_env *env, void *buffer, size_t buffer_len) {
    const int w = 20;
    const int h = 12;
    unsigned char cells[w * h];
    for (int i = 0; i < w * h; i++) cells[i] = 0;
    int ax = (int)((env->pang.player_x / (float)PANG_W) * (w - 1));
    int ay = (int)(((env->pang.player_y - env->pang.ship_height * 0.5f) / (float)PANG_H) * (h - 1));
    return tfrl_render_grid_write(env, buffer, buffer_len, w, h,
                                  (uint32_t)ax, (uint32_t)ay,
                                  (uint32_t)(w - 1), (uint32_t)(h - 1),
                                  (uint32_t)PANG_MAX_STEPS, (uint32_t)env->kind,
                                  cells, 0, NULL);
}
