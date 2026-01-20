#include "envs/envs_internal.h"
#include "envs/render_grid.h"

size_t tfrl_env_render_bytes_point1d(const tfrl_env *env) {
    if (!env) return 0;
    return tfrl_render_grid_bytes(POINT1D_W, POINT1D_H, 0);
}

size_t tfrl_env_render_write_point1d(tfrl_env *env, void *buffer, size_t buffer_len) {
    if (!env || !buffer) return 0;
    unsigned char cells[POINT1D_W * POINT1D_H];
    for (int i = 0; i < POINT1D_W * POINT1D_H; i++) {
        cells[i] = 0;
    }
    float norm = (env->pos + 1.0f) * 0.5f;
    int px = (int)(norm * (float)(POINT1D_W - 1));
    if (px < 0) px = 0;
    if (px >= POINT1D_W) px = POINT1D_W - 1;
    return tfrl_render_grid_write(env, buffer, buffer_len, POINT1D_W, POINT1D_H,
                                  (uint32_t)px, 0,
                                  (uint32_t)(POINT1D_W - 1), 0,
                                  (uint32_t)POINT1D_MAX_STEPS, (uint32_t)env->kind,
                                  cells, 0, NULL);
}
