#include "envs/render_grid.h"
#include "envs/envs_internal.h"

size_t tfrl_env_render_bytes_point1d(const tfrl_env *env) {
    (void)env;
    return tfrl_render_grid_bytes(POINT1D_W, POINT1D_H, 0);
}

size_t tfrl_env_render_write_point1d(tfrl_env *env, void *buffer, size_t buffer_len) {
    unsigned char cells[POINT1D_W * POINT1D_H];
    for (int i = 0; i < POINT1D_W; i++) cells[i] = 0;
    int agent_x = (int)((env->pos + 1.0f) * 0.5f * (POINT1D_W - 1));
    int goal_x = POINT1D_W - 1;
    return tfrl_render_grid_write(env, buffer, buffer_len, POINT1D_W, POINT1D_H,
                                  (uint32_t)agent_x, 0, (uint32_t)goal_x, 0,
                                  (uint32_t)POINT1D_MAX_STEPS, (uint32_t)env->kind,
                                  cells, 0, NULL);
}
