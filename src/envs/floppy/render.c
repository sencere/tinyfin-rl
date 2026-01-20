#include "envs/envs_internal.h"
#include "envs/render_grid.h"

#define FLOPPY_RENDER_W 40
#define FLOPPY_RENDER_H 23
#define FLOPPY_TUBE_TOP_H 255.0f
#define FLOPPY_TUBE_GAP 90.0f

static int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int map_x(float x) {
    float scale = (float)FLOPPY_W / (float)FLOPPY_RENDER_W;
    int gx = (int)(x / scale);
    return clamp_int(gx, 0, FLOPPY_RENDER_W - 1);
}

static int map_y(float y) {
    float scale = (float)FLOPPY_H / (float)FLOPPY_RENDER_H;
    int gy = (int)(y / scale);
    return clamp_int(gy, 0, FLOPPY_RENDER_H - 1);
}

size_t tfrl_env_render_bytes_floppy(const tfrl_env *env) {
    if (!env) return 0;
    return tfrl_render_grid_bytes(FLOPPY_RENDER_W, FLOPPY_RENDER_H, 1);
}

size_t tfrl_env_render_write_floppy(tfrl_env *env, void *buffer, size_t buffer_len) {
    if (!env || !buffer) return 0;
    unsigned char cells[FLOPPY_RENDER_W * FLOPPY_RENDER_H];
    for (int i = 0; i < FLOPPY_RENDER_W * FLOPPY_RENDER_H; i++) {
        cells[i] = 0;
    }

    for (int i = 0; i < FLOPPY_MAX_TUBES; i++) {
        float rx = env->floppy.tubes_x[i];
        float ry_top = env->floppy.tubes_offset_y[i];
        float ry_bot = 600.0f + env->floppy.tubes_offset_y[i] - FLOPPY_TUBE_TOP_H;
        float rx2 = rx + (float)FLOPPY_TUBES_WIDTH;
        float top_y2 = ry_top + FLOPPY_TUBE_TOP_H;
        float bot_y2 = ry_bot + FLOPPY_TUBE_TOP_H;

        if (rx2 < 0.0f || rx > (float)FLOPPY_W) continue;

        int gx0 = map_x(rx);
        int gx1 = map_x(rx2);
        int gy0 = map_y(ry_top);
        int gy1 = map_y(top_y2);
        for (int y = gy0; y <= gy1; y++) {
            for (int x = gx0; x <= gx1; x++) {
                cells[y * FLOPPY_RENDER_W + x] = 1;
            }
        }

        int gy2 = map_y(ry_bot);
        int gy3 = map_y(bot_y2);
        for (int y = gy2; y <= gy3; y++) {
            for (int x = gx0; x <= gx1; x++) {
                cells[y * FLOPPY_RENDER_W + x] = 1;
            }
        }
    }

    tfrl_entity_snapshot entities[1];
    entities[0].type = 2;
    entities[0].x = (float)map_x(env->floppy.x);
    entities[0].y = (float)map_y(env->floppy.y);
    entities[0].value = 1.0f;

    uint32_t ax = (uint32_t)entities[0].x;
    uint32_t ay = (uint32_t)entities[0].y;
    return tfrl_render_grid_write(env, buffer, buffer_len, FLOPPY_RENDER_W, FLOPPY_RENDER_H,
                                  ax, ay, ax, ay, (uint32_t)FLOPPY_MAX_STEPS, (uint32_t)env->kind,
                                  cells, 1, entities);
}
