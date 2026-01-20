#include "envs/envs_internal.h"
#include "envs/render_grid.h"

#define PANG_RENDER_W 40
#define PANG_RENDER_H 23

static int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int map_x(float x) {
    float scale = (float)PANG_W / (float)PANG_RENDER_W;
    int gx = (int)(x / scale);
    return clamp_int(gx, 0, PANG_RENDER_W - 1);
}

static int map_y(float y) {
    float scale = (float)PANG_H / (float)PANG_RENDER_H;
    int gy = (int)(y / scale);
    return clamp_int(gy, 0, PANG_RENDER_H - 1);
}

size_t tfrl_env_render_bytes_pang(const tfrl_env *env) {
    if (!env) return 0;
    int entity_count = 1 + PANG_MAX_BIG + PANG_MAX_MED + PANG_MAX_SMALL;
    if (env->pang.shot_active) entity_count += 1;
    return tfrl_render_grid_bytes(PANG_RENDER_W, PANG_RENDER_H, entity_count);
}

size_t tfrl_env_render_write_pang(tfrl_env *env, void *buffer, size_t buffer_len) {
    if (!env || !buffer) return 0;
    unsigned char cells[PANG_RENDER_W * PANG_RENDER_H];
    for (int i = 0; i < PANG_RENDER_W * PANG_RENDER_H; i++) {
        cells[i] = 0;
    }

    tfrl_entity_snapshot entities[1 + PANG_MAX_BIG + PANG_MAX_MED + PANG_MAX_SMALL + 1];
    int idx = 0;
    entities[idx].type = 2;
    entities[idx].x = (float)map_x(env->pang.player_x);
    entities[idx].y = (float)map_y(env->pang.player_y - env->pang.ship_height * 0.5f);
    entities[idx].value = 1.0f;
    idx++;

    for (int i = 0; i < PANG_MAX_BIG; i++) {
        if (!env->pang.big[i].active) continue;
        entities[idx].type = 1;
        entities[idx].x = (float)map_x(env->pang.big[i].x);
        entities[idx].y = (float)map_y(env->pang.big[i].y);
        entities[idx].value = 1.0f;
        idx++;
    }
    for (int i = 0; i < PANG_MAX_MED; i++) {
        if (!env->pang.med[i].active) continue;
        entities[idx].type = 1;
        entities[idx].x = (float)map_x(env->pang.med[i].x);
        entities[idx].y = (float)map_y(env->pang.med[i].y);
        entities[idx].value = 1.0f;
        idx++;
    }
    for (int i = 0; i < PANG_MAX_SMALL; i++) {
        if (!env->pang.small[i].active) continue;
        entities[idx].type = 1;
        entities[idx].x = (float)map_x(env->pang.small[i].x);
        entities[idx].y = (float)map_y(env->pang.small[i].y);
        entities[idx].value = 1.0f;
        idx++;
    }

    if (env->pang.shot_active) {
        entities[idx].type = 2;
        entities[idx].x = (float)map_x(env->pang.shot_x);
        entities[idx].y = (float)map_y(env->pang.shot_y);
        entities[idx].value = 1.0f;
        idx++;
    }

    uint32_t ax = (uint32_t)entities[0].x;
    uint32_t ay = (uint32_t)entities[0].y;
    return tfrl_render_grid_write(env, buffer, buffer_len, PANG_RENDER_W, PANG_RENDER_H,
                                  ax, ay, ax, ay, (uint32_t)PANG_MAX_STEPS, (uint32_t)env->kind,
                                  cells, idx, entities);
}
