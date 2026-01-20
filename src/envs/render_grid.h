#ifndef TFRL_RENDER_GRID_H
#define TFRL_RENDER_GRID_H

#include <stddef.h>
#include <stdint.h>

#include "core/render_snapshot.h"

typedef struct tfrl_env tfrl_env;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t agent_x;
    uint32_t agent_y;
    uint32_t goal_x;
    uint32_t goal_y;
    uint32_t step;
    uint32_t max_steps;
    float reward;
    uint32_t done;
    uint32_t env_kind;
    uint32_t entity_count;
} tfrl_grid_snapshot_v3;

typedef struct {
    uint32_t type;
    float x;
    float y;
    float value;
} tfrl_entity_snapshot;

size_t tfrl_render_grid_bytes(int w, int h, int entity_count);
size_t tfrl_render_grid_write(tfrl_env *env, void *buffer, size_t buffer_len,
                              int w, int h, uint32_t agent_x, uint32_t agent_y,
                              uint32_t goal_x, uint32_t goal_y, uint32_t max_steps,
                              uint32_t env_kind, const unsigned char *cells,
                              int entity_count, const tfrl_entity_snapshot *entities);

#endif
