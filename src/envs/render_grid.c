#include "envs/render_grid.h"

#include <string.h>

#include "envs/envs_internal.h"

size_t tfrl_render_grid_bytes(int w, int h, int entity_count) {
    if (w <= 0 || h <= 0 || entity_count < 0) return 0;
    size_t payload_bytes = sizeof(tfrl_grid_snapshot_v3) + (size_t)(w * h) +
                           (size_t)entity_count * sizeof(tfrl_entity_snapshot);
    return sizeof(tfrl_render_snapshot_header) + payload_bytes;
}

size_t tfrl_render_grid_write(tfrl_env *env, void *buffer, size_t buffer_len,
                              int w, int h, uint32_t agent_x, uint32_t agent_y,
                              uint32_t goal_x, uint32_t goal_y, uint32_t max_steps,
                              uint32_t env_kind, const unsigned char *cells,
                              int entity_count, const tfrl_entity_snapshot *entities) {
    if (!env || !buffer || !cells || w <= 0 || h <= 0 || entity_count < 0) return 0;
    size_t payload_bytes = sizeof(tfrl_grid_snapshot_v3) + (size_t)(w * h) +
                           (size_t)entity_count * sizeof(tfrl_entity_snapshot);
    size_t total = sizeof(tfrl_render_snapshot_header) + payload_bytes;
    if (buffer_len < total) return 0;

    tfrl_render_snapshot_header header = {
        .api_version = TFRL_SNAPSHOT_API_VERSION,
        .payload_kind = TFRL_SNAPSHOT_KIND_GRID,
        .frame_index = env->frame_index,
        .payload_bytes = (uint32_t)payload_bytes,
    };
    memcpy(buffer, &header, sizeof(header));

    tfrl_grid_snapshot_v3 payload = {
        .width = (uint32_t)w,
        .height = (uint32_t)h,
        .agent_x = agent_x,
        .agent_y = agent_y,
        .goal_x = goal_x,
        .goal_y = goal_y,
        .step = (uint32_t)env->steps,
        .max_steps = max_steps,
        .reward = env->last_reward,
        .done = (uint32_t)env->last_done,
        .env_kind = env_kind,
        .entity_count = (uint32_t)entity_count,
    };
    unsigned char *payload_dst = (unsigned char *)buffer + sizeof(header);
    memcpy(payload_dst, &payload, sizeof(payload));

    unsigned char *cells_dst = payload_dst + sizeof(payload);
    memcpy(cells_dst, cells, (size_t)(w * h));

    if (entity_count > 0 && entities) {
        tfrl_entity_snapshot *entities_dst = (tfrl_entity_snapshot *)(cells_dst + (w * h));
        memcpy(entities_dst, entities, (size_t)entity_count * sizeof(tfrl_entity_snapshot));
    }

    env->frame_index++;
    return total;
}
