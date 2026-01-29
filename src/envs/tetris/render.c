#include "core/render_snapshot.h"
#include "envs/envs_internal.h"

#include <string.h>

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t step;
    uint32_t max_steps;
    float reward;
    uint32_t done;
    uint32_t piece;
    uint32_t next_piece;
    uint32_t rot;
    uint32_t x;
    uint32_t y;
    double score;
    uint32_t lines;
} tfrl_tetris_snapshot;

size_t tfrl_env_render_bytes_tetris(const tfrl_env *env) {
    (void)env;
    size_t payload_bytes = sizeof(tfrl_tetris_snapshot) + (size_t)(TETRIS_W * TETRIS_H);
    return sizeof(tfrl_render_snapshot_header) + payload_bytes;
}

size_t tfrl_env_render_write_tetris(tfrl_env *env, void *buffer, size_t buffer_len) {
    if (!env || !buffer) return 0;
    size_t payload_bytes = sizeof(tfrl_tetris_snapshot) + (size_t)(TETRIS_W * TETRIS_H);
    size_t total = sizeof(tfrl_render_snapshot_header) + payload_bytes;
    if (buffer_len < total) return 0;

    tfrl_render_snapshot_header header = {
        .api_version = TFRL_SNAPSHOT_API_VERSION,
        .payload_kind = TFRL_SNAPSHOT_KIND_TETRIS,
        .frame_index = env->frame_index,
        .payload_bytes = (uint32_t)payload_bytes,
    };
    memcpy(buffer, &header, sizeof(header));

    tfrl_tetris_snapshot payload = {
        .width = TETRIS_W,
        .height = TETRIS_H,
        .step = (uint32_t)env->steps,
        .max_steps = TETRIS_MAX_STEPS,
        .reward = env->last_reward,
        .done = (uint32_t)env->last_done,
        .piece = (uint32_t)env->tetris.piece,
        .next_piece = (uint32_t)env->tetris.next_piece,
        .rot = (uint32_t)env->tetris.rot,
        .x = (uint32_t)env->tetris.x,
        .y = (uint32_t)env->tetris.y,
        .score = env->tetris.score,
        .lines = (uint32_t)env->tetris.lines,
    };

    unsigned char *payload_dst = (unsigned char *)buffer + sizeof(header);
    memcpy(payload_dst, &payload, sizeof(payload));
    unsigned char *grid_dst = payload_dst + sizeof(payload);
    for (int y = 0; y < TETRIS_H; y++) {
        for (int x = 0; x < TETRIS_W; x++) {
            grid_dst[y * TETRIS_W + x] = (unsigned char)(env->tetris.grid[x][y] ? 1 : 0);
        }
    }

    env->frame_index++;
    return total;
}
