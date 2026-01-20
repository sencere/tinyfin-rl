#include "envs/envs_internal.h"

#include <string.h>

#include "core/render_snapshot.h"

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
    if (!env) return 0;
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

    int piece = env->tetris.piece;
    if (piece < 0) piece = 0;
    if (piece > 6) piece = 6;
    int next_piece = env->tetris.next_piece;
    if (next_piece < 0) next_piece = 0;
    if (next_piece > 6) next_piece = 6;

    tfrl_tetris_snapshot payload = {
        .width = (uint32_t)TETRIS_W,
        .height = (uint32_t)TETRIS_H,
        .step = (uint32_t)env->steps,
        .max_steps = (uint32_t)TETRIS_MAX_STEPS,
        .reward = env->last_reward,
        .done = (uint32_t)env->last_done,
        .piece = (uint32_t)piece,
        .next_piece = (uint32_t)next_piece,
        .rot = (uint32_t)env->tetris.rot,
        .x = (uint32_t)env->tetris.x,
        .y = (uint32_t)env->tetris.y,
        .score = env->tetris.score,
        .lines = (uint32_t)env->tetris.lines,
    };
    unsigned char *payload_dst = (unsigned char *)buffer + sizeof(header);
    memcpy(payload_dst, &payload, sizeof(payload));

    unsigned char *cells = payload_dst + sizeof(payload);
    for (int y = 0; y < TETRIS_H; y++) {
        for (int x = 0; x < TETRIS_W; x++) {
            int cell = env->tetris.grid[x][y];
            if (cell < 0) cell = 0;
            if (cell > 7) cell = 7;
            cells[y * TETRIS_W + x] = (unsigned char)cell;
        }
    }

    env->frame_index++;
    return total;
}
