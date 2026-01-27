#include "envs/envs_internal.h"

#include <string.h>

#include "core/render_snapshot.h"

#define BREAKOUT_MAX_STEPS 6000
#define BREAKOUT_LINES 4
#define BREAKOUT_BRICKS_PER_LINE 8
#define BREAKOUT_BRICK_OFFSET_Y 40.0f

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t step;
    uint32_t max_steps;
    float reward;
    uint32_t done;
    float player_x;
    float player_y;
    float player_w;
    float player_h;
    uint32_t life;
    float score;
    float reward_vec[5];
    float ball_x;
    float ball_y;
    float ball_vx;
    float ball_vy;
    float ball_radius;
    float brick_w;
    float brick_h;
    float brick_offset_y;
    uint32_t lines;
    uint32_t bricks_per_line;
} tfrl_arkanoid_snapshot;

size_t tfrl_env_render_bytes_breakout(const tfrl_env *env) {
    if (!env) return 0;
    size_t bricks_bytes = (size_t)(BREAKOUT_LINES * BREAKOUT_BRICKS_PER_LINE);
    size_t payload_bytes = sizeof(tfrl_arkanoid_snapshot) + bricks_bytes;
    return sizeof(tfrl_render_snapshot_header) + payload_bytes;
}

size_t tfrl_env_render_write_breakout(tfrl_env *env, void *buffer, size_t buffer_len) {
    if (!env || !buffer) return 0;
    size_t bricks_bytes = (size_t)(BREAKOUT_LINES * BREAKOUT_BRICKS_PER_LINE);
    size_t payload_bytes = sizeof(tfrl_arkanoid_snapshot) + bricks_bytes;
    size_t total = sizeof(tfrl_render_snapshot_header) + payload_bytes;
    if (buffer_len < total) return 0;

    tfrl_render_snapshot_header header = {
        .api_version = TFRL_SNAPSHOT_API_VERSION,
        .payload_kind = TFRL_SNAPSHOT_KIND_ARKANOID,
        .frame_index = env->frame_index,
        .payload_bytes = (uint32_t)payload_bytes,
    };
    memcpy(buffer, &header, sizeof(header));

    tfrl_arkanoid_snapshot payload = {
        .width = (uint32_t)ARKANOID_W,
        .height = (uint32_t)ARKANOID_H,
        .step = (uint32_t)env->steps,
        .max_steps = (uint32_t)BREAKOUT_MAX_STEPS,
        .reward = env->last_reward,
        .done = (uint32_t)env->last_done,
        .player_x = env->arkanoid.player_x,
        .player_y = env->arkanoid.player_y,
        .player_w = env->arkanoid.player_w,
        .player_h = env->arkanoid.player_h,
        .life = (uint32_t)env->arkanoid.life,
        .score = env->arkanoid.score,
        .reward_vec = {
            env->arkanoid.reward_vec[0],
            env->arkanoid.reward_vec[1],
            env->arkanoid.reward_vec[2],
            env->arkanoid.reward_vec[3],
            env->arkanoid.reward_vec[4],
        },
        .ball_x = env->arkanoid.ball_x,
        .ball_y = env->arkanoid.ball_y,
        .ball_vx = env->arkanoid.ball_vx,
        .ball_vy = env->arkanoid.ball_vy,
        .ball_radius = env->arkanoid.ball_radius,
        .brick_w = env->arkanoid.brick_w,
        .brick_h = env->arkanoid.brick_h,
        .brick_offset_y = BREAKOUT_BRICK_OFFSET_Y,
        .lines = (uint32_t)BREAKOUT_LINES,
        .bricks_per_line = (uint32_t)BREAKOUT_BRICKS_PER_LINE,
    };
    unsigned char *payload_dst = (unsigned char *)buffer + sizeof(header);
    memcpy(payload_dst, &payload, sizeof(payload));

    unsigned char *bricks_dst = payload_dst + sizeof(payload);
    for (int i = 0; i < BREAKOUT_LINES; i++) {
        for (int j = 0; j < BREAKOUT_BRICKS_PER_LINE; j++) {
            bricks_dst[i * BREAKOUT_BRICKS_PER_LINE + j] =
                (unsigned char)(env->arkanoid.bricks[i][j] ? 1 : 0);
        }
    }

    env->frame_index++;
    return total;
}

