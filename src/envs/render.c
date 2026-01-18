#include "envs_internal.h"

#include <string.h>

#include "core/render_snapshot.h"

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t agent_x;
    uint32_t agent_y;
    uint32_t goal_x;
    uint32_t goal_y;
    uint32_t step;
    uint32_t max_steps;
} tfrl_grid_snapshot;

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
} tfrl_grid_snapshot_v2;

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

size_t tfrl_env_render_bytes_needed(const tfrl_env *env) {
    if (!env) return 0;
    if (env->kind == TFRL_ENV_PYBRIDGE) return 0;
    if (env->kind == TFRL_ENV_SNAKE || env->kind == TFRL_ENV_FLOPPY ||
        env->kind == TFRL_ENV_PANG) {
        return 0;
    }
    int w = MAZE_W;
    int h = MAZE_H;
    if (env->kind == TFRL_ENV_LINEWORLD || env->kind == TFRL_ENV_LINEWORLD_CONT) {
        w = LINEWORLD_W;
        h = LINEWORLD_H;
    } else if (env->kind == TFRL_ENV_POINT1D) {
        w = POINT1D_W;
        h = POINT1D_H;
    } else if (env->kind == TFRL_ENV_COIN_MAZE) {
        w = COIN_MAZE_W;
        h = COIN_MAZE_H;
    } else if (env->kind == TFRL_ENV_TETRIS) {
        w = TETRIS_W;
        h = TETRIS_H;
    }
    int entity_count = 0;
    if (env->kind == TFRL_ENV_COIN_MAZE || env->kind == TFRL_ENV_COIN_MAZE_DUO) {
        for (int i = 0; i < COIN_MAZE_COINS; i++) {
            if (!env->coins_collected[i]) entity_count++;
        }
    }
    if (env->kind == TFRL_ENV_LINEWORLD_DUO) {
        entity_count += 1;
    }
    if (env->kind == TFRL_ENV_COIN_MAZE_DUO) {
        entity_count += 1;
    }
    if (env->kind == TFRL_ENV_TETRIS) {
        entity_count += 4;
    }
    size_t payload_bytes = sizeof(tfrl_grid_snapshot_v3) + (w * h) + (entity_count * sizeof(tfrl_entity_snapshot));
    return sizeof(tfrl_render_snapshot_header) + payload_bytes;
}

static const int TETRIS_BLOCKS[7][4][2] = {
    {{0, 1}, {1, 1}, {2, 1}, {3, 1}},
    {{1, 0}, {2, 0}, {1, 1}, {2, 1}},
    {{1, 0}, {0, 1}, {1, 1}, {2, 1}},
    {{0, 0}, {0, 1}, {1, 1}, {2, 1}},
    {{2, 0}, {0, 1}, {1, 1}, {2, 1}},
    {{1, 0}, {2, 0}, {0, 1}, {1, 1}},
    {{0, 0}, {1, 0}, {1, 1}, {2, 1}}
};

static void tetris_rot_coord(int x, int y, int rot, int *out_x, int *out_y) {
    int rx = x;
    int ry = y;
    for (int i = 0; i < rot; i++) {
        int tmp = rx;
        rx = ry;
        ry = 3 - tmp;
    }
    *out_x = rx;
    *out_y = ry;
}

size_t tfrl_env_render_write(tfrl_env *env, void *buffer, size_t buffer_len) {
    if (!env || !buffer) return 0;
    if (env->kind == TFRL_ENV_PYBRIDGE) return 0;
    if (env->kind == TFRL_ENV_SNAKE || env->kind == TFRL_ENV_FLOPPY ||
        env->kind == TFRL_ENV_PANG) {
        return 0;
    }
    int w = MAZE_W;
    int h = MAZE_H;
    if (env->kind == TFRL_ENV_LINEWORLD || env->kind == TFRL_ENV_LINEWORLD_CONT) {
        w = LINEWORLD_W;
        h = LINEWORLD_H;
    } else if (env->kind == TFRL_ENV_POINT1D) {
        w = POINT1D_W;
        h = POINT1D_H;
    } else if (env->kind == TFRL_ENV_COIN_MAZE) {
        w = COIN_MAZE_W;
        h = COIN_MAZE_H;
    } else if (env->kind == TFRL_ENV_TETRIS) {
        w = TETRIS_W;
        h = TETRIS_H;
    }
    int entity_count = 0;
    if (env->kind == TFRL_ENV_COIN_MAZE || env->kind == TFRL_ENV_COIN_MAZE_DUO) {
        for (int i = 0; i < COIN_MAZE_COINS; i++) {
            if (!env->coins_collected[i]) entity_count++;
        }
    }
    if (env->kind == TFRL_ENV_LINEWORLD_DUO) {
        entity_count += 1;
    }
    if (env->kind == TFRL_ENV_COIN_MAZE_DUO) {
        entity_count += 1;
    }
    if (env->kind == TFRL_ENV_TETRIS) {
        entity_count += 4;
    }
    size_t payload_bytes = sizeof(tfrl_grid_snapshot_v3) + (w * h) + (entity_count * sizeof(tfrl_entity_snapshot));
    size_t total = sizeof(tfrl_render_snapshot_header) + payload_bytes;
    if (buffer_len < total) return 0;

    tfrl_render_snapshot_header header = {
        .api_version = TFRL_SNAPSHOT_API_VERSION,
        .frame_index = env->frame_index,
        .payload_bytes = (uint32_t)payload_bytes,
    };
    memcpy(buffer, &header, sizeof(header));

    uint32_t agent_x = (uint32_t)env->x;
    uint32_t agent_y = (uint32_t)env->y;
    uint32_t goal_x = (uint32_t)(w - 1);
    uint32_t goal_y = (uint32_t)(h - 1);
    uint32_t max_steps = (uint32_t)MAZE_MAX_STEPS;
    if (env->kind == TFRL_ENV_LINEWORLD || env->kind == TFRL_ENV_LINEWORLD_CONT || env->kind == TFRL_ENV_LINEWORLD_DUO) {
        max_steps = (uint32_t)LINEWORLD_MAX_STEPS;
    } else if (env->kind == TFRL_ENV_POINT1D) {
        max_steps = (uint32_t)POINT1D_MAX_STEPS;
        float norm = (env->pos + 1.0f) * 0.5f;
        int px = (int)(norm * (float)(w - 1));
        if (px < 0) px = 0;
        if (px >= w) px = w - 1;
        agent_x = (uint32_t)px;
        agent_y = 0;
        goal_x = (uint32_t)(w - 1);
        goal_y = 0;
    } else if (env->kind == TFRL_ENV_TETRIS) {
        max_steps = (uint32_t)TETRIS_MAX_STEPS;
        agent_x = (uint32_t)env->tetris.x;
        agent_y = (uint32_t)env->tetris.y;
        goal_x = agent_x;
        goal_y = agent_y;
    }

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
        .env_kind = (uint32_t)env->kind,
        .entity_count = (uint32_t)entity_count,
    };
    unsigned char *payload_dst = (unsigned char *)buffer + sizeof(header);
    memcpy(payload_dst, &payload, sizeof(payload));

    unsigned char *walls = payload_dst + sizeof(payload);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (env->kind == TFRL_ENV_LINEWORLD || env->kind == TFRL_ENV_LINEWORLD_CONT ||
                env->kind == TFRL_ENV_LINEWORLD_DUO || env->kind == TFRL_ENV_POINT1D ||
                env->kind == TFRL_ENV_COIN_MAZE || env->kind == TFRL_ENV_COIN_MAZE_DUO) {
                walls[y * w + x] = 0;
            } else if (env->kind == TFRL_ENV_TETRIS) {
                int cell = env->tetris.grid[x][y];
                if (cell < 0) cell = 0;
                if (cell > 7) cell = 7;
                walls[y * w + x] = (unsigned char)cell;
            } else {
                walls[y * w + x] = (unsigned char)(tfrl_env_is_wall_maze(x, y) ? 1 : 0);
            }
        }
    }
    tfrl_entity_snapshot *entities = (tfrl_entity_snapshot *)(walls + (w * h));
    if (env->kind == TFRL_ENV_COIN_MAZE || env->kind == TFRL_ENV_COIN_MAZE_DUO) {
        int idx = 0;
        for (int i = 0; i < COIN_MAZE_COINS; i++) {
            if (env->coins_collected[i]) continue;
            entities[idx].type = 1;
            entities[idx].x = (float)env->coins_x[i];
            entities[idx].y = (float)env->coins_y[i];
            entities[idx].value = 1.0f;
            idx++;
        }
        if (env->kind == TFRL_ENV_COIN_MAZE_DUO) {
            entities[idx].type = 2;
            entities[idx].x = (float)env->x2;
            entities[idx].y = (float)env->y2;
            entities[idx].value = 1.0f;
        }
    }
    if (env->kind == TFRL_ENV_LINEWORLD_DUO) {
        entities[0].type = 2;
        entities[0].x = (float)env->x2;
        entities[0].y = (float)env->y2;
        entities[0].value = 1.0f;
    }
    if (env->kind == TFRL_ENV_TETRIS) {
        int idx = 0;
        int piece = env->tetris.piece;
        if (piece < 0) piece = 0;
        if (piece > 6) piece = 6;
        for (int i = 0; i < 4; i++) {
            int bx = 0;
            int by = 0;
            tetris_rot_coord(TETRIS_BLOCKS[piece][i][0], TETRIS_BLOCKS[piece][i][1],
                              env->tetris.rot, &bx, &by);
            entities[idx].type = 2;
            entities[idx].x = (float)(env->tetris.x + bx);
            entities[idx].y = (float)(env->tetris.y + by);
            entities[idx].value = (float)(piece + 1);
            idx++;
        }
    }
    env->frame_index++;
    return total;
}
