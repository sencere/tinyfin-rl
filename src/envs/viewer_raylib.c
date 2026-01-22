#include "viewer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#include "core/render_snapshot.h"
#include "envs_internal.h"

#define CELL_SIZE 48

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

struct tfrl_viewer {
    int init;
    int fps;
    const char *title;
    char *meta;
    int width;
    int height;
};

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

static Color tetris_color(uint32_t id) {
    switch (id) {
        case 1: return (Color){40, 90, 170, 255};
        case 2: return (Color){55, 110, 195, 255};
        case 3: return (Color){70, 130, 220, 255};
        case 4: return (Color){90, 150, 235, 255};
        case 5: return (Color){110, 170, 245, 255};
        case 6: return (Color){130, 190, 255, 255};
        case 7: return (Color){30, 70, 140, 255};
        default: return DARKGRAY;
    }
}

static Color arkanoid_brick_color(uint32_t row, uint32_t col) {
    static const Color palette[] = {
        {10, 35, 95, 255},
        {15, 45, 110, 255},
        {20, 55, 125, 255},
        {25, 65, 140, 255},
        {30, 75, 155, 255}
    };
    uint32_t idx = (row * 131u + col * 197u + 17u) % (uint32_t)(sizeof(palette) / sizeof(palette[0]));
    return palette[idx];
}

static char *tfrl_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *out = (char *)malloc(len);
    if (!out) return NULL;
    memcpy(out, s, len);
    return out;
}

static int ensure_init(tfrl_viewer *viewer, int width, int height) {
    if (viewer->init) return 1;
    int w = width * CELL_SIZE;
    int h = height * CELL_SIZE + 60;
    InitWindow(w, h, viewer->title ? viewer->title : "tinyfin-rl");
    SetTargetFPS(viewer->fps > 0 ? viewer->fps : 60);
    viewer->width = width;
    viewer->height = height;
    viewer->init = 1;
    return 1;
}

static int ensure_init_pixels(tfrl_viewer *viewer, int width, int height) {
    if (viewer->init) return 1;
    InitWindow(width, height, viewer->title ? viewer->title : "tinyfin-rl");
    SetTargetFPS(viewer->fps > 0 ? viewer->fps : 60);
    viewer->width = width;
    viewer->height = height;
    viewer->init = 1;
    return 1;
}

tfrl_viewer *tfrl_viewer_create(const tfrl_viewer_config *cfg) {
    tfrl_viewer *viewer = (tfrl_viewer *)calloc(1, sizeof(tfrl_viewer));
    if (!viewer) return NULL;
    viewer->fps = cfg ? cfg->fps : 60;
    viewer->title = cfg ? cfg->title : NULL;
    if (cfg && cfg->meta) {
        viewer->meta = tfrl_strdup(cfg->meta);
    }
    return viewer;
}

void tfrl_viewer_draw(tfrl_viewer *viewer, const void *snapshot, size_t len) {
    if (!viewer || !snapshot || len < sizeof(tfrl_render_snapshot_header)) return;
    const unsigned char *bytes = (const unsigned char *)snapshot;
    const tfrl_render_snapshot_header *header = (const tfrl_render_snapshot_header *)bytes;
    const unsigned char *payload = bytes + sizeof(tfrl_render_snapshot_header);
    if (header->api_version == 0x0004 && header->payload_kind == TFRL_SNAPSHOT_KIND_TETRIS) {
        if (header->payload_bytes < sizeof(tfrl_tetris_snapshot)) return;
        const tfrl_tetris_snapshot *tetris = (const tfrl_tetris_snapshot *)payload;
        size_t grid_bytes = header->payload_bytes - sizeof(tfrl_tetris_snapshot);
        if (grid_bytes < (size_t)(tetris->width * tetris->height)) return;
        const unsigned char *cells = payload + sizeof(tfrl_tetris_snapshot);
        if (!ensure_init(viewer, (int)tetris->width + 6, (int)tetris->height)) return;
        if (WindowShouldClose()) return;

        BeginDrawing();
        ClearBackground(RAYWHITE);
        int board_w = (int)tetris->width;
        int board_h = (int)tetris->height;
        for (int y = 0; y < board_h; y++) {
            for (int x = 0; x < board_w; x++) {
                int px = x * CELL_SIZE;
                int py = y * CELL_SIZE;
                unsigned char cell = cells[y * board_w + x];
                if (cell) {
                    DrawRectangle(px, py, CELL_SIZE, CELL_SIZE, tetris_color(cell));
                } else {
                    DrawRectangleLines(px, py, CELL_SIZE, CELL_SIZE, LIGHTGRAY);
                }
            }
        }

        int piece = (int)tetris->piece;
        if (piece < 0) piece = 0;
        if (piece > 6) piece = 6;
        for (int i = 0; i < 4; i++) {
            int bx = 0;
            int by = 0;
            tetris_rot_coord(TETRIS_BLOCKS[piece][i][0], TETRIS_BLOCKS[piece][i][1],
                             (int)tetris->rot, &bx, &by);
            int gx = (int)tetris->x + bx;
            int gy = (int)tetris->y + by;
            if (gx >= 0 && gx < board_w && gy >= 0 && gy < board_h) {
                DrawRectangle(gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE,
                              tetris_color((uint32_t)(piece + 1)));
            }
        }

        int panel_x = board_w * CELL_SIZE + 10;
        DrawText(TextFormat("score: %.1f", tetris->score), panel_x, 10, 18, DARKGRAY);
        DrawText(TextFormat("lines: %u", tetris->lines), panel_x, 32, 18, DARKGRAY);
        DrawText(TextFormat("step: %u/%u", tetris->step, tetris->max_steps), panel_x, 54, 18, DARKGRAY);
        DrawText(TextFormat("reward: %.3f", tetris->reward), panel_x, 76, 18, DARKGRAY);
        DrawText(TextFormat("done: %u", tetris->done), panel_x, 98, 18, DARKGRAY);

        DrawText("next:", panel_x, 130, 18, DARKGRAY);
        int next_piece = (int)tetris->next_piece;
        if (next_piece < 0) next_piece = 0;
        if (next_piece > 6) next_piece = 6;
        int preview_x = panel_x + 10;
        int preview_y = 160;
        for (int i = 0; i < 4; i++) {
            int bx = TETRIS_BLOCKS[next_piece][i][0];
            int by = TETRIS_BLOCKS[next_piece][i][1];
            DrawRectangle(preview_x + bx * (CELL_SIZE / 2),
                          preview_y + by * (CELL_SIZE / 2),
                          CELL_SIZE / 2, CELL_SIZE / 2,
                          tetris_color((uint32_t)(next_piece + 1)));
        }

        if (viewer->meta) {
            char meta_buf[180];
            snprintf(meta_buf, sizeof(meta_buf), "meta: %.160s", viewer->meta);
            DrawText(meta_buf, panel_x, board_h * CELL_SIZE - 20, 16, DARKGRAY);
        }
        EndDrawing();
        return;
    }
    if (header->api_version == 0x0004 && header->payload_kind == TFRL_SNAPSHOT_KIND_ARKANOID) {
        if (header->payload_bytes < sizeof(tfrl_arkanoid_snapshot)) return;
        const tfrl_arkanoid_snapshot *ark = (const tfrl_arkanoid_snapshot *)payload;
        size_t bricks_bytes = header->payload_bytes - sizeof(tfrl_arkanoid_snapshot);
        size_t expected_bricks = (size_t)(ark->lines * ark->bricks_per_line);
        if (bricks_bytes < expected_bricks) return;
        const unsigned char *bricks = payload + sizeof(tfrl_arkanoid_snapshot);
        if (!ensure_init_pixels(viewer, (int)ark->width, (int)ark->height)) return;
        if (WindowShouldClose()) return;

        BeginDrawing();
        ClearBackground(RAYWHITE);

        Color paddle_color = (Color){70, 150, 255, 255};
        Color ball_color = (Color){40, 110, 210, 255};

        // Draw paddle.
        DrawRectangle((int)(ark->player_x - ark->player_w * 0.5f),
                      (int)(ark->player_y - ark->player_h * 0.5f),
                      (int)ark->player_w, (int)ark->player_h, paddle_color);

        // Draw ball.
        DrawCircle((int)ark->ball_x, (int)ark->ball_y, ark->ball_radius, ball_color);

        // Draw bricks.
        for (uint32_t i = 0; i < ark->lines; i++) {
            for (uint32_t j = 0; j < ark->bricks_per_line; j++) {
                if (!bricks[i * ark->bricks_per_line + j]) continue;
                float bx = (float)j * ark->brick_w + ark->brick_w * 0.5f;
                float by = (float)i * ark->brick_h + ark->brick_offset_y;
                int x = (int)(bx - ark->brick_w * 0.5f);
                int y = (int)(by - ark->brick_h * 0.5f);
                Color color = arkanoid_brick_color(i, j);
                DrawRectangle(x, y, (int)ark->brick_w, (int)ark->brick_h, color);
            }
        }

        // Lives indicators.
        for (uint32_t i = 0; i < ark->life; i++) {
            DrawRectangle(20 + (int)(40 * i), (int)ark->height - 30, 35, 10, LIGHTGRAY);
        }

        DrawText(TextFormat("step: %u/%u", ark->step, ark->max_steps), 20, 10, 18, DARKGRAY);
        DrawText(TextFormat("reward: %.3f", ark->reward), 20, 32, 18, DARKGRAY);
        DrawText(TextFormat("vec: b %.2f l %.2f c %.2f tr %.3f p %.2f",
                            ark->reward_vec[0], ark->reward_vec[1],
                            ark->reward_vec[2], ark->reward_vec[3],
                            ark->reward_vec[4]),
                 20, 54, 18, DARKGRAY);
        DrawText(TextFormat("score: %.0f", ark->score), 20, 76, 18, DARKGRAY);
        DrawText(TextFormat("done: %u", ark->done), 20, 98, 18, DARKGRAY);
        DrawText(TextFormat("lives: %u", ark->life), 20, 120, 18, DARKGRAY);
        if (viewer->meta) {
            char meta_buf[180];
            snprintf(meta_buf, sizeof(meta_buf), "meta: %.160s", viewer->meta);
            DrawText(meta_buf, 20, (int)ark->height - 20, 16, DARKGRAY);
        }

        EndDrawing();
        return;
    }
    if (header->api_version == 0x0004 && header->payload_kind != TFRL_SNAPSHOT_KIND_GRID) return;
    const tfrl_grid_snapshot *grid = NULL;
    tfrl_grid_snapshot_v2 grid_v2 = {0};
    tfrl_grid_snapshot_v3 grid_v3 = {0};
    size_t payload_header_size = 0;
    if (header->api_version == 0x0004 || header->api_version == 0x0003) {
        if (header->payload_bytes < sizeof(tfrl_grid_snapshot_v3)) return;
        const tfrl_grid_snapshot_v3 *grid_ptr = (const tfrl_grid_snapshot_v3 *)payload;
        grid_v3 = *grid_ptr;
        grid = (const tfrl_grid_snapshot *)&grid_v3;
        payload_header_size = sizeof(tfrl_grid_snapshot_v3);
    } else if (header->api_version == 0x0002) {
        if (header->payload_bytes < sizeof(tfrl_grid_snapshot_v2)) return;
        const tfrl_grid_snapshot_v2 *grid_ptr = (const tfrl_grid_snapshot_v2 *)payload;
        grid_v2 = *grid_ptr;
        grid = (const tfrl_grid_snapshot *)&grid_v2;
        payload_header_size = sizeof(tfrl_grid_snapshot_v2);
    } else {
        if (header->payload_bytes < sizeof(tfrl_grid_snapshot)) return;
        grid = (const tfrl_grid_snapshot *)payload;
        payload_header_size = sizeof(tfrl_grid_snapshot);
    }
    size_t walls_bytes = header->payload_bytes - payload_header_size;
    if (walls_bytes < (size_t)(grid->width * grid->height)) return;
    const unsigned char *walls = payload + payload_header_size;
    const tfrl_entity_snapshot *entities = NULL;
    size_t entity_count = 0;
    int is_grid_v3 = (header->api_version == 0x0004 || header->api_version == 0x0003);
    if (is_grid_v3) {
        size_t entities_offset = (size_t)(grid->width * grid->height);
        size_t entities_bytes = walls_bytes - entities_offset;
        entity_count = grid_v3.entity_count;
        if (entities_bytes < entity_count * sizeof(tfrl_entity_snapshot)) return;
        entities = (const tfrl_entity_snapshot *)(walls + entities_offset);
    }
    int is_tetris_grid = is_grid_v3 && grid_v3.env_kind == TFRL_ENV_TETRIS;

    if (!ensure_init(viewer, (int)grid->width, (int)grid->height)) return;
    if (WindowShouldClose()) return;

    BeginDrawing();
    ClearBackground(RAYWHITE);
    for (uint32_t y = 0; y < grid->height; y++) {
        for (uint32_t x = 0; x < grid->width; x++) {
            int px = (int)x * CELL_SIZE;
            int py = (int)y * CELL_SIZE;
            unsigned char cell = walls[y * grid->width + x];
            if (cell) {
                if (is_tetris_grid) {
                    DrawRectangle(px, py, CELL_SIZE, CELL_SIZE, tetris_color(cell));
                } else {
                    DrawRectangle(px, py, CELL_SIZE, CELL_SIZE, DARKGRAY);
                }
            } else {
                DrawRectangleLines(px, py, CELL_SIZE, CELL_SIZE, LIGHTGRAY);
            }
        }
    }
    if (!is_tetris_grid) {
        DrawRectangle((int)grid->goal_x * CELL_SIZE, (int)grid->goal_y * CELL_SIZE, CELL_SIZE, CELL_SIZE, (Color){255, 210, 120, 255});
        DrawRectangle((int)grid->agent_x * CELL_SIZE, (int)grid->agent_y * CELL_SIZE, CELL_SIZE, CELL_SIZE, (Color){70, 130, 255, 255});
    }
    if (entities) {
        for (size_t i = 0; i < entity_count; i++) {
            int cx = (int)entities[i].x * CELL_SIZE + CELL_SIZE / 2;
            int cy = (int)entities[i].y * CELL_SIZE + CELL_SIZE / 2;
            if (is_tetris_grid) {
                uint32_t id = (uint32_t)(entities[i].value + 0.5f);
                DrawRectangle((int)entities[i].x * CELL_SIZE, (int)entities[i].y * CELL_SIZE,
                              CELL_SIZE, CELL_SIZE, tetris_color(id));
            } else if (entities[i].type == 2) {
                DrawRectangle(cx - CELL_SIZE / 4, cy - CELL_SIZE / 4, CELL_SIZE / 2, CELL_SIZE / 2, (Color){220, 80, 80, 255});
            } else {
                DrawCircle(cx, cy, CELL_SIZE * 0.2f, (Color){255, 215, 0, 255});
            }
        }
    }

    int text_y = (int)grid->height * CELL_SIZE + 10;
    DrawText(TextFormat("step: %u / %u", grid->step, grid->max_steps), 10, text_y, 18, DARKGRAY);
    if (is_grid_v3) {
        DrawText(TextFormat("reward: %.3f", grid_v3.reward), 200, text_y, 18, DARKGRAY);
        DrawText(TextFormat("done: %u", grid_v3.done), 360, text_y, 18, DARKGRAY);
        DrawText(TextFormat("entities: %u", grid_v3.entity_count), 470, text_y, 18, DARKGRAY);
    } else if (header->api_version == 0x0002) {
        DrawText(TextFormat("reward: %.3f", grid_v2.reward), 200, text_y, 18, DARKGRAY);
        DrawText(TextFormat("done: %u", grid_v2.done), 360, text_y, 18, DARKGRAY);
    }
    if (viewer->meta) {
        char meta_buf[180];
        snprintf(meta_buf, sizeof(meta_buf), "meta: %.160s", viewer->meta);
        DrawText(meta_buf, 10, text_y + 22, 16, DARKGRAY);
    }
    EndDrawing();
}

int tfrl_viewer_should_close(tfrl_viewer *viewer) {
    if (!viewer || !viewer->init) return 1;
    return WindowShouldClose();
}

void tfrl_viewer_destroy(tfrl_viewer *viewer) {
    if (!viewer) return;
    if (viewer->init) {
        CloseWindow();
    }
    free(viewer->meta);
    free(viewer);
}
