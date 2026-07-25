#include "viewer.h"

#ifdef USE_RAYLIB
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#include "core/render_snapshot.h"
#include "envs/envs_internal.h"

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

typedef struct {
    uint32_t width, height, step, max_steps;
    float reward;
    uint32_t done, piece, next_piece, rot, x, y;
    double score;
    uint32_t lines;
} tfrl_tetris_snapshot;

struct tfrl_viewer {
    int fps;
    const char *title;
    char *meta;
    int width;
    int height;
};

tfrl_viewer *tfrl_viewer_create(const tfrl_viewer_config *cfg) {
    tfrl_viewer *viewer = (tfrl_viewer *)calloc(1, sizeof(tfrl_viewer));
    if (!viewer) return NULL;
    viewer->fps = cfg && cfg->fps > 0 ? cfg->fps : 60;
    viewer->title = cfg && cfg->title ? cfg->title : "tinyfin-rl";
    if (cfg && cfg->meta) {
        viewer->meta = (char *)calloc(strlen(cfg->meta) + 1, 1);
        if (viewer->meta) strcpy(viewer->meta, cfg->meta);
    }
    viewer->width = 800;
    viewer->height = 450;
    InitWindow(viewer->width, viewer->height, viewer->title);
    SetTargetFPS(viewer->fps);
    return viewer;
}

void tfrl_viewer_destroy(tfrl_viewer *viewer) {
    if (!viewer) return;
    CloseWindow();
    free(viewer->meta);
    free(viewer);
}

int tfrl_viewer_should_close(tfrl_viewer *viewer) {
    (void)viewer;
    return WindowShouldClose();
}

static void draw_grid(const tfrl_grid_snapshot_v3 *grid, const unsigned char *cells) {
    if (!grid || !cells) return;
    int w = (int)grid->width;
    int h = (int)grid->height;
    float cell_w = (float)GetScreenWidth() / (float)w;
    float cell_h = (float)GetScreenHeight() / (float)h;
    if (w <= 0 || h <= 0) return;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            unsigned char v = cells[y * w + x];
            Color c = (v > 0) ? DARKGRAY : RAYWHITE;
            int rx = (int)(x * cell_w);
            int ry = (int)(y * cell_h);
            int rw = (int)(cell_w + 1.0f);
            int rh = (int)(cell_h + 1.0f);
            DrawRectangle(rx, ry, rw, rh, c);
            DrawRectangleLines(rx, ry, rw, rh, LIGHTGRAY);
        }
    }
    if (grid->goal_x < grid->width && grid->goal_y < grid->height) {
        int gx = (int)((float)grid->goal_x * cell_w);
        int gy = (int)((float)grid->goal_y * cell_h);
        DrawRectangle(gx + 4, gy + 4, (int)cell_w - 8, (int)cell_h - 8, GREEN);
    }
    if (grid->agent_x < grid->width && grid->agent_y < grid->height) {
        float cx = ((float)grid->agent_x + 0.5f) * cell_w;
        float cy = ((float)grid->agent_y + 0.5f) * cell_h;
        float radius = cell_w < cell_h ? cell_w * 0.32f : cell_h * 0.32f;
        DrawCircle((int)cx, (int)cy, radius, BLUE);
    }
    DrawText(TextFormat("step %u/%u  reward %.3f", grid->step, grid->max_steps, grid->reward),
             12, 12, 20, BLACK);
}

static void draw_arkanoid(const tfrl_arkanoid_snapshot *ark, const unsigned char *bricks) {
    if (!ark || !bricks) return;
    DrawRectangle(0, 0, (int)ark->width, (int)ark->height, BLACK);
    DrawRectangle((int)(ark->player_x - ark->player_w * 0.5f),
                  (int)(ark->player_y - ark->player_h * 0.5f),
                  (int)ark->player_w, (int)ark->player_h, WHITE);
    DrawCircle((int)ark->ball_x, (int)ark->ball_y, ark->ball_radius, WHITE);
    for (uint32_t i = 0; i < ark->lines; i++) {
        for (uint32_t j = 0; j < ark->bricks_per_line; j++) {
            if (bricks[i * ark->bricks_per_line + j] == 0) continue;
            int x = (int)(j * ark->brick_w);
            int y = (int)(ark->brick_offset_y + i * ark->brick_h);
            DrawRectangle(x + 1, y + 1, (int)ark->brick_w - 2, (int)ark->brick_h - 2, RED);
        }
    }
}

static void draw_tetris(const tfrl_tetris_snapshot *game, const unsigned char *cells) {
    if (!game || !cells || game->width == 0 || game->height == 0) return;
    int board_w = (int)game->width;
    int board_h = (int)game->height;
    int cell = GetScreenHeight() / (board_h + 2);
    if (cell < 8) cell = 8;
    int ox = 24;
    int oy = 24;
    DrawRectangle(ox - 2, oy - 2, board_w * cell + 4, board_h * cell + 4, DARKGRAY);
    for (int y = 0; y < board_h; y++) {
        for (int x = 0; x < board_w; x++) {
            Color c = cells[y * board_w + x] ? RED : (Color){30, 35, 45, 255};
            DrawRectangle(ox + x * cell, oy + y * cell, cell - 1, cell - 1, c);
        }
    }
    static const int pieces[7][4][2] = {{{1,0},{1,1},{1,2},{1,3}},{{0,0},{1,0},{0,1},{1,1}},{{1,0},{0,1},{1,1},{2,1}},{{1,0},{2,0},{0,1},{1,1}},{{0,0},{1,0},{1,1},{2,1}},{{0,0},{0,1},{0,2},{1,2}},{{2,0},{2,1},{1,2},{2,2}}};
    for (int i = 0; i < 4; i++) {
        int px = pieces[game->piece % 7][i][0], py = pieces[game->piece % 7][i][1];
        for (uint32_t r = 0; r < (game->rot & 3); r++) { int pivot = game->piece % 7 == 0 ? 3 : 2; int nx = pivot - py; py = px; px = nx; }
        int bx = (int)game->x + px, by = (int)game->y + py;
        if (bx >= 0 && by >= 0 && bx < board_w && by < board_h)
            DrawRectangle(ox + bx * cell, oy + by * cell, cell - 1, cell - 1, SKYBLUE);
    }
    DrawText(TextFormat("TETRIS  step %u/%u  lines %u  score %.0f", game->step,
                        game->max_steps, game->lines, game->score),
             ox, oy + board_h * cell + 12, 18, RAYWHITE);
    DrawText(TextFormat("reward %.3f%s", game->reward, game->done ? "  DONE" : ""),
             ox, oy + board_h * cell + 34, 18, game->done ? ORANGE : LIGHTGRAY);
}

void tfrl_viewer_draw(tfrl_viewer *viewer, const void *snapshot, size_t len) {
    if (!viewer || !snapshot || len < sizeof(tfrl_render_snapshot_header)) return;
    const unsigned char *bytes = (const unsigned char *)snapshot;
    const tfrl_render_snapshot_header *header = (const tfrl_render_snapshot_header *)bytes;
    const unsigned char *payload = bytes + sizeof(tfrl_render_snapshot_header);

    BeginDrawing();
    ClearBackground(RAYWHITE);

    if (header->api_version == TFRL_SNAPSHOT_API_VERSION && header->payload_kind == TFRL_SNAPSHOT_KIND_GRID) {
        if (header->payload_bytes < sizeof(tfrl_grid_snapshot_v3)) {
            EndDrawing();
            return;
        }
        const tfrl_grid_snapshot_v3 *grid = (const tfrl_grid_snapshot_v3 *)payload;
        const unsigned char *cells = payload + sizeof(tfrl_grid_snapshot_v3);
        draw_grid(grid, cells);
        EndDrawing();
        return;
    }

    if (header->api_version == TFRL_SNAPSHOT_API_VERSION && header->payload_kind == TFRL_SNAPSHOT_KIND_ARKANOID) {
        if (header->payload_bytes < sizeof(tfrl_arkanoid_snapshot)) {
            EndDrawing();
            return;
        }
        const tfrl_arkanoid_snapshot *ark = (const tfrl_arkanoid_snapshot *)payload;
        const unsigned char *bricks = payload + sizeof(tfrl_arkanoid_snapshot);
        draw_arkanoid(ark, bricks);
        EndDrawing();
        return;
    }

    if (header->api_version == TFRL_SNAPSHOT_API_VERSION && header->payload_kind == TFRL_SNAPSHOT_KIND_TETRIS) {
        if (header->payload_bytes < sizeof(tfrl_tetris_snapshot)) {
            EndDrawing();
            return;
        }
        const tfrl_tetris_snapshot *game = (const tfrl_tetris_snapshot *)payload;
        size_t cells_bytes = (size_t)game->width * (size_t)game->height;
        if (header->payload_bytes < sizeof(*game) + cells_bytes) {
            EndDrawing();
            return;
        }
        draw_tetris(game, payload + sizeof(*game));
        EndDrawing();
        return;
    }

    EndDrawing();
}

#else

#include <stdlib.h>

tfrl_viewer *tfrl_viewer_create(const tfrl_viewer_config *cfg) {
    (void)cfg;
    return NULL;
}

void tfrl_viewer_destroy(tfrl_viewer *viewer) {
    free(viewer);
}

int tfrl_viewer_should_close(tfrl_viewer *viewer) {
    (void)viewer;
    return 1;
}

void tfrl_viewer_draw(tfrl_viewer *viewer, const void *snapshot, size_t len) {
    (void)viewer;
    (void)snapshot;
    (void)len;
}

#endif
