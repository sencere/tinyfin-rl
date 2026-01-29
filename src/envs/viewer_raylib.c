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
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            unsigned char v = cells[y * w + x];
            Color c = (v > 0) ? DARKGRAY : RAYWHITE;
            DrawRectangle((int)(x * cell_w), (int)(y * cell_h), (int)cell_w, (int)cell_h, c);
        }
    }
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
