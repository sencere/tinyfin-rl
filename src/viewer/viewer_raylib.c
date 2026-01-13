#include "viewer.h"

#include <stdint.h>
#include <stdlib.h>

#include "raylib.h"

#include "core/render_snapshot.h"

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

struct tfrl_viewer {
    int init;
    int fps;
    const char *title;
    int width;
    int height;
};

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

tfrl_viewer *tfrl_viewer_create(const tfrl_viewer_config *cfg) {
    tfrl_viewer *viewer = (tfrl_viewer *)calloc(1, sizeof(tfrl_viewer));
    if (!viewer) return NULL;
    viewer->fps = cfg ? cfg->fps : 60;
    viewer->title = cfg ? cfg->title : NULL;
    return viewer;
}

void tfrl_viewer_draw(tfrl_viewer *viewer, const void *snapshot, size_t len) {
    if (!viewer || !snapshot || len < sizeof(tfrl_render_snapshot_header) + sizeof(tfrl_grid_snapshot)) return;
    const unsigned char *bytes = (const unsigned char *)snapshot;
    const tfrl_render_snapshot_header *header = (const tfrl_render_snapshot_header *)bytes;
    const unsigned char *payload = bytes + sizeof(tfrl_render_snapshot_header);
    const tfrl_grid_snapshot *grid = NULL;
    tfrl_grid_snapshot_v2 grid_v2 = {0};
    size_t payload_header_size = 0;
    if (header->api_version == 0x0002) {
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

    if (!ensure_init(viewer, (int)grid->width, (int)grid->height)) return;
    if (WindowShouldClose()) return;

    BeginDrawing();
    ClearBackground(RAYWHITE);
    for (uint32_t y = 0; y < grid->height; y++) {
        for (uint32_t x = 0; x < grid->width; x++) {
            int px = (int)x * CELL_SIZE;
            int py = (int)y * CELL_SIZE;
            if (walls[y * grid->width + x]) {
                DrawRectangle(px, py, CELL_SIZE, CELL_SIZE, DARKGRAY);
            } else {
                DrawRectangleLines(px, py, CELL_SIZE, CELL_SIZE, LIGHTGRAY);
            }
        }
    }
    DrawRectangle((int)grid->goal_x * CELL_SIZE, (int)grid->goal_y * CELL_SIZE, CELL_SIZE, CELL_SIZE, (Color){255, 210, 120, 255});
    DrawRectangle((int)grid->agent_x * CELL_SIZE, (int)grid->agent_y * CELL_SIZE, CELL_SIZE, CELL_SIZE, (Color){70, 130, 255, 255});

    int text_y = (int)grid->height * CELL_SIZE + 10;
    DrawText(TextFormat("step: %u / %u", grid->step, grid->max_steps), 10, text_y, 18, DARKGRAY);
    if (header->api_version == 0x0002) {
        DrawText(TextFormat("reward: %.3f", grid_v2.reward), 200, text_y, 18, DARKGRAY);
        DrawText(TextFormat("done: %u", grid_v2.done), 360, text_y, 18, DARKGRAY);
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
    free(viewer);
}
