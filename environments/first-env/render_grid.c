#include "raylib.h"

#include "tinyfin_rl/rl_render.h"

static int g_grid_size = 10;
static int g_cell_size = 48;
static int g_initialized = 0;

int tfrl_renderer_init(int grid_size, int cell_size, const char *title) {
    if (g_initialized) {
        return 1;
    }
    if (grid_size <= 0 || cell_size <= 0) {
        return 0;
    }
    g_grid_size = grid_size;
    g_cell_size = cell_size;
    int screen_width = g_grid_size * g_cell_size;
    int screen_height = g_grid_size * g_cell_size + 80;
    InitWindow(screen_width, screen_height, title ? title : "tinyfin-rl grid");
    SetTargetFPS(60);
    g_initialized = 1;
    return 1;
}

void tfrl_renderer_draw(const tfrl_value *obs, double reward, int done) {
    if (!g_initialized) {
        return;
    }
    int idx = (int)obs->as.i64;
    int x = idx % g_grid_size;
    int y = idx / g_grid_size;
    int goal_x = g_grid_size - 1;
    int goal_y = g_grid_size - 1;

    BeginDrawing();
    ClearBackground(RAYWHITE);

    for (int gy = 0; gy < g_grid_size; gy++) {
        for (int gx = 0; gx < g_grid_size; gx++) {
            int px = gx * g_cell_size;
            int py = gy * g_cell_size;
            DrawRectangleLines(px, py, g_cell_size, g_cell_size, LIGHTGRAY);
        }
    }

    DrawRectangle(goal_x * g_cell_size, goal_y * g_cell_size, g_cell_size, g_cell_size, (Color){255, 210, 120, 255});
    DrawRectangle(x * g_cell_size, y * g_cell_size, g_cell_size, g_cell_size, (Color){70, 130, 255, 255});

    int text_y = g_grid_size * g_cell_size + 10;
    DrawText(TextFormat("reward: %.2f", reward), 10, text_y, 18, DARKGRAY);
    if (done) {
        DrawText("done", 10, text_y + 24, 18, RED);
    }
    EndDrawing();
}

int tfrl_renderer_should_close(void) {
    if (!g_initialized) {
        return 1;
    }
    return WindowShouldClose();
}

void tfrl_renderer_close(void) {
    if (!g_initialized) {
        return;
    }
    CloseWindow();
    g_initialized = 0;
}
