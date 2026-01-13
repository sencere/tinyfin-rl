#include "raylib.h"

#include "tinyfin_rl/rl_render.h"

static int g_line_len = 7;
static int g_cell_size = 64;
static int g_initialized = 0;

int tfrl_renderer_init(int grid_size, int cell_size, const char *title) {
    if (g_initialized) {
        return 1;
    }
    if (grid_size <= 0 || cell_size <= 0) {
        return 0;
    }
    g_line_len = grid_size;
    g_cell_size = cell_size;
    int screen_width = g_line_len * g_cell_size;
    int screen_height = g_cell_size + 60;
    InitWindow(screen_width, screen_height, title ? title : "tinyfin-rl lineworld");
    SetTargetFPS(60);
    g_initialized = 1;
    return 1;
}

void tfrl_renderer_draw(const tfrl_value *obs, double reward, int done) {
    if (!g_initialized) {
        return;
    }
    int pos = (int)obs->as.i64;
    int goal = g_line_len - 1;

    BeginDrawing();
    ClearBackground(RAYWHITE);

    for (int i = 0; i < g_line_len; i++) {
        int px = i * g_cell_size;
        DrawRectangleLines(px, 0, g_cell_size, g_cell_size, LIGHTGRAY);
    }

    DrawRectangle(goal * g_cell_size, 0, g_cell_size, g_cell_size, (Color){255, 210, 120, 255});
    DrawRectangle(pos * g_cell_size, 0, g_cell_size, g_cell_size, (Color){70, 130, 255, 255});

    int text_y = g_cell_size + 10;
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
