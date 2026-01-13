#include "raylib.h"

#include "tinyfin_rl/rl_render.h"

#define CLIFF_W 12
#define CLIFF_H 4

static int g_cell = 48;
static int g_init = 0;

int tfrl_renderer_init(int grid_size, int cell_size, const char *title) {
    (void)grid_size;
    if (g_init) return 1;
    if (cell_size <= 0) return 0;
    g_cell = cell_size;
    InitWindow(CLIFF_W * g_cell, CLIFF_H * g_cell + 60, title ? title : "cliffwalk");
    SetTargetFPS(60);
    g_init = 1;
    return 1;
}

void tfrl_renderer_draw(const tfrl_value *obs, double reward, int done) {
    if (!g_init) return;
    int idx = (int)obs->as.i64;
    int x = idx % CLIFF_W;
    int y = idx / CLIFF_W;
    BeginDrawing();
    ClearBackground(RAYWHITE);
    for (int yy = 0; yy < CLIFF_H; yy++) {
        for (int xx = 0; xx < CLIFF_W; xx++) {
            int px = xx * g_cell;
            int py = yy * g_cell;
            if (yy == CLIFF_H - 1 && xx > 0 && xx < CLIFF_W - 1) {
                DrawRectangle(px, py, g_cell, g_cell, (Color){200, 50, 50, 255});
            } else {
                DrawRectangleLines(px, py, g_cell, g_cell, LIGHTGRAY);
            }
        }
    }
    DrawRectangle((CLIFF_W - 1) * g_cell, (CLIFF_H - 1) * g_cell, g_cell, g_cell, (Color){255, 210, 120, 255});
    DrawRectangle(x * g_cell, y * g_cell, g_cell, g_cell, (Color){70, 130, 255, 255});
    int text_y = CLIFF_H * g_cell + 10;
    DrawText(TextFormat("reward: %.2f", reward), 10, text_y, 18, DARKGRAY);
    if (done) DrawText("done", 10, text_y + 24, 18, RED);
    EndDrawing();
}

int tfrl_renderer_should_close(void) {
    return g_init ? WindowShouldClose() : 1;
}

void tfrl_renderer_close(void) {
    if (!g_init) return;
    CloseWindow();
    g_init = 0;
}
