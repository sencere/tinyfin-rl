#include "raylib.h"

#include "tinyfin_rl/rl_render.h"

#define MAZE_W 10
#define MAZE_H 10

static int g_cell = 48;
static int g_init = 0;

static int is_wall(int x, int y) {
    if (x < 0 || x >= MAZE_W || y < 0 || y >= MAZE_H) return 1;
    if (x == 4 && y != 2 && y != 7) return 1;
    if (y == 4 && x != 2 && x != 7) return 1;
    return 0;
}

int tfrl_renderer_init(int grid_size, int cell_size, const char *title) {
    if (g_init) return 1;
    (void)grid_size;
    if (cell_size <= 0) return 0;
    g_cell = cell_size;
    InitWindow(MAZE_W * g_cell, MAZE_H * g_cell + 60, title ? title : "maze_rooms");
    SetTargetFPS(60);
    g_init = 1;
    return 1;
}

void tfrl_renderer_draw(const tfrl_value *obs, double reward, int done) {
    if (!g_init) return;
    int idx = (int)obs->as.i64;
    int x = idx % MAZE_W;
    int y = idx / MAZE_W;
    BeginDrawing();
    ClearBackground(RAYWHITE);

    for (int yy = 0; yy < MAZE_H; yy++) {
        for (int xx = 0; xx < MAZE_W; xx++) {
            int px = xx * g_cell;
            int py = yy * g_cell;
            if (is_wall(xx, yy)) {
                DrawRectangle(px, py, g_cell, g_cell, DARKGRAY);
            } else {
                DrawRectangleLines(px, py, g_cell, g_cell, LIGHTGRAY);
            }
        }
    }
    DrawRectangle((MAZE_W - 1) * g_cell, (MAZE_H - 1) * g_cell, g_cell, g_cell, (Color){255, 210, 120, 255});
    DrawRectangle(x * g_cell, y * g_cell, g_cell, g_cell, (Color){70, 130, 255, 255});

    int text_y = MAZE_H * g_cell + 10;
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
