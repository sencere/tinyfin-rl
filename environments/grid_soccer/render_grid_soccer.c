#include "raylib.h"

#include "tinyfin_rl/rl_render.h"

#define SOCCER_W 8
#define SOCCER_H 6

static int g_cell = 48;
static int g_init = 0;

int tfrl_renderer_init(int grid_size, int cell_size, const char *title) {
    (void)grid_size;
    if (g_init) return 1;
    if (cell_size <= 0) return 0;
    g_cell = cell_size;
    InitWindow(SOCCER_W * g_cell, SOCCER_H * g_cell + 60, title ? title : "grid_soccer");
    SetTargetFPS(60);
    g_init = 1;
    return 1;
}

static void decode_obs(int obs, int *ax, int *ay, int *bx, int *by) {
    int board = SOCCER_W * SOCCER_H;
    int a = obs % board;
    int b = obs / board;
    *ax = a % SOCCER_W;
    *ay = a / SOCCER_W;
    *bx = b % SOCCER_W;
    *by = b / SOCCER_W;
}

void tfrl_renderer_draw(const tfrl_value *obs, double reward, int done) {
    if (!g_init) return;
    int ax = 0, ay = 0, bx = 0, by = 0;
    decode_obs((int)obs->as.i64, &ax, &ay, &bx, &by);
    BeginDrawing();
    ClearBackground(RAYWHITE);
    for (int yy = 0; yy < SOCCER_H; yy++) {
        for (int xx = 0; xx < SOCCER_W; xx++) {
            int px = xx * g_cell;
            int py = yy * g_cell;
            DrawRectangleLines(px, py, g_cell, g_cell, LIGHTGRAY);
        }
    }
    int goal_x = SOCCER_W - 1;
    int goal_y = SOCCER_H / 2;
    DrawRectangle(goal_x * g_cell, goal_y * g_cell, g_cell, g_cell, (Color){255, 210, 120, 255});
    DrawCircle(bx * g_cell + g_cell / 2, by * g_cell + g_cell / 2, g_cell / 4, ORANGE);
    DrawRectangle(ax * g_cell + 8, ay * g_cell + 8, g_cell - 16, g_cell - 16, BLUE);
    int text_y = SOCCER_H * g_cell + 10;
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
