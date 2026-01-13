#include <math.h>

#include "raylib.h"

#include "tinyfin_rl/rl_render.h"

static int g_init = 0;
static int g_width = 800;
static int g_height = 400;

int tfrl_renderer_init(int grid_size, int cell_size, const char *title) {
    (void)grid_size;
    (void)cell_size;
    if (g_init) return 1;
    InitWindow(g_width, g_height, title ? title : "cartpole_simple");
    SetTargetFPS(60);
    g_init = 1;
    return 1;
}

void tfrl_renderer_draw(const tfrl_value *obs, double reward, int done) {
    (void)reward;
    if (!g_init) return;
    const float *state = NULL;
    if (obs && obs->type == TFRL_VALUE_BUFFER) {
        state = (const float *)obs->as.buffer.data;
    }
    float x = state ? state[0] : 0.0f;
    float theta = state ? state[2] : 0.0f;

    BeginDrawing();
    ClearBackground(RAYWHITE);
    int center_y = g_height / 2 + 60;
    DrawLine(0, center_y, g_width, center_y, DARKGRAY);

    float cart_x = g_width / 2 + x * 100.0f;
    float cart_y = center_y - 20;
    DrawRectangle((int)(cart_x - 30), (int)(cart_y - 10), 60, 20, BLUE);

    float pole_len = 100.0f;
    float px = cart_x + pole_len * sinf(theta);
    float py = cart_y - pole_len * cosf(theta);
    DrawLine((int)cart_x, (int)cart_y, (int)px, (int)py, RED);
    DrawCircle((int)px, (int)py, 6, RED);

    if (done) {
        DrawText("done", 10, 10, 20, RED);
    }
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
