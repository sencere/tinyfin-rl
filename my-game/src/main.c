#include "raylib.h"

int main(void)
{
    const int screen_width = 800;
    const int screen_height = 450;

    InitWindow(screen_width, screen_height, "raylib C example");
    SetTargetFPS(60);

    while (!WindowShouldClose())
    {
        BeginDrawing();

        ClearBackground(RAYWHITE);
        DrawText("Hello, raylib!", 300, 210, 20, DARKGRAY);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
