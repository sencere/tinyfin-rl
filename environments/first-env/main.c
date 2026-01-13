#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#include "tinyfin_rl/rl.h"

#define GRID_SIZE 10
#define CELL_SIZE 48
#define MAX_STEPS 200
#define ACTIONS 4

typedef struct {
    int x;
    int y;
    int steps;
    int goal_x;
    int goal_y;
} grid_env;

static int grid_state(const grid_env *env) {
    return env->y * GRID_SIZE + env->x;
}

static int load_q_table(const char *path, double q[GRID_SIZE * GRID_SIZE][ACTIONS]) {
    FILE *f = fopen(path, "r");
    if (!f) {
        return 0;
    }
    char line[128];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return 0;
    }
    while (fgets(line, sizeof(line), f)) {
        int s = 0;
        int a = 0;
        double v = 0.0;
        if (sscanf(line, "%d,%d,%lf", &s, &a, &v) == 3) {
            if (s >= 0 && s < GRID_SIZE * GRID_SIZE && a >= 0 && a < ACTIONS) {
                q[s][a] = v;
            }
        }
    }
    fclose(f);
    return 1;
}

static int argmax_action(const double q[GRID_SIZE * GRID_SIZE][ACTIONS], int state) {
    int best = 0;
    double best_v = q[state][0];
    for (int a = 1; a < ACTIONS; a++) {
        if (q[state][a] > best_v) {
            best_v = q[state][a];
            best = a;
        }
    }
    return best;
}

static int grid_reset(void *ctx, tfrl_value *out_obs) {
    grid_env *env = (grid_env *)ctx;
    env->x = 0;
    env->y = 0;
    env->steps = 0;
    env->goal_x = GRID_SIZE - 1;
    env->goal_y = GRID_SIZE - 1;
    out_obs->type = TFRL_VALUE_I64;
    out_obs->as.i64 = env->y * GRID_SIZE + env->x;
    return TFRL_OK;
}

static int grid_step(void *ctx, const tfrl_value *action, tfrl_step *out_step) {
    grid_env *env = (grid_env *)ctx;
    int act = (int)action->as.i64;
    if (act == 0 && env->y > 0) {
        env->y -= 1;
    } else if (act == 1 && env->y < GRID_SIZE - 1) {
        env->y += 1;
    } else if (act == 2 && env->x > 0) {
        env->x -= 1;
    } else if (act == 3 && env->x < GRID_SIZE - 1) {
        env->x += 1;
    }
    env->steps += 1;
    int reached = env->x == env->goal_x && env->y == env->goal_y;

    out_step->observation.type = TFRL_VALUE_I64;
    out_step->observation.as.i64 = env->y * GRID_SIZE + env->x;
    out_step->reward = reached ? 1.0 : -0.01;
    out_step->terminated = reached;
    out_step->truncated = env->steps >= MAX_STEPS;
    return TFRL_OK;
}

int main(void) {
    const int screen_width = GRID_SIZE * CELL_SIZE;
    const int screen_height = GRID_SIZE * CELL_SIZE + 80;
    InitWindow(screen_width, screen_height, "tinyfin-rl first-env");
    SetTargetFPS(60);

    grid_env env_state = {0};
    const tfrl_env_vtable env_vtable = {
        .reset = grid_reset,
        .step = grid_step,
        .seed = NULL,
        .destroy = NULL
    };
    tfrl_env env = {&env_state, &env_vtable};

    tfrl_value obs = {0};
    tfrl_step step = {0};
    tfrl_env_reset(&env, &obs);
    int done = 0;
    int auto_play = 0;
    double step_timer = 0.0;
    double step_interval = 0.2;
    double q_table[GRID_SIZE * GRID_SIZE][ACTIONS] = {0};
    int q_loaded = load_q_table("q_table.csv", q_table);

    while (!WindowShouldClose()) {
        int action = -1;
        if (IsKeyPressed(KEY_UP)) action = 0;
        if (IsKeyPressed(KEY_DOWN)) action = 1;
        if (IsKeyPressed(KEY_LEFT)) action = 2;
        if (IsKeyPressed(KEY_RIGHT)) action = 3;
        if (IsKeyPressed(KEY_R)) {
            tfrl_env_reset(&env, &obs);
            done = 0;
        }
        if (IsKeyPressed(KEY_P)) {
            auto_play = !auto_play;
        }
        if (IsKeyPressed(KEY_L)) {
            q_loaded = load_q_table("q_table.csv", q_table);
        }

        if (auto_play && q_loaded && !done) {
            step_timer += GetFrameTime();
            if (step_timer >= step_interval) {
                step_timer = 0.0;
                int state = grid_state(&env_state);
                action = argmax_action(q_table, state);
            } else {
                action = -1;
            }
        }

        if (action >= 0 && !done) {
            tfrl_value act = {.type = TFRL_VALUE_I64, .as.i64 = action};
            tfrl_env_step(&env, &act, &step);
            done = tfrl_step_done(&step);
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        for (int y = 0; y < GRID_SIZE; y++) {
            for (int x = 0; x < GRID_SIZE; x++) {
                int px = x * CELL_SIZE;
                int py = y * CELL_SIZE;
                DrawRectangleLines(px, py, CELL_SIZE, CELL_SIZE, LIGHTGRAY);
            }
        }

        DrawRectangle(env_state.goal_x * CELL_SIZE, env_state.goal_y * CELL_SIZE, CELL_SIZE, CELL_SIZE, (Color){255, 210, 120, 255});
        DrawRectangle(env_state.x * CELL_SIZE, env_state.y * CELL_SIZE, CELL_SIZE, CELL_SIZE, (Color){70, 130, 255, 255});

        int text_y = GRID_SIZE * CELL_SIZE + 10;
        DrawText("Arrow keys: step | R: reset | P: autoplay | L: reload policy", 10, text_y, 18, DARKGRAY);
        DrawText(TextFormat("steps: %d", env_state.steps), 10, text_y + 24, 18, DARKGRAY);
        DrawText(TextFormat("reward: %.2f", step.reward), 10, text_y + 48, 18, DARKGRAY);
        DrawText(TextFormat("mode: %s", auto_play ? "auto" : "manual"), screen_width - 160, text_y, 18, DARKGRAY);
        DrawText(TextFormat("policy: %s", q_loaded ? "loaded" : "missing"), screen_width - 160, text_y + 24, 18, DARKGRAY);
        if (done) {
            DrawText("done", screen_width - 80, text_y + 24, 18, RED);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
