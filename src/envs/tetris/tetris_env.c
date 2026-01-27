#include "envs/envs_internal.h"

#include <stdlib.h>
#include <string.h>

static double tetris_lines_score(int lines) {
    static const double s[5] = {0.0, 1.0, 3.0, 5.0, 8.0};
    if (lines < 0) lines = 0;
    if (lines > 4) lines = 4;
    return s[lines];
}

static const int TETRIS_BLOCKS[7][4][2] = {
    {{0, 1}, {1, 1}, {2, 1}, {3, 1}}, /* I */
    {{1, 0}, {2, 0}, {1, 1}, {2, 1}}, /* O */
    {{1, 0}, {0, 1}, {1, 1}, {2, 1}}, /* T */
    {{0, 0}, {0, 1}, {1, 1}, {2, 1}}, /* L */
    {{2, 0}, {0, 1}, {1, 1}, {2, 1}}, /* J */
    {{1, 0}, {2, 0}, {0, 1}, {1, 1}}, /* S */
    {{0, 0}, {1, 0}, {1, 1}, {2, 1}}  /* Z */
};

static void tetris_rot_coord(int x, int y, int rot, int *out_x, int *out_y) {
    int rx = x;
    int ry = y;
    for (int i = 0; i < rot; i++) {
        int tmp = rx;
        rx = ry;
        ry = 3 - tmp;
    }
    *out_x = rx;
    *out_y = ry;
}

static double tetris_piece_depth(const tfrl_env *env) {
    int sum_y = 0;
    for (int i = 0; i < 4; i++) {
        int bx = 0;
        int by = 0;
        tetris_rot_coord(TETRIS_BLOCKS[env->tetris.piece][i][0],
                          TETRIS_BLOCKS[env->tetris.piece][i][1],
                          env->tetris.rot, &bx, &by);
        sum_y += env->tetris.y + by;
    }
    double avg_y = (double)sum_y / 4.0;
    if (avg_y < 0.0) avg_y = 0.0;
    if (avg_y > (double)(TETRIS_H - 1)) avg_y = (double)(TETRIS_H - 1);
    return avg_y / (double)(TETRIS_H - 1);
}

static int tetris_grid_fits(const int grid[TETRIS_W][TETRIS_H], int piece, int rot, int x, int y) {
    for (int i = 0; i < 4; i++) {
        int bx = 0;
        int by = 0;
        tetris_rot_coord(TETRIS_BLOCKS[piece][i][0], TETRIS_BLOCKS[piece][i][1], rot, &bx, &by);
        int gx = x + bx;
        int gy = y + by;
        if (gx < 0 || gx >= TETRIS_W || gy < 0 || gy >= TETRIS_H) return 0;
        if (grid[gx][gy] != 0) return 0;
    }
    return 1;
}

static void tetris_grid_place(int grid[TETRIS_W][TETRIS_H], int piece, int rot, int x, int y) {
    for (int i = 0; i < 4; i++) {
        int bx = 0;
        int by = 0;
        tetris_rot_coord(TETRIS_BLOCKS[piece][i][0], TETRIS_BLOCKS[piece][i][1], rot, &bx, &by);
        int gx = x + bx;
        int gy = y + by;
        if (gx >= 0 && gx < TETRIS_W && gy >= 0 && gy < TETRIS_H) {
            grid[gx][gy] = piece + 1;
        }
    }
}

static int tetris_grid_clear_lines(int grid[TETRIS_W][TETRIS_H]) {
    int cleared = 0;
    for (int y = TETRIS_H - 1; y >= 0; y--) {
        int full = 1;
        for (int x = 0; x < TETRIS_W; x++) {
            if (grid[x][y] == 0) {
                full = 0;
                break;
            }
        }
        if (full) {
            for (int yy = y; yy > 0; yy--) {
                for (int x = 0; x < TETRIS_W; x++) {
                    grid[x][yy] = grid[x][yy - 1];
                }
            }
            for (int x = 0; x < TETRIS_W; x++) {
                grid[x][0] = 0;
            }
            cleared++;
            y++;
        }
    }
    return cleared;
}

static void tetris_grid_stats(const int grid[TETRIS_W][TETRIS_H], int *agg_height, int *holes, int *bumpiness) {
    int heights[TETRIS_W] = {0};
    int total_holes = 0;
    for (int x = 0; x < TETRIS_W; x++) {
        int found = 0;
        int height = 0;
        int column_holes = 0;
        for (int y = 0; y < TETRIS_H; y++) {
            if (grid[x][y] != 0) {
                if (!found) {
                    found = 1;
                    height = TETRIS_H - y;
                }
            } else if (found) {
                column_holes++;
            }
        }
        heights[x] = height;
        total_holes += column_holes;
    }
    int total_height = 0;
    for (int x = 0; x < TETRIS_W; x++) {
        total_height += heights[x];
    }
    int bump = 0;
    for (int x = 1; x < TETRIS_W; x++) {
        int diff = heights[x] - heights[x - 1];
        if (diff < 0) diff = -diff;
        bump += diff;
    }
    *agg_height = total_height;
    *holes = total_holes;
    *bumpiness = bump;
}

static double tetris_grid_score(const int grid[TETRIS_W][TETRIS_H], int lines_cleared) {
    int agg_height = 0;
    int holes = 0;
    int bumpiness = 0;
    tetris_grid_stats(grid, &agg_height, &holes, &bumpiness);
    return -0.51066 * (double)agg_height
           + 0.76066 * (double)lines_cleared
           - 0.35663 * (double)holes
           - 0.18448 * (double)bumpiness;
}

static void tetris_plan_best(const tfrl_env *env, int piece, int next_piece, int *out_rot, int *out_x) {
    double best_score = -1e30;
    int best_rot = env->tetris.rot;
    int best_x = env->tetris.x;
    int grid0[TETRIS_W][TETRIS_H];
    memcpy(grid0, env->tetris.grid, sizeof(grid0));

    for (int rot = 0; rot < 4; rot++) {
        for (int x = -2; x < TETRIS_W + 2; x++) {
            int y = 0;
            if (!tetris_grid_fits(grid0, piece, rot, x, y)) continue;
            while (tetris_grid_fits(grid0, piece, rot, x, y + 1)) y++;

            int grid1[TETRIS_W][TETRIS_H];
            memcpy(grid1, grid0, sizeof(grid1));
            tetris_grid_place(grid1, piece, rot, x, y);
            int lines1 = tetris_grid_clear_lines(grid1);
            double score1 = tetris_grid_score(grid1, lines1);

            double best_next = score1;
            if (next_piece >= 0) {
                best_next = -1e30;
                for (int rot2 = 0; rot2 < 4; rot2++) {
                    for (int x2 = -2; x2 < TETRIS_W + 2; x2++) {
                        int y2 = 0;
                        if (!tetris_grid_fits(grid1, next_piece, rot2, x2, y2)) continue;
                        while (tetris_grid_fits(grid1, next_piece, rot2, x2, y2 + 1)) y2++;
                        int grid2[TETRIS_W][TETRIS_H];
                        memcpy(grid2, grid1, sizeof(grid2));
                        tetris_grid_place(grid2, next_piece, rot2, x2, y2);
                        int lines2 = tetris_grid_clear_lines(grid2);
                        double score2 = tetris_grid_score(grid2, lines2);
                        if (score2 > best_next) best_next = score2;
                    }
                }
            }

            if (best_next > best_score) {
                best_score = best_next;
                best_rot = rot;
                best_x = x;
            }
        }
    }

    *out_rot = best_rot;
    *out_x = best_x;
}

static int tetris_plan_action(const tfrl_env *env) {
    int target_rot = env->tetris.rot;
    int target_x = env->tetris.x;
    tetris_plan_best(env, env->tetris.piece, env->tetris.next_piece, &target_rot, &target_x);
    if (env->tetris.rot != target_rot) return 3;
    if (env->tetris.x < target_x) return 2;
    if (env->tetris.x > target_x) return 1;
    return 4;
}

static int tetris_piece_fits(const tfrl_env *env, int x, int y, int rot) {
    for (int i = 0; i < 4; i++) {
        int bx = 0;
        int by = 0;
        tetris_rot_coord(TETRIS_BLOCKS[env->tetris.piece][i][0],
                          TETRIS_BLOCKS[env->tetris.piece][i][1],
                          rot, &bx, &by);
        int gx = x + bx;
        int gy = y + by;
        if (gx < 0 || gx >= TETRIS_W || gy < 0 || gy >= TETRIS_H) return 0;
        if (env->tetris.grid[gx][gy] != 0) return 0;
    }
    return 1;
}

static void tetris_place_piece(tfrl_env *env) {
    for (int i = 0; i < 4; i++) {
        int bx = 0;
        int by = 0;
        tetris_rot_coord(TETRIS_BLOCKS[env->tetris.piece][i][0],
                          TETRIS_BLOCKS[env->tetris.piece][i][1],
                          env->tetris.rot, &bx, &by);
        int gx = env->tetris.x + bx;
        int gy = env->tetris.y + by;
        if (gx >= 0 && gx < TETRIS_W && gy >= 0 && gy < TETRIS_H) {
            env->tetris.grid[gx][gy] = env->tetris.piece + 1;
        }
    }
}

static int tetris_clear_lines(tfrl_env *env) {
    int cleared = 0;
    for (int y = TETRIS_H - 1; y >= 0; y--) {
        int full = 1;
        for (int x = 0; x < TETRIS_W; x++) {
            if (env->tetris.grid[x][y] == 0) {
                full = 0;
                break;
            }
        }
        if (full) {
            for (int yy = y; yy > 0; yy--) {
                for (int x = 0; x < TETRIS_W; x++) {
                    env->tetris.grid[x][yy] = env->tetris.grid[x][yy - 1];
                }
            }
            for (int x = 0; x < TETRIS_W; x++) {
                env->tetris.grid[x][0] = 0;
            }
            cleared++;
            y++;
        }
    }
    env->tetris.lines += cleared;
    return cleared;
}

static int tetris_spawn_piece(tfrl_env *env) {
    env->tetris.piece = env->tetris.next_piece;
    env->tetris.next_piece = tfrl_env_rand_range_int(env, 0, 6);
    env->tetris.rot = 0;
    env->tetris.x = (TETRIS_W - 4) / 2;
    env->tetris.y = 0;
    return tetris_piece_fits(env, env->tetris.x, env->tetris.y, env->tetris.rot);
}

enum {
    TETRIS_OBS_DIMS = 27,
    TETRIS_DISC_OBS_N = 4096
};

static const tfrl_obs_field TETRIS_OBS_FIELDS[] = {
    {.name = "heights", .len = TETRIS_W, .dims = 1, .shape = {TETRIS_W, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "holes", .len = TETRIS_W, .dims = 1, .shape = {TETRIS_W, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "x", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "y", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "rot", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "piece", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "next_piece", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "bumpiness", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "max_height", .len = 1, .dims = 1, .shape = {1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
};

static const tfrl_obs_layout TETRIS_OBS_LAYOUT = {
    .field_count = (int)(sizeof(TETRIS_OBS_FIELDS) / sizeof(TETRIS_OBS_FIELDS[0])),
    .fields = TETRIS_OBS_FIELDS,
};

static void tetris_calc_features(const tfrl_env *env, float *heights_out, float *holes_out,
                                 float *bumpiness_out, float *max_height_out, float *total_holes_out) {
    int heights[TETRIS_W] = {0};
    int holes[TETRIS_W] = {0};
    int total_holes = 0;
    int max_height = 0;
    for (int x = 0; x < TETRIS_W; x++) {
        int found = 0;
        int height = 0;
        int column_holes = 0;
        for (int y = 0; y < TETRIS_H; y++) {
            if (env->tetris.grid[x][y] != 0) {
                if (!found) {
                    found = 1;
                    height = TETRIS_H - y;
                }
            } else if (found) {
                column_holes++;
            }
        }
        heights[x] = height;
        holes[x] = column_holes;
        total_holes += column_holes;
        if (height > max_height) max_height = height;
    }

    float bumpiness = 0.0f;
    for (int x = 1; x < TETRIS_W; x++) {
        int diff = heights[x] - heights[x - 1];
        if (diff < 0) diff = -diff;
        bumpiness += (float)diff;
    }

    float height_norm = (float)TETRIS_H;
    float bump_norm = (float)(TETRIS_H * (TETRIS_W - 1));
    float holes_norm = (float)TETRIS_H;
    for (int x = 0; x < TETRIS_W; x++) {
        heights_out[x] = heights[x] / height_norm;
        holes_out[x] = holes[x] / holes_norm;
    }
    *bumpiness_out = bumpiness / bump_norm;
    *max_height_out = (float)max_height / height_norm;
    *total_holes_out = (float)total_holes / (float)(TETRIS_W * TETRIS_H);
}

static void tetris_fill_obs(const tfrl_env *env, tfrl_obs *obs) {
    float heights[TETRIS_W];
    float holes[TETRIS_W];
    float bumpiness = 0.0f;
    float max_height = 0.0f;
    float total_holes = 0.0f;
    tetris_calc_features(env, heights, holes, &bumpiness, &max_height, &total_holes);
    (void)total_holes;

    float pos_x[1] = {(float)env->tetris.x / (float)(TETRIS_W - 1)};
    float pos_y[1] = {(float)env->tetris.y / (float)(TETRIS_H - 1)};
    float rot[1] = {(float)env->tetris.rot / 3.0f};
    float piece[1] = {(float)env->tetris.piece / 6.0f};
    float next_piece[1] = {(float)env->tetris.next_piece / 6.0f};
    float bump[1] = {bumpiness};
    float max_h[1] = {max_height};
    const float *fields[] = {heights, holes, pos_x, pos_y, rot, piece, next_piece, bump, max_h};
    if (!tfrl_obs_flatten(&TETRIS_OBS_LAYOUT, fields, obs)) {
        obs->data_len = 0;
    }
}

static int tetris_hash_obs(const tfrl_env *env) {
    uint32_t h = 2166136261u;
    for (int y = 0; y < TETRIS_H; y++) {
        for (int x = 0; x < TETRIS_W; x++) {
            h ^= (uint32_t)env->tetris.grid[x][y];
            h *= 16777619u;
        }
    }
    h ^= (uint32_t)env->tetris.piece;
    h *= 16777619u;
    h ^= (uint32_t)env->tetris.next_piece;
    h *= 16777619u;
    h ^= (uint32_t)env->tetris.rot;
    h *= 16777619u;
    h ^= (uint32_t)env->tetris.x;
    h *= 16777619u;
    h ^= (uint32_t)env->tetris.y;
    h *= 16777619u;
    h ^= (uint32_t)env->tetris.lines;
    h *= 16777619u;
    return (int)(h % (uint32_t)TETRIS_DISC_OBS_N);
}

static const tfrl_env_spec TETRIS_SPEC = {
    .name = "tetris",
    .obs_n = TETRIS_OBS_DIMS,
    .action_n = 5,
    .max_steps = TETRIS_MAX_STEPS,
    .width = TETRIS_W,
    .height = TETRIS_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = TETRIS_OBS_DIMS,
    .action_dims = 1,
    .obs_shape = {TETRIS_OBS_DIMS, 0},
    .action_shape = {5, 0},
    .obs_dtype = TFRL_DTYPE_FLOAT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = 1.0,
    .action_low = 0.0,
    .action_high = 4.0,
    .agent_count = 1,
    .obs_layout = &TETRIS_OBS_LAYOUT,
};

static const tfrl_env_spec TETRIS_DISC_SPEC = {
    .name = "tetris_disc",
    .obs_n = TETRIS_DISC_OBS_N,
    .action_n = 5,
    .max_steps = TETRIS_MAX_STEPS,
    .width = TETRIS_W,
    .height = TETRIS_H,
    .obs_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {TETRIS_DISC_OBS_N, 0},
    .action_shape = {5, 0},
    .obs_dtype = TFRL_DTYPE_INT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = (double)(TETRIS_DISC_OBS_N - 1),
    .action_low = 0.0,
    .action_high = 4.0,
    .agent_count = 1,
    .obs_layout = NULL,
};

const tfrl_env_spec *tfrl_env_spec_tetris(void) {
    return &TETRIS_SPEC;
}

const tfrl_env_spec *tfrl_env_spec_tetris_disc(void) {
    return &TETRIS_DISC_SPEC;
}

tfrl_obs tfrl_env_reset_tetris(tfrl_env *env, uint64_t seed) {
    (void)seed;
    tfrl_env_reset_state(env);
    for (int x = 0; x < TETRIS_W; x++) {
        for (int y = 0; y < TETRIS_H; y++) {
            env->tetris.grid[x][y] = 0;
        }
    }
    env->tetris.lines = 0;
    env->tetris.score = 0.0;
    env->tetris.next_piece = tfrl_env_rand_range_int(env, 0, 6);
    tetris_spawn_piece(env);

    tfrl_obs obs = {0};
    tetris_fill_obs(env, &obs);
    return obs;
}

tfrl_obs tfrl_env_reset_tetris_disc(tfrl_env *env, uint64_t seed) {
    tfrl_obs obs = tfrl_env_reset_tetris(env, seed);
    obs.index = tetris_hash_obs(env);
    obs.data_len = 0;
    return obs;
}

typedef struct {
    int lines_cleared;
    int piece_locked;
    double depth_bonus;
    int done;
} tetris_step_info;

static void tetris_step_apply(tfrl_env *env, tfrl_action action, tetris_step_info *info) {
    int act = action.index;
    if (act < 0) {
        act = tetris_plan_action(env);
    }
    if (act < 0) act = 0;
    if (act > 4) act = 4;
    if (act == 1) {
        if (tetris_piece_fits(env, env->tetris.x - 1, env->tetris.y, env->tetris.rot)) {
            env->tetris.x -= 1;
        }
    } else if (act == 2) {
        if (tetris_piece_fits(env, env->tetris.x + 1, env->tetris.y, env->tetris.rot)) {
            env->tetris.x += 1;
        }
    } else if (act == 3) {
        int next_rot = (env->tetris.rot + 1) % 4;
        if (tetris_piece_fits(env, env->tetris.x, env->tetris.y, next_rot)) {
            env->tetris.rot = next_rot;
        }
    }

    info->lines_cleared = 0;
    info->piece_locked = 0;
    info->depth_bonus = 0.0;
    info->done = 0;

    if (act == 4) {
        while (tetris_piece_fits(env, env->tetris.x, env->tetris.y + 1, env->tetris.rot)) {
            env->tetris.y += 1;
        }
        info->depth_bonus = tetris_piece_depth(env);
        tetris_place_piece(env);
        info->piece_locked = 1;
        info->lines_cleared = tetris_clear_lines(env);
        if (!tetris_spawn_piece(env)) {
            info->done = 1;
        }
    } else {
        if (tetris_piece_fits(env, env->tetris.x, env->tetris.y + 1, env->tetris.rot)) {
            env->tetris.y += 1;
        } else {
            info->depth_bonus = tetris_piece_depth(env);
            tetris_place_piece(env);
            info->piece_locked = 1;
            info->lines_cleared = tetris_clear_lines(env);
            if (!tetris_spawn_piece(env)) {
                info->done = 1;
            }
        }
    }
}

tfrl_step_result tfrl_env_step_tetris(tfrl_env *env, tfrl_action action) {
    tetris_step_info info;
    tetris_step_apply(env, action, &info);

    env->steps += 1;
    int done = info.done || (env->steps >= TETRIS_MAX_STEPS);

    const double w_lines = 1.0;
    const double w_total_lines = 0.05;
    const double w_time = 0.001;
    const double w_depth = 0.5;
    const double w_game_over = -1.0;
    double prev_score = env->tetris.score;
    env->tetris.score += w_lines * tetris_lines_score(info.lines_cleared);
    env->tetris.score += w_total_lines * (double)env->tetris.lines;
    env->tetris.score += w_time;
    if (info.piece_locked && !done) env->tetris.score += w_depth * info.depth_bonus;
    if (done) env->tetris.score += w_game_over;
    double reward = env->tetris.score - prev_score;

    tfrl_step_result out = {0};
    tetris_fill_obs(env, &out.observation);
    out.reward = reward;
    out.done = done;
    env->last_reward = (float)out.reward;
    env->last_done = out.done;
    return out;
}

tfrl_step_result tfrl_env_step_tetris_disc(tfrl_env *env, tfrl_action action) {
    tfrl_step_result out = tfrl_env_step_tetris(env, action);
    out.observation.index = tetris_hash_obs(env);
    out.observation.data_len = 0;
    return out;
}
