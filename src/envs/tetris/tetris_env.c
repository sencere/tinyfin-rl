#include "envs/envs_internal.h"

#include <string.h>

static const int TETRIS_PIECE_COUNT = 1;
static const int TETRIS_PIECES[1][4][2] = {
    {{0, 0}, {1, 0}, {0, 1}, {1, 1}},
};

static int tetris_can_place(const tfrl_env *env, int px, int py);

static const tfrl_obs_field TETRIS_OBS_FIELDS[] = {
    {.name = "col_heights", .len = TETRIS_W, .dims = 1, .shape = {TETRIS_W, 0},
     .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "col_holes", .len = TETRIS_W, .dims = 1, .shape = {TETRIS_W, 0},
     .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "col_diffs", .len = TETRIS_W - 1, .dims = 1,
     .shape = {TETRIS_W - 1, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "piece", .len = 4, .dims = 1, .shape = {4, 0}, .dtype = TFRL_DTYPE_FLOAT32},
    {.name = "metrics", .len = 7, .dims = 1, .shape = {7, 0}, .dtype = TFRL_DTYPE_FLOAT32},
};

static const tfrl_obs_layout TETRIS_OBS_LAYOUT = {
    .field_count = (int)(sizeof(TETRIS_OBS_FIELDS) / sizeof(TETRIS_OBS_FIELDS[0])),
    .fields = TETRIS_OBS_FIELDS,
};

static const tfrl_env_spec TETRIS_SPEC = {
    .name = "tetris",
    .obs_n = TETRIS_OBS_DIMS,
    .action_n = 4,
    .max_steps = TETRIS_MAX_STEPS,
    .width = TETRIS_W,
    .height = TETRIS_H,
    .obs_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = TETRIS_OBS_DIMS,
    .action_dims = 1,
    .obs_shape = {TETRIS_OBS_DIMS, 0},
    .action_shape = {4, 0},
    .obs_dtype = TFRL_DTYPE_FLOAT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = 1.0,
    .action_low = 0.0,
    .action_high = 3.0,
    .agent_count = 1,
    .obs_layout = &TETRIS_OBS_LAYOUT,
};

static const tfrl_env_spec TETRIS_DISC_SPEC = {
    .name = "tetris_disc",
    .obs_n = TETRIS_DISC_OBS_N,
    .action_n = 4,
    .max_steps = TETRIS_MAX_STEPS,
    .width = TETRIS_W,
    .height = TETRIS_H,
    .obs_type = TFRL_SPACE_DISCRETE,
    .action_type = TFRL_SPACE_DISCRETE,
    .obs_dims = 1,
    .action_dims = 1,
    .obs_shape = {TETRIS_DISC_OBS_N, 0},
    .action_shape = {4, 0},
    .obs_dtype = TFRL_DTYPE_INT32,
    .action_dtype = TFRL_DTYPE_INT32,
    .obs_low = 0.0,
    .obs_high = (double)(TETRIS_DISC_OBS_N - 1),
    .action_low = 0.0,
    .action_high = 3.0,
    .agent_count = 1,
    .obs_layout = NULL,
};

const tfrl_env_spec *tfrl_env_spec_tetris(void) { return &TETRIS_SPEC; }
const tfrl_env_spec *tfrl_env_spec_tetris_disc(void) { return &TETRIS_DISC_SPEC; }

static void tetris_spawn_piece(tfrl_env *env) {
    env->tetris.piece = 0;
    env->tetris.rot = 0;
    env->tetris.x = TETRIS_W / 2 - 1;
    env->tetris.y = 0;
}

static float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static void tetris_compute_columns(const tfrl_env *env, int *heights, int *holes,
                                   int *total_filled) {
    if (total_filled) *total_filled = 0;
    for (int x = 0; x < TETRIS_W; x++) {
        int height = 0;
        int hole = 0;
        int found = 0;
        for (int y = 0; y < TETRIS_H; y++) {
            if (env->tetris.grid[x][y]) {
                if (total_filled) (*total_filled)++;
                if (!found) {
                    height = TETRIS_H - y;
                    found = 1;
                }
            } else if (found) {
                hole++;
            }
        }
        heights[x] = height;
        holes[x] = hole;
    }
}

static void tetris_fill_obs(const tfrl_env *env, tfrl_obs *obs) {
    if (!obs) return;
    int heights[TETRIS_W];
    int holes[TETRIS_W];
    int total_filled = 0;
    tetris_compute_columns(env, heights, holes, &total_filled);

    float heights_f[TETRIS_W];
    float holes_f[TETRIS_W];
    float diffs_f[TETRIS_W - 1];
    int sum_heights = 0;
    int sum_holes = 0;
    int max_height = 0;
    int sum_diff = 0;

    for (int x = 0; x < TETRIS_W; x++) {
        sum_heights += heights[x];
        sum_holes += holes[x];
        if (heights[x] > max_height) max_height = heights[x];
        heights_f[x] = clamp01((float)heights[x] / (float)TETRIS_H);
        holes_f[x] = clamp01((float)holes[x] / (float)TETRIS_H);
        if (x < TETRIS_W - 1) {
            int diff = heights[x] - heights[x + 1];
            if (diff < 0) diff = -diff;
            sum_diff += diff;
            diffs_f[x] = clamp01((float)diff / (float)TETRIS_H);
        }
    }

    float piece_x = clamp01((float)env->tetris.x / (float)(TETRIS_W - 1));
    float piece_y = clamp01((float)env->tetris.y / (float)(TETRIS_H - 1));
    float piece_rot = clamp01((float)env->tetris.rot / 3.0f);
    float piece_kind = 0.0f;
    if (TETRIS_PIECE_COUNT > 1) {
        piece_kind = clamp01((float)env->tetris.piece / (float)(TETRIS_PIECE_COUNT - 1));
    }
    float piece_f[4] = {piece_x, piece_y, piece_rot, piece_kind};

    int drop_dist = 0;
    while (tetris_can_place(env, env->tetris.x, env->tetris.y + drop_dist + 1)) {
        drop_dist++;
    }

    float metrics[7];
    metrics[0] = clamp01((float)sum_heights / (float)(TETRIS_W * TETRIS_H));
    metrics[1] = clamp01((float)max_height / (float)TETRIS_H);
    metrics[2] = clamp01((float)sum_holes / (float)(TETRIS_W * TETRIS_H));
    metrics[3] = clamp01((float)sum_diff / (float)(TETRIS_W * TETRIS_H));
    metrics[4] = clamp01((float)total_filled / (float)(TETRIS_W * TETRIS_H));
    metrics[5] = clamp01((float)env->tetris.lines / 100.0f);
    metrics[6] = clamp01((float)drop_dist / (float)TETRIS_H);

    const float *fields[] = {heights_f, holes_f, diffs_f, piece_f, metrics};
    if (!tfrl_obs_flatten(&TETRIS_OBS_LAYOUT, fields, obs)) {
        obs->data_len = 0;
    }
}

static int tetris_hash_obs(const tfrl_env *env) {
    int heights[TETRIS_W];
    int holes[TETRIS_W];
    int total_filled = 0;
    tetris_compute_columns(env, heights, holes, &total_filled);

    uint32_t h = 2166136261u;
    for (int x = 0; x < TETRIS_W; x++) {
        h ^= (uint32_t)(heights[x] & 0xff); h *= 16777619u;
        h ^= (uint32_t)(holes[x] & 0xff); h *= 16777619u;
    }
    h ^= (uint32_t)env->tetris.x; h *= 16777619u;
    h ^= (uint32_t)env->tetris.y; h *= 16777619u;
    h ^= (uint32_t)env->tetris.rot; h *= 16777619u;
    h ^= (uint32_t)env->tetris.piece; h *= 16777619u;
    return (int)(h % (uint32_t)TETRIS_DISC_OBS_N);
}

tfrl_obs tfrl_env_reset_tetris(tfrl_env *env, uint64_t seed) {
    tfrl_env_reset_state(env);
    env->rng_state = (uint32_t)(seed ? seed : 1u);
    memset(env->tetris.grid, 0, sizeof(env->tetris.grid));
    env->tetris.lines = 0;
    env->tetris.score = 0.0;
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

static int tetris_can_place(const tfrl_env *env, int px, int py) {
    for (int i = 0; i < 4; i++) {
        int x = px + TETRIS_PIECES[0][i][0];
        int y = py + TETRIS_PIECES[0][i][1];
        if (x < 0 || x >= TETRIS_W || y < 0 || y >= TETRIS_H) return 0;
        if (env->tetris.grid[x][y]) return 0;
    }
    return 1;
}

static void tetris_lock_piece(tfrl_env *env) {
    for (int i = 0; i < 4; i++) {
        int x = env->tetris.x + TETRIS_PIECES[0][i][0];
        int y = env->tetris.y + TETRIS_PIECES[0][i][1];
        if (x >= 0 && x < TETRIS_W && y >= 0 && y < TETRIS_H) {
            env->tetris.grid[x][y] = 1;
        }
    }
}

static int tetris_clear_lines(tfrl_env *env) {
    int cleared = 0;
    for (int y = 0; y < TETRIS_H; y++) {
        int full = 1;
        for (int x = 0; x < TETRIS_W; x++) {
            if (!env->tetris.grid[x][y]) full = 0;
        }
        if (full) {
            cleared++;
            for (int yy = y; yy > 0; yy--) {
                for (int x = 0; x < TETRIS_W; x++) {
                    env->tetris.grid[x][yy] = env->tetris.grid[x][yy - 1];
                }
            }
            for (int x = 0; x < TETRIS_W; x++) env->tetris.grid[x][0] = 0;
        }
    }
    env->tetris.lines += cleared;
    return cleared;
}

tfrl_step_result tfrl_env_step_tetris(tfrl_env *env, tfrl_action action) {
    int act = action.index;
    if (act < 0 || act > 3) act = 0;
    int nx = env->tetris.x;
    int ny = env->tetris.y;
    if (act == 0) nx -= 1;
    if (act == 1) nx += 1;
    if (act == 2) ny += 1;
    if (act == 3) ny += 2;
    if (tetris_can_place(env, nx, ny)) {
        env->tetris.x = nx;
        env->tetris.y = ny;
    }

    double reward = -0.01;
    int done = 0;

    if (!tetris_can_place(env, env->tetris.x, env->tetris.y + 1)) {
        tetris_lock_piece(env);
        int cleared = tetris_clear_lines(env);
        if (cleared > 0) reward += (double)cleared;
        tetris_spawn_piece(env);
        if (!tetris_can_place(env, env->tetris.x, env->tetris.y)) {
            done = 1;
            reward = -1.0;
        }
    } else {
        env->tetris.y += 1;
    }

    env->steps += 1;
    if (env->steps >= TETRIS_MAX_STEPS) done = 1;

    tfrl_step_result out = {0};
    tetris_fill_obs(env, &out.observation);
    out.reward = reward;
    out.done = done;
    env->last_reward = (float)reward;
    env->last_done = done;
    return out;
}

tfrl_step_result tfrl_env_step_tetris_disc(tfrl_env *env, tfrl_action action) {
    tfrl_step_result out = tfrl_env_step_tetris(env, action);
    out.observation.index = tetris_hash_obs(env);
    out.observation.data_len = 0;
    return out;
}
