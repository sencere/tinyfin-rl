#ifndef TFRL_ENVS_INTERNAL_H
#define TFRL_ENVS_INTERNAL_H

#include "core/env_api.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef enum {
    TFRL_ENV_MAZE = 0,
    TFRL_ENV_LINEWORLD = 1,
    TFRL_ENV_POINT1D = 2,
    TFRL_ENV_LINEWORLD_CONT = 3,
    TFRL_ENV_COIN_MAZE = 4,
    TFRL_ENV_LINEWORLD_DUO = 5,
    TFRL_ENV_COIN_MAZE_DUO = 6,
    TFRL_ENV_PYBRIDGE = 7,
    TFRL_ENV_SNAKE = 8,
    TFRL_ENV_FLOPPY = 9,
    TFRL_ENV_TETRIS = 10,
    TFRL_ENV_PANG = 11
} tfrl_env_kind;

#define MAZE_W 10
#define MAZE_H 10
#define MAZE_MAX_STEPS 200

#define LINEWORLD_W 7
#define LINEWORLD_H 1
#define LINEWORLD_MAX_STEPS 50
#define LINEWORLD_DUO_AGENTS 2

#define POINT1D_W 21
#define POINT1D_H 1
#define POINT1D_MAX_STEPS 100
#define POINT1D_STEP_SCALE 0.1f

#define COIN_MAZE_W 10
#define COIN_MAZE_H 10
#define COIN_MAZE_MAX_STEPS 200
#define COIN_MAZE_COINS 4

#define SNAKE_W 25
#define SNAKE_H 14
#define SNAKE_MAX_STEPS 500
#define SNAKE_MAX_LEN 256

#define FLOPPY_W 800
#define FLOPPY_H 450
#define FLOPPY_MAX_STEPS 10000
#define FLOPPY_MAX_TUBES 100
#define FLOPPY_TUBES_WIDTH 80
#define FLOPPY_RADIUS 24

#define TETRIS_W 10
#define TETRIS_H 20
#define TETRIS_MAX_STEPS 2000

#define PANG_W 800
#define PANG_H 450
#define PANG_MAX_STEPS 2000
#define PANG_MAX_BIG 2
#define PANG_MAX_MED (PANG_MAX_BIG * 2)
#define PANG_MAX_SMALL (PANG_MAX_BIG * 4)

typedef struct {
    int len;
    int dir;
    int xs[SNAKE_MAX_LEN];
    int ys[SNAKE_MAX_LEN];
    int fruit_x;
    int fruit_y;
} tfrl_snake_state;

typedef struct {
    float x;
    float y;
    int score;
    int last_action;
    float tubes_x[FLOPPY_MAX_TUBES];
    float tubes_offset_y[FLOPPY_MAX_TUBES];
    int tubes_scored[FLOPPY_MAX_TUBES];
} tfrl_floppy_state;

typedef struct {
    int grid[TETRIS_W][TETRIS_H];
    int piece;
    int next_piece;
    int rot;
    int x;
    int y;
    int lines;
    double score;
} tfrl_tetris_state;

typedef struct {
    float x;
    float y;
    float vx;
    float vy;
    float radius;
    int active;
} tfrl_pang_ball;

typedef struct {
    float player_x;
    float player_y;
    float ship_height;
    float gravity;
    int shot_active;
    float shot_x;
    float shot_y;
    float line_x;
    tfrl_pang_ball big[PANG_MAX_BIG];
    tfrl_pang_ball med[PANG_MAX_MED];
    tfrl_pang_ball small[PANG_MAX_SMALL];
} tfrl_pang_state;

typedef struct tfrl_env_ops {
    const char *name;
    tfrl_env_kind kind;
    int agent_count;
    const tfrl_env_spec *(*spec)(void);
    tfrl_obs (*reset)(struct tfrl_env *env, uint64_t seed);
    tfrl_step_result (*step)(struct tfrl_env *env, tfrl_action action);
    int (*step_multi)(struct tfrl_env *env, const tfrl_action *actions, int action_count,
                      tfrl_step_result *out_steps, int max_agents);
    size_t (*render_bytes)(const struct tfrl_env *env);
    size_t (*render_write)(struct tfrl_env *env, void *buffer, size_t buffer_len);
} tfrl_env_ops;

struct tfrl_env {
    tfrl_env_kind kind;
    const tfrl_env_ops *ops;
    uint32_t rng_state;
    int x;
    int y;
    int x2;
    int y2;
    float pos;
    int steps;
    int coins_x[COIN_MAZE_COINS];
    int coins_y[COIN_MAZE_COINS];
    int coins_collected[COIN_MAZE_COINS];
    float last_reward;
    int last_done;
    uint32_t frame_index;
    int py_fd;
    FILE *py_fp;
    tfrl_env_spec py_spec;
    int py_agent_count;
    char py_name[128];
    tfrl_snake_state snake;
    tfrl_floppy_state floppy;
    tfrl_tetris_state tetris;
    tfrl_pang_state pang;
};

typedef struct tfrl_env tfrl_env;

static inline uint32_t tfrl_env_rng_next(tfrl_env *env) {
    env->rng_state = env->rng_state * 1664525u + 1013904223u;
    return env->rng_state;
}

static inline float tfrl_env_rand_uniform(tfrl_env *env) {
    return (tfrl_env_rng_next(env) & 0x00FFFFFFu) / 16777215.0f;
}

static inline int tfrl_env_rand_range_int(tfrl_env *env, int min, int max) {
    if (max <= min) return min;
    int span = max - min + 1;
    return min + (int)(tfrl_env_rng_next(env) % (uint32_t)span);
}

static inline float tfrl_env_rand_range_float(tfrl_env *env, float min, float max) {
    float r = tfrl_env_rand_uniform(env);
    return min + (max - min) * r;
}

void tfrl_env_reset_state(tfrl_env *env);
int tfrl_env_is_wall_maze(int x, int y);

const tfrl_env_ops *tfrl_env_find_ops(const char *name);

const tfrl_env_spec *tfrl_env_spec_maze(void);
const tfrl_env_spec *tfrl_env_spec_lineworld(void);
const tfrl_env_spec *tfrl_env_spec_lineworld_cont(void);
const tfrl_env_spec *tfrl_env_spec_lineworld_duo(void);
const tfrl_env_spec *tfrl_env_spec_point1d(void);
const tfrl_env_spec *tfrl_env_spec_coin_maze(void);
const tfrl_env_spec *tfrl_env_spec_coin_maze_duo(void);
const tfrl_env_spec *tfrl_env_spec_snake(void);
const tfrl_env_spec *tfrl_env_spec_floppy(void);
const tfrl_env_spec *tfrl_env_spec_tetris(void);
const tfrl_env_spec *tfrl_env_spec_tetris_disc(void);
const tfrl_env_spec *tfrl_env_spec_pang(void);

tfrl_obs tfrl_env_reset_maze(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_lineworld(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_lineworld_cont(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_lineworld_duo(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_point1d(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_coin_maze(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_coin_maze_duo(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_snake(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_floppy(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_tetris(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_tetris_disc(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_pang(tfrl_env *env, uint64_t seed);

tfrl_step_result tfrl_env_step_maze(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_lineworld(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_lineworld_cont(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_lineworld_duo(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_point1d(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_coin_maze(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_coin_maze_duo(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_snake(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_floppy(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_tetris(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_tetris_disc(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_pang(tfrl_env *env, tfrl_action action);

int tfrl_env_step_multi_lineworld_duo(tfrl_env *env, const tfrl_action *actions, int action_count,
                                     tfrl_step_result *out_steps, int max_agents);
int tfrl_env_step_multi_coin_maze_duo(tfrl_env *env, const tfrl_action *actions, int action_count,
                                      tfrl_step_result *out_steps, int max_agents);

int tfrl_env_py_connect(tfrl_env *env, const char *socket_path, const char *kind, const char *env_id, uint64_t seed);
void tfrl_env_py_close(tfrl_env *env);
int tfrl_env_py_reset_multi(tfrl_env *env, uint64_t seed, tfrl_obs *out_obs, int max_agents);
int tfrl_env_py_step_multi(tfrl_env *env, const tfrl_action *actions, int action_count,
                           tfrl_step_result *out_steps, int max_agents);

size_t tfrl_env_render_bytes_maze(const tfrl_env *env);
size_t tfrl_env_render_bytes_lineworld(const tfrl_env *env);
size_t tfrl_env_render_bytes_point1d(const tfrl_env *env);
size_t tfrl_env_render_bytes_coin_maze(const tfrl_env *env);
size_t tfrl_env_render_bytes_tetris(const tfrl_env *env);
size_t tfrl_env_render_bytes_snake(const tfrl_env *env);
size_t tfrl_env_render_bytes_floppy(const tfrl_env *env);
size_t tfrl_env_render_bytes_pang(const tfrl_env *env);

size_t tfrl_env_render_write_maze(tfrl_env *env, void *buffer, size_t buffer_len);
size_t tfrl_env_render_write_lineworld(tfrl_env *env, void *buffer, size_t buffer_len);
size_t tfrl_env_render_write_point1d(tfrl_env *env, void *buffer, size_t buffer_len);
size_t tfrl_env_render_write_coin_maze(tfrl_env *env, void *buffer, size_t buffer_len);
size_t tfrl_env_render_write_tetris(tfrl_env *env, void *buffer, size_t buffer_len);
size_t tfrl_env_render_write_snake(tfrl_env *env, void *buffer, size_t buffer_len);
size_t tfrl_env_render_write_floppy(tfrl_env *env, void *buffer, size_t buffer_len);
size_t tfrl_env_render_write_pang(tfrl_env *env, void *buffer, size_t buffer_len);

#endif
