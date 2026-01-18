#ifndef TFRL_ENVS_INTERNAL_H
#define TFRL_ENVS_INTERNAL_H

#include "core/env_api.h"

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
    TFRL_ENV_PYBRIDGE = 7
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

typedef struct tfrl_env_ops {
    const char *name;
    tfrl_env_kind kind;
    int agent_count;
    const tfrl_env_spec *(*spec)(void);
    tfrl_obs (*reset)(struct tfrl_env *env, uint64_t seed);
    tfrl_step_result (*step)(struct tfrl_env *env, tfrl_action action);
    int (*step_multi)(struct tfrl_env *env, const tfrl_action *actions, int action_count,
                      tfrl_step_result *out_steps, int max_agents);
} tfrl_env_ops;

struct tfrl_env {
    tfrl_env_kind kind;
    const tfrl_env_ops *ops;
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
};

typedef struct tfrl_env tfrl_env;

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

tfrl_obs tfrl_env_reset_maze(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_lineworld(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_lineworld_cont(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_lineworld_duo(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_point1d(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_coin_maze(tfrl_env *env, uint64_t seed);
tfrl_obs tfrl_env_reset_coin_maze_duo(tfrl_env *env, uint64_t seed);

tfrl_step_result tfrl_env_step_maze(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_lineworld(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_lineworld_cont(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_lineworld_duo(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_point1d(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_coin_maze(tfrl_env *env, tfrl_action action);
tfrl_step_result tfrl_env_step_coin_maze_duo(tfrl_env *env, tfrl_action action);

int tfrl_env_step_multi_lineworld_duo(tfrl_env *env, const tfrl_action *actions, int action_count,
                                     tfrl_step_result *out_steps, int max_agents);
int tfrl_env_step_multi_coin_maze_duo(tfrl_env *env, const tfrl_action *actions, int action_count,
                                      tfrl_step_result *out_steps, int max_agents);

int tfrl_env_py_connect(tfrl_env *env, const char *socket_path, const char *kind, const char *env_id, uint64_t seed);
void tfrl_env_py_close(tfrl_env *env);
int tfrl_env_py_reset_multi(tfrl_env *env, uint64_t seed, tfrl_obs *out_obs, int max_agents);
int tfrl_env_py_step_multi(tfrl_env *env, const tfrl_action *actions, int action_count,
                           tfrl_step_result *out_steps, int max_agents);

#endif
