#include "envs_internal.h"

#include <string.h>

static const tfrl_env_ops ENVS[] = {
    {
        .name = "maze_rooms",
        .kind = TFRL_ENV_MAZE,
        .agent_count = 1,
        .spec = tfrl_env_spec_maze,
        .reset = tfrl_env_reset_maze,
        .step = tfrl_env_step_maze,
        .step_multi = NULL,
    },
    {
        .name = "lineworld",
        .kind = TFRL_ENV_LINEWORLD,
        .agent_count = 1,
        .spec = tfrl_env_spec_lineworld,
        .reset = tfrl_env_reset_lineworld,
        .step = tfrl_env_step_lineworld,
        .step_multi = NULL,
    },
    {
        .name = "lineworld_cont",
        .kind = TFRL_ENV_LINEWORLD_CONT,
        .agent_count = 1,
        .spec = tfrl_env_spec_lineworld_cont,
        .reset = tfrl_env_reset_lineworld_cont,
        .step = tfrl_env_step_lineworld_cont,
        .step_multi = NULL,
    },
    {
        .name = "lineworld_duo",
        .kind = TFRL_ENV_LINEWORLD_DUO,
        .agent_count = LINEWORLD_DUO_AGENTS,
        .spec = tfrl_env_spec_lineworld_duo,
        .reset = tfrl_env_reset_lineworld_duo,
        .step = tfrl_env_step_lineworld_duo,
        .step_multi = tfrl_env_step_multi_lineworld_duo,
    },
    {
        .name = "point1d",
        .kind = TFRL_ENV_POINT1D,
        .agent_count = 1,
        .spec = tfrl_env_spec_point1d,
        .reset = tfrl_env_reset_point1d,
        .step = tfrl_env_step_point1d,
        .step_multi = NULL,
    },
    {
        .name = "coin_maze",
        .kind = TFRL_ENV_COIN_MAZE,
        .agent_count = 1,
        .spec = tfrl_env_spec_coin_maze,
        .reset = tfrl_env_reset_coin_maze,
        .step = tfrl_env_step_coin_maze,
        .step_multi = NULL,
    },
    {
        .name = "coin_maze_duo",
        .kind = TFRL_ENV_COIN_MAZE_DUO,
        .agent_count = LINEWORLD_DUO_AGENTS,
        .spec = tfrl_env_spec_coin_maze_duo,
        .reset = tfrl_env_reset_coin_maze_duo,
        .step = tfrl_env_step_coin_maze_duo,
        .step_multi = tfrl_env_step_multi_coin_maze_duo,
    },
};

const tfrl_env_ops *tfrl_env_find_ops(const char *name) {
    if (!name || !name[0]) return &ENVS[0];
    for (size_t i = 0; i < sizeof(ENVS) / sizeof(ENVS[0]); i++) {
        if (strcmp(ENVS[i].name, name) == 0) {
            return &ENVS[i];
        }
    }
    return &ENVS[0];
}
