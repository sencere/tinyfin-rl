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
        .render_bytes = tfrl_env_render_bytes_maze,
        .render_write = tfrl_env_render_write_maze,
    },
    {
        .name = "lineworld",
        .kind = TFRL_ENV_LINEWORLD,
        .agent_count = 1,
        .spec = tfrl_env_spec_lineworld,
        .reset = tfrl_env_reset_lineworld,
        .step = tfrl_env_step_lineworld,
        .step_multi = NULL,
        .render_bytes = tfrl_env_render_bytes_lineworld,
        .render_write = tfrl_env_render_write_lineworld,
    },
    {
        .name = "lineworld_cont",
        .kind = TFRL_ENV_LINEWORLD_CONT,
        .agent_count = 1,
        .spec = tfrl_env_spec_lineworld_cont,
        .reset = tfrl_env_reset_lineworld_cont,
        .step = tfrl_env_step_lineworld_cont,
        .step_multi = NULL,
        .render_bytes = tfrl_env_render_bytes_lineworld,
        .render_write = tfrl_env_render_write_lineworld,
    },
    {
        .name = "lineworld_duo",
        .kind = TFRL_ENV_LINEWORLD_DUO,
        .agent_count = LINEWORLD_DUO_AGENTS,
        .spec = tfrl_env_spec_lineworld_duo,
        .reset = tfrl_env_reset_lineworld_duo,
        .step = tfrl_env_step_lineworld_duo,
        .step_multi = tfrl_env_step_multi_lineworld_duo,
        .render_bytes = tfrl_env_render_bytes_lineworld,
        .render_write = tfrl_env_render_write_lineworld,
    },
    {
        .name = "point1d",
        .kind = TFRL_ENV_POINT1D,
        .agent_count = 1,
        .spec = tfrl_env_spec_point1d,
        .reset = tfrl_env_reset_point1d,
        .step = tfrl_env_step_point1d,
        .step_multi = NULL,
        .render_bytes = tfrl_env_render_bytes_point1d,
        .render_write = tfrl_env_render_write_point1d,
    },
    {
        .name = "coin_maze",
        .kind = TFRL_ENV_COIN_MAZE,
        .agent_count = 1,
        .spec = tfrl_env_spec_coin_maze,
        .reset = tfrl_env_reset_coin_maze,
        .step = tfrl_env_step_coin_maze,
        .step_multi = NULL,
        .render_bytes = tfrl_env_render_bytes_coin_maze,
        .render_write = tfrl_env_render_write_coin_maze,
    },
    {
        .name = "coin_maze_duo",
        .kind = TFRL_ENV_COIN_MAZE_DUO,
        .agent_count = LINEWORLD_DUO_AGENTS,
        .spec = tfrl_env_spec_coin_maze_duo,
        .reset = tfrl_env_reset_coin_maze_duo,
        .step = tfrl_env_step_coin_maze_duo,
        .step_multi = tfrl_env_step_multi_coin_maze_duo,
        .render_bytes = tfrl_env_render_bytes_coin_maze,
        .render_write = tfrl_env_render_write_coin_maze,
    },
    {
        .name = "snake",
        .kind = TFRL_ENV_SNAKE,
        .agent_count = 1,
        .spec = tfrl_env_spec_snake,
        .reset = tfrl_env_reset_snake,
        .step = tfrl_env_step_snake,
        .step_multi = NULL,
        .render_bytes = tfrl_env_render_bytes_snake,
        .render_write = tfrl_env_render_write_snake,
    },
    {
        .name = "floppy",
        .kind = TFRL_ENV_FLOPPY,
        .agent_count = 1,
        .spec = tfrl_env_spec_floppy,
        .reset = tfrl_env_reset_floppy,
        .step = tfrl_env_step_floppy,
        .step_multi = NULL,
        .render_bytes = tfrl_env_render_bytes_floppy,
        .render_write = tfrl_env_render_write_floppy,
    },
    {
        .name = "tetris",
        .kind = TFRL_ENV_TETRIS,
        .agent_count = 1,
        .spec = tfrl_env_spec_tetris,
        .reset = tfrl_env_reset_tetris,
        .step = tfrl_env_step_tetris,
        .step_multi = NULL,
        .render_bytes = tfrl_env_render_bytes_tetris,
        .render_write = tfrl_env_render_write_tetris,
    },
    {
        .name = "tetris_disc",
        .kind = TFRL_ENV_TETRIS,
        .agent_count = 1,
        .spec = tfrl_env_spec_tetris_disc,
        .reset = tfrl_env_reset_tetris_disc,
        .step = tfrl_env_step_tetris_disc,
        .step_multi = NULL,
        .render_bytes = tfrl_env_render_bytes_tetris,
        .render_write = tfrl_env_render_write_tetris,
    },
    {
        .name = "pang",
        .kind = TFRL_ENV_PANG,
        .agent_count = 1,
        .spec = tfrl_env_spec_pang,
        .reset = tfrl_env_reset_pang,
        .step = tfrl_env_step_pang,
        .step_multi = NULL,
        .render_bytes = tfrl_env_render_bytes_pang,
        .render_write = tfrl_env_render_write_pang,
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

size_t tfrl_env_count(void) {
    return sizeof(ENVS) / sizeof(ENVS[0]);
}

const tfrl_env_ops *tfrl_env_get_ops(size_t index) {
    size_t count = tfrl_env_count();
    if (index >= count) return NULL;
    return &ENVS[index];
}
