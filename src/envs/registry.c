#include "envs_internal.h"

#include <string.h>

static const tfrl_env_ops ENVS[] = {
#if defined(TFRL_ENV_MODULE_MAZE_ROOMS)
    {"maze_rooms", TFRL_ENV_MAZE, 1, tfrl_env_spec_maze, tfrl_env_reset_maze, tfrl_env_step_maze, NULL, tfrl_env_render_bytes_maze, tfrl_env_render_write_maze},
#elif defined(TFRL_ENV_MODULE_LINEWORLD)
    {"lineworld", TFRL_ENV_LINEWORLD, 1, tfrl_env_spec_lineworld, tfrl_env_reset_lineworld, tfrl_env_step_lineworld, NULL, tfrl_env_render_bytes_lineworld, tfrl_env_render_write_lineworld},
#elif defined(TFRL_ENV_MODULE_COIN_MAZE)
    {"coin_maze", TFRL_ENV_COIN_MAZE, 1, tfrl_env_spec_coin_maze, tfrl_env_reset_coin_maze, tfrl_env_step_coin_maze, NULL, tfrl_env_render_bytes_coin_maze, tfrl_env_render_write_coin_maze},
#else
    {"maze_rooms", TFRL_ENV_MAZE, 1, tfrl_env_spec_maze, tfrl_env_reset_maze, tfrl_env_step_maze, NULL, tfrl_env_render_bytes_maze, tfrl_env_render_write_maze},
    {"lineworld", TFRL_ENV_LINEWORLD, 1, tfrl_env_spec_lineworld, tfrl_env_reset_lineworld, tfrl_env_step_lineworld, NULL, tfrl_env_render_bytes_lineworld, tfrl_env_render_write_lineworld},
    {"lineworld_cont", TFRL_ENV_LINEWORLD_CONT, 1, tfrl_env_spec_lineworld_cont, tfrl_env_reset_lineworld_cont, tfrl_env_step_lineworld_cont, NULL, tfrl_env_render_bytes_lineworld, tfrl_env_render_write_lineworld},
    {"lineworld_duo", TFRL_ENV_LINEWORLD_DUO, LINEWORLD_DUO_AGENTS, tfrl_env_spec_lineworld_duo, tfrl_env_reset_lineworld_duo, tfrl_env_step_lineworld_duo, tfrl_env_step_multi_lineworld_duo, tfrl_env_render_bytes_lineworld, tfrl_env_render_write_lineworld},
    {"point1d", TFRL_ENV_POINT1D, 1, tfrl_env_spec_point1d, tfrl_env_reset_point1d, tfrl_env_step_point1d, NULL, tfrl_env_render_bytes_point1d, tfrl_env_render_write_point1d},
    {"coin_maze", TFRL_ENV_COIN_MAZE, 1, tfrl_env_spec_coin_maze, tfrl_env_reset_coin_maze, tfrl_env_step_coin_maze, NULL, tfrl_env_render_bytes_coin_maze, tfrl_env_render_write_coin_maze},
    {"coin_maze_duo", TFRL_ENV_COIN_MAZE_DUO, LINEWORLD_DUO_AGENTS, tfrl_env_spec_coin_maze_duo, tfrl_env_reset_coin_maze_duo, tfrl_env_step_coin_maze_duo, tfrl_env_step_multi_coin_maze_duo, tfrl_env_render_bytes_coin_maze, tfrl_env_render_write_coin_maze},
    {"snake", TFRL_ENV_SNAKE, 1, tfrl_env_spec_snake, tfrl_env_reset_snake, tfrl_env_step_snake, NULL, tfrl_env_render_bytes_snake, tfrl_env_render_write_snake},
    {"floppy", TFRL_ENV_FLOPPY, 1, tfrl_env_spec_floppy, tfrl_env_reset_floppy, tfrl_env_step_floppy, NULL, tfrl_env_render_bytes_floppy, tfrl_env_render_write_floppy},
    {"tetris", TFRL_ENV_TETRIS, 1, tfrl_env_spec_tetris, tfrl_env_reset_tetris, tfrl_env_step_tetris, NULL, tfrl_env_render_bytes_tetris, tfrl_env_render_write_tetris},
    {"tetris_disc", TFRL_ENV_TETRIS, 1, tfrl_env_spec_tetris_disc, tfrl_env_reset_tetris_disc, tfrl_env_step_tetris_disc, NULL, tfrl_env_render_bytes_tetris, tfrl_env_render_write_tetris},
    {"breakout", TFRL_ENV_BREAKOUT, 1, tfrl_env_spec_breakout, tfrl_env_reset_breakout, tfrl_env_step_breakout, NULL, tfrl_env_render_bytes_breakout, tfrl_env_render_write_breakout},
    {"breakout_disc", TFRL_ENV_BREAKOUT, 1, tfrl_env_spec_breakout_disc, tfrl_env_reset_breakout_disc, tfrl_env_step_breakout_disc, NULL, tfrl_env_render_bytes_breakout, tfrl_env_render_write_breakout},
    {"breakout_atari", TFRL_ENV_BREAKOUT, 1, tfrl_env_spec_breakout_atari, tfrl_env_reset_breakout_atari, tfrl_env_step_breakout_atari, NULL, tfrl_env_render_bytes_breakout_atari, tfrl_env_render_write_breakout_atari},
    {"breakout_atari_disc", TFRL_ENV_BREAKOUT, 1, tfrl_env_spec_breakout_atari_disc, tfrl_env_reset_breakout_atari_disc, tfrl_env_step_breakout_atari_disc, NULL, tfrl_env_render_bytes_breakout_atari, tfrl_env_render_write_breakout_atari},
    {"seq_pixels", TFRL_ENV_SEQ_PIXELS, 1, tfrl_env_spec_seq_pixels, tfrl_env_reset_seq_pixels, tfrl_env_step_seq_pixels, NULL, tfrl_env_render_bytes_seq_pixels, tfrl_env_render_write_seq_pixels},
    {"pang", TFRL_ENV_PANG, 1, tfrl_env_spec_pang, tfrl_env_reset_pang, tfrl_env_step_pang, NULL, tfrl_env_render_bytes_pang, tfrl_env_render_write_pang},
#endif
};

const tfrl_env_ops *tfrl_env_find_ops(const char *name) {
    if (!name || !name[0]) return &ENVS[0];
    for (size_t i = 0; i < sizeof(ENVS) / sizeof(ENVS[0]); i++) {
        if (strcmp(ENVS[i].name, name) == 0) return &ENVS[i];
    }
    return NULL;
}

size_t tfrl_env_count(void) {
    return sizeof(ENVS) / sizeof(ENVS[0]);
}

const tfrl_env_ops *tfrl_env_get_ops(size_t index) {
    size_t count = tfrl_env_count();
    if (index >= count) return NULL;
    return &ENVS[index];
}
