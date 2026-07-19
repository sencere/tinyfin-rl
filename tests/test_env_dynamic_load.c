#include "tfrl/env.h"

#include <stdio.h>
#include <string.h>

static int fail(const char *msg) {
    fprintf(stderr, "test_env_dynamic_load: %s\n", msg);
    return 1;
}

static int check_path(const char *path, const char *expected_name) {
    tfrl_env_config cfg = {.name = path, .seed = 11};
    tfrl_env *env = tfrl_env_create(&cfg);
    if (!env) return fail("could not create env from shared library path");

    const tfrl_env_spec *spec = tfrl_env_get_spec(env);
    if (!spec || strcmp(spec->name, expected_name) != 0) {
        tfrl_env_destroy(env);
        return fail("dynamic env returned wrong spec");
    }

    tfrl_obs obs = tfrl_env_reset(env, 11);
    if (obs.data_len < 0 || obs.data_len > TFRL_MAX_BOX_DIMS) {
        tfrl_env_destroy(env);
        return fail("dynamic env reset returned invalid observation");
    }

    tfrl_action action = {0};
    tfrl_step_result step = tfrl_env_step(env, action);
    if (step.observation.data_len < 0 || step.observation.data_len > TFRL_MAX_BOX_DIMS) {
        tfrl_env_destroy(env);
        return fail("dynamic env step returned invalid observation");
    }

    tfrl_env_destroy(env);
    return 0;
}

int main(void) {
    if (check_path("build/libtfrl_env_lineworld.so", "lineworld")) return 1;
    if (check_path("build/libtfrl_env_maze_rooms.so", "maze_rooms")) return 1;
    if (check_path("build/libtfrl_env_coin_maze.so", "coin_maze")) return 1;
    if (check_path("envs/lineworld/env.toml", "lineworld")) return 1;
    if (check_path("envs/maze_rooms/env.toml", "maze_rooms")) return 1;
    if (check_path("envs/coin_maze/env.toml", "coin_maze")) return 1;
    if (check_path("lineworld", "lineworld")) return 1;
    if (check_path("maze_rooms", "maze_rooms")) return 1;
    if (check_path("coin_maze", "coin_maze")) return 1;
    return 0;
}
