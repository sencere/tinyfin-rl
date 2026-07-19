#include <stdio.h>

#include "core/env_api.h"

static int expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "test_maze_rooms_smoke: %s\n", msg);
        return 0;
    }
    return 1;
}

static int obs_index(int width, int x, int y) {
    return y * width + x;
}

int main(void) {
    tfrl_env_config cfg = {.name = "maze_rooms", .seed = 0};
    tfrl_env *env = tfrl_env_create(&cfg);
    if (!expect(env != NULL, "create")) return 1;

    const tfrl_env_spec *spec = tfrl_env_get_spec(env);
    if (!expect(spec != NULL, "spec")) return 1;
    if (!expect(spec->obs_type == TFRL_SPACE_DISCRETE, "obs type")) return 1;
    if (!expect(spec->action_type == TFRL_SPACE_DISCRETE, "action type")) return 1;
    if (!expect(spec->width == 8 && spec->height == 8, "shape")) return 1;

    tfrl_obs obs = tfrl_env_reset(env, 0);
    if (!expect(obs.index == obs_index(spec->width, 1, 1), "start position")) return 1;

    tfrl_action invalid = {.index = 99};
    tfrl_step_result blocked = tfrl_env_step(env, invalid);
    if (!expect(blocked.observation.index == obs_index(spec->width, 1, 1), "invalid action clamps into wall")) return 1;
    if (!expect(!blocked.done, "invalid action should not terminate")) return 1;

    tfrl_env_reset(env, 0);
    const int path[] = {
        3, 3, 3, 3, 3,
        1, 1, 1, 1, 1,
    };
    tfrl_step_result step = {0};
    for (size_t i = 0; i < sizeof(path) / sizeof(path[0]); i++) {
        tfrl_action action = {.index = path[i]};
        step = tfrl_env_step(env, action);
    }
    if (!expect(step.observation.index == obs_index(spec->width, spec->width - 2, spec->height - 2), "goal position")) return 1;
    if (!expect(step.done, "goal terminates")) return 1;
    if (!expect(step.reward > 0.0, "goal reward")) return 1;

    tfrl_env_destroy(env);
    return 0;
}
