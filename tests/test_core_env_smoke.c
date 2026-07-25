#include <stdio.h>

#include "core/env_api.h"

static int check(int ok, const char *message) {
    if (!ok) fprintf(stderr, "test_core_env_smoke: %s\n", message);
    return ok;
}

static int test_lineworld(void) {
    tfrl_env_config cfg = {.name = "lineworld", .seed = 7};
    tfrl_env *env = tfrl_env_create(&cfg);
    if (!check(env != NULL, "lineworld create")) return 0;
    tfrl_obs obs = tfrl_env_reset(env, 7);
    if (!check(obs.index == 0, "lineworld reset")) return 0;
    tfrl_step_result step = {0};
    for (int i = 0; i < 8; i++) step = tfrl_env_step(env, (tfrl_action){.index = 1});
    int ok = check(step.done, "lineworld reaches terminal state") &&
             check(step.reward > 0.0, "lineworld terminal reward");
    tfrl_env_destroy(env);
    return ok;
}

static int test_coin_maze(void) {
    tfrl_env_config cfg = {.name = "coin_maze", .seed = 11};
    tfrl_env *env = tfrl_env_create(&cfg);
    if (!check(env != NULL, "coin_maze create")) return 0;
    tfrl_obs obs = tfrl_env_reset(env, 11);
    tfrl_step_result step = tfrl_env_step(env, (tfrl_action){.index = 99});
    int ok = check(step.observation.data_len == obs.data_len, "coin_maze observation shape") &&
             check(!step.done, "coin_maze invalid action does not terminate");
    tfrl_env_destroy(env);
    return ok;
}

int main(void) {
    return test_lineworld() && test_coin_maze() ? 0 : 1;
}
