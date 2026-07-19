#include <stdio.h>

#include "core/env_api.h"

static int expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "test_lineworld_duo_smoke: %s\n", msg);
        return 0;
    }
    return 1;
}

int main(void) {
    tfrl_env_config cfg = {.name = "lineworld_duo", .seed = 0};
    tfrl_env *env = tfrl_env_create(&cfg);
    if (!expect(env != NULL, "create")) return 1;

    const tfrl_env_spec *spec = tfrl_env_get_spec(env);
    if (!expect(spec != NULL, "spec")) return 1;
    if (!expect(spec->agent_count == 2, "agent_count")) return 1;

    tfrl_obs obs[2] = {0};
    int got = tfrl_env_reset_multi(env, 0, obs, 2);
    if (!expect(got == 2, "reset_multi")) return 1;
    if (!expect(obs[0].index == 0, "agent0 start")) return 1;
    if (!expect(obs[1].index == 0, "agent1 start")) return 1;

    tfrl_action actions[2] = {{.index = 1}, {.index = 1}};
    tfrl_step_result steps[2] = {0};
    int stepped = tfrl_env_step_multi(env, actions, 2, steps, 2);
    if (!expect(stepped == 2, "step_multi")) return 1;
    if (!expect(steps[0].observation.index == 1, "agent0 step")) return 1;
    if (!expect(steps[1].observation.index == 1, "agent1 step")) return 1;

    tfrl_env_destroy(env);
    return 0;
}
