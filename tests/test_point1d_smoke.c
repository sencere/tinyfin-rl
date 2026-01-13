#include <math.h>
#include <stdio.h>

#include "core/env_api.h"

static int expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "test_point1d_smoke: %s\n", msg);
        return 0;
    }
    return 1;
}

int main(void) {
    tfrl_env_config cfg = {.name = "point1d", .seed = 0};
    tfrl_env *env = tfrl_env_create(&cfg);
    if (!expect(env != NULL, "create")) return 1;

    const tfrl_env_spec *spec = tfrl_env_get_spec(env);
    if (!expect(spec != NULL, "spec")) return 1;
    if (!expect(spec->obs_type == TFRL_SPACE_BOX, "obs type")) return 1;
    if (!expect(spec->action_type == TFRL_SPACE_BOX, "action type")) return 1;

    tfrl_obs obs = tfrl_env_reset(env, 0);
    if (!expect(obs.data_len == 1, "reset obs len")) return 1;
    if (!expect(obs.data[0] >= -1.0f && obs.data[0] <= 1.0f, "reset obs range")) return 1;

    tfrl_action action = {0};
    action.data_len = 1;
    action.data[0] = 1.0f;

    for (int i = 0; i < 10; i++) {
        tfrl_step_result step = tfrl_env_step(env, action);
        if (!expect(step.observation.data_len == 1, "step obs len")) return 1;
        if (!expect(fabsf(step.observation.data[0]) <= 1.0f, "step obs range")) return 1;
    }

    tfrl_env_destroy(env);
    return 0;
}
