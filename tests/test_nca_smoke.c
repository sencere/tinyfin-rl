#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "envs/envs_internal.h"
#include "tfrl/nca.h"

static int expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        return 0;
    }
    return 1;
}

int main(void) {
    tfrl_env_config env_cfg = {.name = "lineworld", .seed = 1};
    tfrl_env *env = tfrl_env_create(&env_cfg);
    if (!expect(env != NULL, "create env")) return 1;

    const tfrl_env_spec *spec = tfrl_env_get_spec(env);
    if (!expect(spec != NULL, "env spec")) return 1;

    tfrl_algo_config cfg = {0};
    cfg.name = "nca";
    cfg.obs_n = spec->obs_n;
    cfg.action_n = spec->action_n;
    cfg.obs_type = spec->obs_type;
    cfg.action_type = spec->action_type;
    cfg.obs_dims = spec->obs_dims;
    cfg.action_dims = spec->action_dims;
    cfg.obs_low = spec->obs_low;
    cfg.obs_high = spec->obs_high;
    cfg.action_low = spec->action_low;
    cfg.action_high = spec->action_high;
    cfg.gamma = 0.95f;
    cfg.lr = 0.05f;
    cfg.epsilon = 0.2f;
    cfg.seed = 7;

    tfrl_algo algo = tfrl_algo_nca_create(&cfg);
    if (!expect(algo.ctx != NULL, "create nca")) return 1;

    tfrl_obs obs = tfrl_env_reset(env, 1);
    for (int i = 0; i < 80; i++) {
        tfrl_action action = tfrl_algo_act(&algo, env, obs);
        if (!expect(action.index >= 0 && action.index < spec->action_n, "action range")) return 1;
        tfrl_step_result step = tfrl_env_step(env, action);
        tfrl_transition tr = {
            .obs = obs,
            .action = action,
            .reward = step.reward,
            .next_obs = step.observation,
            .done = step.done,
        };
        algo.vtable->update(algo.ctx, &tr);
        obs = step.done ? tfrl_env_reset(env, (uint64_t)(i + 2)) : step.observation;
    }

    char diag[128];
    if (!expect(algo.vtable->diagnostics(algo.ctx, diag, sizeof(diag)), "diagnostics")) return 1;
    if (!expect(strstr(diag, "nca") != NULL, "diagnostics include nca")) return 1;
    if (!expect(strstr(diag, "updates=80") != NULL, "diagnostics include updates")) return 1;

    const char *path = "/tmp/tfrl_nca_smoke.policy";
    algo.vtable->save(algo.ctx, path);
    algo.vtable->destroy(algo.ctx);

    cfg.load_path = path;
    cfg.deterministic = 1;
    tfrl_algo loaded = tfrl_algo_nca_create(&cfg);
    if (!expect(loaded.ctx != NULL, "load nca")) return 1;
    tfrl_action loaded_action = tfrl_algo_act(&loaded, env, tfrl_env_reset(env, 1));
    if (!expect(loaded_action.index >= 0 && loaded_action.index < spec->action_n, "loaded action range")) return 1;

    loaded.vtable->destroy(loaded.ctx);
    tfrl_env_destroy(env);
    remove(path);
    return 0;
}
