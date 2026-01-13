#include <stdlib.h>
#include <time.h>

#include "tinyfin_rl/rl.h"
#include "tinyfin_rl/rl_plugin.h"
#include "tinyfin_rl/rl_render.h"

#ifdef __cplusplus
extern "C" {
#endif
const tfrl_env_plugin *tfrl_env_plugin_get(void);
#ifdef __cplusplus
}
#endif

#define ACTIONS 4

int main(void) {
    const tfrl_env_plugin *plugin = tfrl_env_plugin_get();
    if (!plugin || !plugin->create) return 1;
    tfrl_env env = plugin->create();
    if (!tfrl_renderer_init(12, 48, "cliffwalk viewer")) {
        plugin->destroy(&env);
        return 1;
    }

    srand((unsigned)time(NULL));
    tfrl_value obs = {0};
    tfrl_step step = {0};
    if (tfrl_env_reset(&env, &obs) != TFRL_OK) return 1;

    double reward = 0.0;
    int done = 0;
    while (!tfrl_renderer_should_close()) {
        tfrl_renderer_draw(&obs, reward, done);
        int action = rand() % ACTIONS;
        tfrl_value act = {.type = TFRL_VALUE_I64, .as.i64 = action};
        if (tfrl_env_step(&env, &act, &step) != TFRL_OK) break;
        obs = step.observation;
        reward = step.reward;
        done = tfrl_step_done(&step);
        if (done) {
            tfrl_env_reset(&env, &obs);
            reward = 0.0;
            done = 0;
        }
    }
    tfrl_renderer_close();
    if (plugin->destroy) plugin->destroy(&env);
    return 0;
}
