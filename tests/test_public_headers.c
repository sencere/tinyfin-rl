#include "tfrl/tfrl.h"
#include "tfrl/nca.h"

int main(void) {
    tfrl_env_config env_cfg = {.name = "lineworld", .seed = 1};
    tfrl_algo_config algo_cfg = {.name = "random"};
    tfrl_runner_config runner_cfg = {.mode = TFRL_MODE_TRAIN};
    tfrl_metrics metrics = {0};
    tfrl_checkpoint_manifest manifest = {0};
    tfrl_render_snapshot_header header = {0};
    tfrl_nca_policy_desc nca = {0};

    (void)env_cfg;
    (void)algo_cfg;
    (void)runner_cfg;
    (void)metrics;
    (void)manifest;
    (void)header;
    (void)nca;
    return TFRL_ENV_ABI_VERSION == 2u ? 0 : 1;
}
