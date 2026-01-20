#ifndef TFRL_CLI_H
#define TFRL_CLI_H

#ifdef __cplusplus
extern "C" {
#endif

#include "runner/runner.h"

int tfrl_cli_parse(int argc, char **argv, tfrl_runner_config *out_cfg);
void tfrl_cli_print_usage(const char *name);
void tfrl_cli_print_envs(void);

#ifdef __cplusplus
}
#endif

#endif
