#include <stdio.h>
#include <string.h>

#include "cli/cli.h"
#include "runner/runner.h"

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "list-envs") == 0) {
        tfrl_cli_print_envs();
        return 0;
    }
    tfrl_runner_config cfg;
    if (!tfrl_cli_parse(argc, argv, &cfg)) {
        tfrl_cli_print_usage(argv[0]);
        return 1;
    }
    return tfrl_runner_run(&cfg);
}
