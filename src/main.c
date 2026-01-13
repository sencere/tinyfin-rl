#include <stdio.h>

#include "cli/cli.h"
#include "runner/runner.h"

int main(int argc, char **argv) {
    tfrl_runner_config cfg;
    if (!tfrl_cli_parse(argc, argv, &cfg)) {
        tfrl_cli_print_usage(argv[0]);
        return 1;
    }
    return tfrl_runner_run(&cfg);
}
