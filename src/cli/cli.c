#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli.h"

static int arg_int(int argc, char **argv, const char *name, int def) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0) return atoi(argv[i + 1]);
    }
    return def;
}

static const char *arg_str(int argc, char **argv, const char *name, const char *def) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return def;
}

static float arg_float(int argc, char **argv, const char *name, float def) {
    const char *v = arg_str(argc, argv, name, NULL);
    return v ? (float)atof(v) : def;
}

void tfrl_cli_print_usage(const char *name) {
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "  %s train --algo dqn [--env NAME] [--envs N] [--steps N] [--seed N] [--gamma G] [--lr LR] [--epsilon E]\n", name);
    fprintf(stderr, "           [--log-every N] [--render off|live] [--render-every N] [--render-fps N] [--trace-out FILE]\n");
    fprintf(stderr, "  %s eval --algo dqn [--env NAME] [--episodes N] [--seed N] [--gamma G] [--lr LR] [--epsilon E]\n", name);
    fprintf(stderr, "          [--log-every N] [--render off|live] [--render-every N] [--render-fps N] [--trace-out FILE]\n");
    fprintf(stderr, "  %s replay --trace-in FILE [--render-fps N]\n", name);
    fprintf(stderr, "algo params:\n");
    fprintf(stderr, "  --clip-eps N --steps-per-batch N --epochs N (ppo)\n");
    fprintf(stderr, "checkpoint params:\n");
    fprintf(stderr, "  --save PATH --load PATH\n");
}

int tfrl_cli_parse(int argc, char **argv, tfrl_runner_config *out_cfg) {
    if (!out_cfg || argc < 2) return 0;
    memset(out_cfg, 0, sizeof(*out_cfg));

    const char *mode = argv[1];
    if (strcmp(mode, "train") == 0) {
        out_cfg->mode = TFRL_MODE_TRAIN;
    } else if (strcmp(mode, "eval") == 0) {
        out_cfg->mode = TFRL_MODE_EVAL;
    } else if (strcmp(mode, "replay") == 0) {
        out_cfg->mode = TFRL_MODE_REPLAY;
    } else {
        return 0;
    }

    out_cfg->algo = arg_str(argc, argv, "--algo", "dqn");
    out_cfg->env_name = arg_str(argc, argv, "--env", "maze_rooms");
    out_cfg->envs = arg_int(argc, argv, "--envs", 1);
    out_cfg->steps = arg_int(argc, argv, "--steps", 1000);
    out_cfg->episodes = arg_int(argc, argv, "--episodes", 10);
    out_cfg->seed = arg_int(argc, argv, "--seed", 0);
    out_cfg->gamma = arg_float(argc, argv, "--gamma", 0.99f);
    out_cfg->lr = arg_float(argc, argv, "--lr", 0.05f);
    out_cfg->epsilon = arg_float(argc, argv, "--epsilon", 0.1f);
    out_cfg->clip_eps = arg_float(argc, argv, "--clip-eps", 0.2f);
    out_cfg->steps_per_batch = arg_int(argc, argv, "--steps-per-batch", 64);
    out_cfg->epochs = arg_int(argc, argv, "--epochs", 2);
    out_cfg->log_every = arg_int(argc, argv, "--log-every", 100);
    out_cfg->render_every = arg_int(argc, argv, "--render-every", 1);
    out_cfg->render_fps = arg_int(argc, argv, "--render-fps", 60);
    out_cfg->trace_out = arg_str(argc, argv, "--trace-out", NULL);
    out_cfg->trace_in = arg_str(argc, argv, "--trace-in", NULL);
    out_cfg->save_path = arg_str(argc, argv, "--save", NULL);
    out_cfg->load_path = arg_str(argc, argv, "--load", NULL);

    const char *render = arg_str(argc, argv, "--render", "off");
    out_cfg->render = (strcmp(render, "live") == 0);

    if (out_cfg->mode == TFRL_MODE_REPLAY && !out_cfg->trace_in) {
        return 0;
    }
    return 1;
}
