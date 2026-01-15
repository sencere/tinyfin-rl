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

static int arg_flag(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0) return 1;
    }
    return 0;
}

static float arg_float(int argc, char **argv, const char *name, float def) {
    const char *v = arg_str(argc, argv, name, NULL);
    return v ? (float)atof(v) : def;
}

void tfrl_cli_print_usage(const char *name) {
    fprintf(stderr, "\033[1;34mTINYFIN\033[0m\n\n");
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "  %s train --algo dqn|rainbow|qrdqn|iqn|ppo|reinforce|a2c|a3c|trpo|sac|td3|impala|mcts|random [--env NAME] [--envs N] [--threads N] [--steps N] [--seed N] [--gamma G] [--lr LR] [--epsilon E] [--deterministic]\n", name);
    fprintf(stderr, "           [--backend cpu|cuda] [--device cpu|gpu] [--log-every N] [--render off|live] [--render-env N] [--render-every N] [--render-fps N] [--trace-out FILE] [--agents N] [--share-policy]\n");
    fprintf(stderr, "           [--profile] [--profile-json FILE]\n");
    fprintf(stderr, "           [--mp-actors N] [--mp-queue N] [--mp-sync-every N] [--mp-sync-path PATH]\n");
    fprintf(stderr, "  %s eval --algo dqn|rainbow|qrdqn|iqn|ppo|reinforce|a2c|a3c|trpo|sac|td3|impala|mcts|random [--env NAME] [--episodes N] [--seed N] [--gamma G] [--lr LR] [--epsilon E] [--deterministic]\n", name);
    fprintf(stderr, "          [--backend cpu|cuda] [--device cpu|gpu] [--log-every N] [--render off|live] [--render-every N] [--render-fps N] [--trace-out FILE] [--agents N] [--share-policy]\n");
    fprintf(stderr, "  %s replay --trace-in FILE [--render-fps N] [--dump-meta-json]\n", name);
    fprintf(stderr, "algo params:\n");
    fprintf(stderr, "  --clip-eps N --steps-per-batch N --epochs N --gae-lambda N --kl-target N (ppo)\n");
    fprintf(stderr, "  --entropy-coef N (a2c/impala/sac)\n");
    fprintf(stderr, "  --replay-size N --batch-size N --per-alpha N --per-beta N (dqn)\n");
    fprintf(stderr, "  --train-every N --learning-starts N --grad-steps N (dqn/rainbow/qrdqn/iqn/sac/td3)\n");
    fprintf(stderr, "  --c51-atoms N --c51-vmin V --c51-vmax V (rainbow)\n");
    fprintf(stderr, "  --iqn-quantiles N --iqn-tau-samples N (iqn)\n");
    fprintf(stderr, "  --actor-count N --learner-batch N --queue-capacity N (impala split)\n");
    fprintf(stderr, "  --mcts-sims N --mcts-depth N (mcts)\n");
    fprintf(stderr, "checkpoint params:\n");
    fprintf(stderr, "  --save PATH --load PATH\n");
}

int tfrl_cli_parse(int argc, char **argv, tfrl_runner_config *out_cfg) {
    if (!out_cfg || argc < 2) return 0;
    if (arg_flag(argc, argv, "--help") || arg_flag(argc, argv, "-h")) return 0;
    if (argc == 2 && strcmp(argv[1], "train") == 0) return 0;
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
    out_cfg->threads = arg_int(argc, argv, "--threads", 0);
    out_cfg->deterministic = arg_flag(argc, argv, "--deterministic");
    out_cfg->steps = arg_int(argc, argv, "--steps", 1000);
    out_cfg->episodes = arg_int(argc, argv, "--episodes", 10);
    out_cfg->seed = arg_int(argc, argv, "--seed", 0);
    out_cfg->gamma = arg_float(argc, argv, "--gamma", -1.0f);
    out_cfg->lr = arg_float(argc, argv, "--lr", -1.0f);
    out_cfg->epsilon = arg_float(argc, argv, "--epsilon", -1.0f);
    out_cfg->entropy_coef = arg_float(argc, argv, "--entropy-coef", -1.0f);
    out_cfg->clip_eps = arg_float(argc, argv, "--clip-eps", -1.0f);
    out_cfg->kl_target = arg_float(argc, argv, "--kl-target", -1.0f);
    out_cfg->gae_lambda = arg_float(argc, argv, "--gae-lambda", -1.0f);
    out_cfg->replay_size = arg_int(argc, argv, "--replay-size", -1);
    out_cfg->batch_size = arg_int(argc, argv, "--batch-size", -1);
    out_cfg->train_every = arg_int(argc, argv, "--train-every", -1);
    out_cfg->learning_starts = arg_int(argc, argv, "--learning-starts", -1);
    out_cfg->grad_steps = arg_int(argc, argv, "--grad-steps", -1);
    out_cfg->per_alpha = arg_float(argc, argv, "--per-alpha", -1.0f);
    out_cfg->per_beta = arg_float(argc, argv, "--per-beta", -1.0f);
    out_cfg->mcts_sims = arg_int(argc, argv, "--mcts-sims", -1);
    out_cfg->mcts_depth = arg_int(argc, argv, "--mcts-depth", -1);
    out_cfg->steps_per_batch = arg_int(argc, argv, "--steps-per-batch", -1);
    out_cfg->epochs = arg_int(argc, argv, "--epochs", -1);
    out_cfg->c51_atoms = arg_int(argc, argv, "--c51-atoms", -1);
    out_cfg->c51_vmin = arg_float(argc, argv, "--c51-vmin", -1.0e30f);
    out_cfg->c51_vmax = arg_float(argc, argv, "--c51-vmax", -1.0e30f);
    out_cfg->iqn_quantiles = arg_int(argc, argv, "--iqn-quantiles", -1);
    out_cfg->iqn_tau_samples = arg_int(argc, argv, "--iqn-tau-samples", -1);
    out_cfg->log_every = arg_int(argc, argv, "--log-every", 100);
    out_cfg->render_every = arg_int(argc, argv, "--render-every", 1);
    out_cfg->render_fps = arg_int(argc, argv, "--render-fps", 60);
    out_cfg->render_env = arg_int(argc, argv, "--render-env", 0);
    out_cfg->trace_out = arg_str(argc, argv, "--trace-out", NULL);
    out_cfg->trace_in = arg_str(argc, argv, "--trace-in", NULL);
    out_cfg->dump_meta_json = arg_flag(argc, argv, "--dump-meta-json");
    out_cfg->profile = arg_flag(argc, argv, "--profile");
    out_cfg->profile_json = arg_str(argc, argv, "--profile-json", NULL);
    out_cfg->actor_count = arg_int(argc, argv, "--actor-count", -1);
    out_cfg->learner_batch = arg_int(argc, argv, "--learner-batch", -1);
    out_cfg->queue_capacity = arg_int(argc, argv, "--queue-capacity", -1);
    out_cfg->save_path = arg_str(argc, argv, "--save", NULL);
    out_cfg->load_path = arg_str(argc, argv, "--load", NULL);
    out_cfg->agents_expected = arg_int(argc, argv, "--agents", 0);
    out_cfg->share_policy = arg_flag(argc, argv, "--share-policy");
    out_cfg->backend = arg_str(argc, argv, "--backend", NULL);
    out_cfg->device = arg_str(argc, argv, "--device", NULL);
    out_cfg->mp_actors = arg_int(argc, argv, "--mp-actors", 0);
    out_cfg->mp_queue = arg_int(argc, argv, "--mp-queue", 0);
    out_cfg->mp_sync_every = arg_int(argc, argv, "--mp-sync-every", 0);
    out_cfg->mp_sync_path = arg_str(argc, argv, "--mp-sync-path", NULL);

    const char *render = arg_str(argc, argv, "--render", "off");
    out_cfg->render = (strcmp(render, "live") == 0);

    if (out_cfg->mode == TFRL_MODE_REPLAY && !out_cfg->trace_in) {
        return 0;
    }
    return 1;
}
