#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/algo_api.h"
#include "core/env_api.h"

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

static void print_usage(const char *name) {
    fprintf(stderr, "usage: %s --algo NAME --env NAME --load PATH [--steps N] [--episodes N]\n", name);
    fprintf(stderr, "       [--seed N] [--deterministic] [--print-every N]\n");
}

static void print_action(const tfrl_env_spec *spec, const tfrl_action *action) {
    if (!spec || !action) return;
    if (spec->action_type == TFRL_SPACE_DISCRETE) {
        fprintf(stdout, "action=%d", action->index);
        return;
    }
    fprintf(stdout, "action=[");
    for (int i = 0; i < action->data_len; i++) {
        if (i) fputc(',', stdout);
        fprintf(stdout, "%.3f", action->data[i]);
    }
    fprintf(stdout, "]");
}

int main(int argc, char **argv) {
    if (argc < 2 || arg_flag(argc, argv, "--help") || arg_flag(argc, argv, "-h")) {
        print_usage(argv[0]);
        return 1;
    }

    const char *algo_name = arg_str(argc, argv, "--algo", NULL);
    const char *env_name = arg_str(argc, argv, "--env", NULL);
    const char *load_path = arg_str(argc, argv, "--load", NULL);
    int steps = arg_int(argc, argv, "--steps", 1000);
    int episodes = arg_int(argc, argv, "--episodes", 0);
    int seed = arg_int(argc, argv, "--seed", 0);
    int deterministic = arg_flag(argc, argv, "--deterministic");
    int print_every = arg_int(argc, argv, "--print-every", 0);

    if (!algo_name || !env_name || !load_path) {
        print_usage(argv[0]);
        return 1;
    }

    tfrl_env_config env_cfg = {0};
    env_cfg.name = env_name;
    env_cfg.seed = (uint64_t)seed;
    tfrl_env *env = tfrl_env_create(&env_cfg);
    if (!env) {
        fprintf(stderr, "failed to create env '%s'\n", env_name);
        return 1;
    }
    const tfrl_env_spec *spec = tfrl_env_get_spec(env);
    if (!spec) {
        fprintf(stderr, "missing env spec for '%s'\n", env_name);
        tfrl_env_destroy(env);
        return 1;
    }

    tfrl_algo_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.name = algo_name;
    cfg.seed = seed;
    cfg.deterministic = deterministic;
    cfg.load_path = load_path;
    cfg.env_name = env_name;
    tfrl_algo_config_apply_defaults(&cfg);
    tfrl_algo algo = tfrl_algo_create(&cfg, spec);
    if (!algo.vtable || !algo.vtable->act) {
        fprintf(stderr, "failed to create algo '%s'\n", algo_name);
        tfrl_env_destroy(env);
        return 1;
    }

    int episode_count = 0;
    uint64_t reset_seed = (uint64_t)seed;
    tfrl_obs obs = tfrl_env_reset(env, reset_seed);
    for (int step = 0; step < steps; step++) {
        tfrl_action action = algo.vtable->act(algo.ctx, obs);
        tfrl_step_result out = tfrl_env_step(env, action);
        if (print_every > 0 && (step % print_every) == 0) {
            fprintf(stdout, "step=%d reward=%.4f done=%d ", step, out.reward, out.done);
            print_action(spec, &action);
            fputc('\n', stdout);
        }
        if (out.done) {
            episode_count++;
            if (episodes > 0 && episode_count >= episodes) break;
            reset_seed += 1;
            obs = tfrl_env_reset(env, reset_seed);
        } else {
            obs = out.observation;
        }
    }

    tfrl_algo_destroy(&algo);
    tfrl_env_destroy(env);
    return 0;
}
