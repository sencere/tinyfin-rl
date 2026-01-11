#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE_LEN 7
#define ACTIONS 2
#define MAX_STEPS 30

typedef struct {
    int pos;
    int goal;
    int steps;
} line_env;

static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static double rng_uniform(uint64_t *state) {
    uint64_t v = rng_next(state) >> 11;
    return (double)v * (1.0 / 9007199254740992.0);
}

static void env_reset(line_env *env) {
    env->pos = 0;
    env->goal = LINE_LEN - 1;
    env->steps = 0;
}

static double env_step(line_env *env, int action, int *done) {
    if (action == 0 && env->pos > 0) {
        env->pos -= 1;
    } else if (action == 1 && env->pos < LINE_LEN - 1) {
        env->pos += 1;
    }
    env->steps += 1;
    int reached = env->pos == env->goal;
    *done = reached || env->steps >= MAX_STEPS;
    return reached ? 1.0 : -0.01;
}

static int arg_int(int argc, char **argv, const char *name, int def) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0) {
            return atoi(argv[i + 1]);
        }
    }
    return def;
}

int main(int argc, char **argv) {
    int episodes = arg_int(argc, argv, "--episodes", 5);
    uint64_t seed = (uint64_t)arg_int(argc, argv, "--seed", 1234);
    uint64_t rng = seed;

    line_env env = {0};
    double return_sum = 0.0;

    for (int ep = 1; ep <= episodes; ep++) {
        env_reset(&env);
        double ep_return = 0.0;
        int done = 0;
        while (!done) {
            int action = (int)(rng_uniform(&rng) * ACTIONS) % ACTIONS;
            double reward = env_step(&env, action, &done);
            ep_return += reward;
        }
        return_sum += ep_return;
        printf("episode=%d return=%.3f\n", ep, ep_return);
    }

    if (episodes > 0) {
        printf("return_mean=%.3f\n", return_sum / episodes);
    }
    return 0;
}
