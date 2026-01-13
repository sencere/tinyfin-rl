#include <math.h>
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

static int env_state(const line_env *env) {
    return env->pos;
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

static double arg_double(int argc, char **argv, const char *name, double def) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0) {
            return atof(argv[i + 1]);
        }
    }
    return def;
}

static int has_flag(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0) {
            return 1;
        }
    }
    return 0;
}

static const char *arg_str(int argc, char **argv, const char *name, const char *def) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return def;
}

static int save_q_table(const char *path, double q[LINE_LEN][ACTIONS]) {
    FILE *f = fopen(path, "w");
    if (!f) {
        return 0;
    }
    fprintf(f, "state,action,value\n");
    for (int s = 0; s < LINE_LEN; s++) {
        for (int a = 0; a < ACTIONS; a++) {
            fprintf(f, "%d,%d,%.6f\n", s, a, q[s][a]);
        }
    }
    fclose(f);
    return 1;
}

static int load_q_table(const char *path, double q[LINE_LEN][ACTIONS]) {
    FILE *f = fopen(path, "r");
    if (!f) {
        return 0;
    }
    char line[128];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return 0;
    }
    while (fgets(line, sizeof(line), f)) {
        int s = 0;
        int a = 0;
        double v = 0.0;
        if (sscanf(line, "%d,%d,%lf", &s, &a, &v) == 3) {
            if (s >= 0 && s < LINE_LEN && a >= 0 && a < ACTIONS) {
                q[s][a] = v;
            }
        }
    }
    fclose(f);
    return 1;
}

static int save_metrics_csv(const char *path, const double *returns, int episodes) {
    FILE *f = fopen(path, "w");
    if (!f) {
        return 0;
    }
    fprintf(f, "episode,return\n");
    for (int i = 0; i < episodes; i++) {
        fprintf(f, "%d,%.6f\n", i + 1, returns[i]);
    }
    fclose(f);
    return 1;
}

static int save_metrics_json(const char *path, const double *returns, int episodes) {
    FILE *f = fopen(path, "w");
    if (!f) {
        return 0;
    }
    double sum = 0.0;
    for (int i = 0; i < episodes; i++) {
        sum += returns[i];
    }
    double mean = episodes > 0 ? sum / episodes : 0.0;
    fprintf(f, "{\"episodes\":[");
    for (int i = 0; i < episodes; i++) {
        fprintf(f, "%.6f", returns[i]);
        if (i + 1 < episodes) {
            fprintf(f, ",");
        }
    }
    fprintf(f, "],\"mean_return\":%.6f}\n", mean);
    fclose(f);
    return 1;
}

int main(int argc, char **argv) {
    int episodes = arg_int(argc, argv, "--episodes", 2000);
    double alpha = arg_double(argc, argv, "--alpha", 0.2);
    double gamma = arg_double(argc, argv, "--gamma", 0.98);
    double eps = arg_double(argc, argv, "--eps", 0.2);
    int report_every = arg_int(argc, argv, "--report", 200);
    uint64_t seed = (uint64_t)arg_int(argc, argv, "--seed", 1234);
    int eval_mode = has_flag(argc, argv, "--eval");
    const char *in_path = arg_str(argc, argv, "--in", "q_table.csv");
    const char *out_path = arg_str(argc, argv, "--out", "q_table.csv");
    const char *metrics_path = arg_str(argc, argv, "--metrics", "");
    const char *metrics_format = arg_str(argc, argv, "--metrics-format", "csv");

    double q[LINE_LEN][ACTIONS];
    for (int s = 0; s < LINE_LEN; s++) {
        for (int a = 0; a < ACTIONS; a++) {
            q[s][a] = 0.0;
        }
    }
    if (eval_mode) {
        if (!load_q_table(in_path, q)) {
            fprintf(stderr, "failed to read %s\n", in_path);
            return 1;
        }
    }

    line_env env = {0};
    uint64_t rng = seed;
    double return_sum = 0.0;
    double *returns = NULL;
    if (metrics_path[0] != '\0') {
        returns = (double *)malloc(sizeof(double) * (size_t)episodes);
        if (!returns) {
            fprintf(stderr, "failed to allocate metrics buffer\n");
            return 1;
        }
    }

    for (int ep = 1; ep <= episodes; ep++) {
        env_reset(&env);
        int state = env_state(&env);
        double ep_return = 0.0;
        int done = 0;

        while (!done) {
            int action = 0;
            if (!eval_mode && rng_uniform(&rng) < eps) {
                action = (int)(rng_uniform(&rng) * ACTIONS) % ACTIONS;
            } else {
                double best = q[state][0];
                action = 0;
                for (int a = 1; a < ACTIONS; a++) {
                    if (q[state][a] > best) {
                        best = q[state][a];
                        action = a;
                    }
                }
            }

            double reward = env_step(&env, action, &done);
            int next_state = env_state(&env);
            double next_best = q[next_state][0];
            for (int a = 1; a < ACTIONS; a++) {
                if (q[next_state][a] > next_best) {
                    next_best = q[next_state][a];
                }
            }

            if (!eval_mode) {
                double td_target = reward + (done ? 0.0 : gamma * next_best);
                q[state][action] += alpha * (td_target - q[state][action]);
            }
            state = next_state;
            ep_return += reward;
        }

        return_sum += ep_return;
        if (returns) {
            returns[ep - 1] = ep_return;
        }
        if (report_every > 0 && ep % report_every == 0) {
            printf("episode=%d return_mean=%.3f\n", ep, return_sum / report_every);
            return_sum = 0.0;
        }
    }

    if (!eval_mode) {
        if (!save_q_table(out_path, q)) {
            fprintf(stderr, "failed to write %s\n", out_path);
            return 1;
        }
        printf("saved=%s\n", out_path);
    }

    if (returns) {
        int ok = 0;
        if (strcmp(metrics_format, "json") == 0) {
            ok = save_metrics_json(metrics_path, returns, episodes);
        } else {
            ok = save_metrics_csv(metrics_path, returns, episodes);
        }
        if (!ok) {
            fprintf(stderr, "failed to write metrics to %s\n", metrics_path);
            free(returns);
            return 1;
        }
        printf("metrics=%s\n", metrics_path);
        free(returns);
    }

    return 0;
}
