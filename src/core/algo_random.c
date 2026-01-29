#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "algo_api.h"

typedef struct {
    int action_n;
    tfrl_space_type action_type;
    int action_dims;
    int action_shape[2];
    double action_low;
    double action_high;
} tfrl_random_algo;

typedef struct {
    int action_n;
    int table_size;
    float *q;
    float alpha;
    float gamma;
    float epsilon;
    tfrl_space_type obs_type;
    double obs_low;
    double obs_high;
    float action_bias_right;
    uint32_t rng;
    int deterministic;
} tfrl_simple_q;

static uint32_t simple_q_rng_next(tfrl_simple_q *algo) {
    algo->rng = algo->rng * 1664525u + 1013904223u;
    return algo->rng;
}

static float simple_q_rand01(tfrl_simple_q *algo) {
    uint32_t v = simple_q_rng_next(algo);
    return (float)((v >> 8) * (1.0 / 16777216.0));
}

static int simple_q_obs_index(tfrl_simple_q *algo, tfrl_obs obs) {
    if (!algo || algo->table_size <= 0) return 0;
    if (algo->obs_type == TFRL_SPACE_DISCRETE || obs.data_len <= 0) {
        int idx = obs.index;
        if (idx < 0) idx = 0;
        if (idx >= algo->table_size) idx %= algo->table_size;
        return idx;
    }

    const int bins = 16;
    double lo = algo->obs_low;
    double hi = algo->obs_high;
    double denom = hi - lo;
    uint32_t h = 2166136261u;
    int len = obs.data_len;
    if (len > TFRL_MAX_BOX_DIMS) len = TFRL_MAX_BOX_DIMS;
    for (int i = 0; i < len; i++) {
        float v = obs.data[i];
        float t;
        if (fabs(denom) > 1e-8) {
            t = (float)((v - lo) / denom);
        } else {
            t = v;
        }
        if (t < 0.0f) t = 0.0f;
        if (t > 0.9999f) t = 0.9999f;
        int b = (int)(t * (float)bins);
        h ^= (uint32_t)b;
        h *= 16777619u;
    }
    return (int)(h % (uint32_t)algo->table_size);
}

static int simple_q_greedy_action(tfrl_simple_q *algo, int state) {
    int best = 0;
    float best_q = algo->q[state * algo->action_n];
    if (algo->action_bias_right != 0.0f && algo->action_n > 2) {
        best_q += algo->action_bias_right;
    }
    for (int a = 1; a < algo->action_n; a++) {
        float q = algo->q[state * algo->action_n + a];
        if (algo->action_bias_right != 0.0f && a == 2) {
            q += algo->action_bias_right;
        }
        if (q > best_q) {
            best_q = q;
            best = a;
        }
    }
    return best;
}

static tfrl_action random_act(void *ctx, tfrl_obs obs) {
    (void)obs;
    tfrl_random_algo *algo = (tfrl_random_algo *)ctx;
    tfrl_action action = {0};
    if (!algo) return action;

    if (algo->action_type == TFRL_SPACE_DISCRETE) {
        int n = algo->action_n > 0 ? algo->action_n : 1;
        action.index = rand() % n;
    } else {
        int dims = algo->action_dims > 0 ? algo->action_dims : 1;
        if (dims > TFRL_MAX_BOX_DIMS) dims = TFRL_MAX_BOX_DIMS;
        action.data_len = dims;
        double lo = algo->action_low;
        double hi = algo->action_high;
        if (hi < lo) {
            double tmp = hi;
            hi = lo;
            lo = tmp;
        }
        for (int i = 0; i < dims; i++) {
            double r = (double)rand() / (double)RAND_MAX;
            action.data[i] = (float)(lo + (hi - lo) * r);
        }
    }
    return action;
}

static tfrl_action simple_q_act(void *ctx, tfrl_obs obs) {
    tfrl_simple_q *algo = (tfrl_simple_q *)ctx;
    tfrl_action action = {0};
    if (!algo || algo->action_n <= 0) return action;

    int state = simple_q_obs_index(algo, obs);
    int act = 0;
    if (!algo->deterministic && algo->epsilon > 0.0f && simple_q_rand01(algo) < algo->epsilon) {
        act = (int)(simple_q_rand01(algo) * (float)algo->action_n);
        if (act < 0) act = 0;
        if (act >= algo->action_n) act = algo->action_n - 1;
    } else {
        act = simple_q_greedy_action(algo, state);
    }
    action.index = act;
    return action;
}

static void random_act_batch(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions) {
    if (!ctx || !out_actions || count <= 0) return;
    for (int i = 0; i < count; i++) {
        out_actions[i] = random_act(ctx, obs ? obs[i] : (tfrl_obs){0});
    }
}

static void simple_q_act_batch(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions) {
    if (!ctx || !out_actions || count <= 0) return;
    for (int i = 0; i < count; i++) {
        out_actions[i] = simple_q_act(ctx, obs ? obs[i] : (tfrl_obs){0});
    }
}

static void random_update(void *ctx, const tfrl_transition *transition) {
    (void)ctx;
    (void)transition;
}

static void simple_q_update(void *ctx, const tfrl_transition *transition) {
    tfrl_simple_q *algo = (tfrl_simple_q *)ctx;
    if (!algo || !transition || algo->action_n <= 0 || algo->table_size <= 0) return;

    int state = simple_q_obs_index(algo, transition->obs);
    int next_state = simple_q_obs_index(algo, transition->next_obs);
    int action = transition->action.index;
    if (action < 0) action = 0;
    if (action >= algo->action_n) action = algo->action_n - 1;

    float q_sa = algo->q[state * algo->action_n + action];
    float max_next = algo->q[next_state * algo->action_n];
    for (int a = 1; a < algo->action_n; a++) {
        float q = algo->q[next_state * algo->action_n + a];
        if (q > max_next) max_next = q;
    }
    float target = (float)transition->reward;
    if (!transition->done) target += algo->gamma * max_next;
    algo->q[state * algo->action_n + action] =
        q_sa + algo->alpha * (target - q_sa);
}

static void random_destroy(void *ctx) {
    free(ctx);
}

static void simple_q_destroy(void *ctx) {
    tfrl_simple_q *algo = (tfrl_simple_q *)ctx;
    if (algo) {
        free(algo->q);
        free(algo);
    }
}

static int random_diagnostics(void *ctx, char *buffer, size_t buffer_len) {
    (void)ctx;
    if (!buffer || buffer_len == 0) return 0;
    buffer[0] = '\0';
    return 1;
}

static int simple_q_diagnostics(void *ctx, char *buffer, size_t buffer_len) {
    tfrl_simple_q *algo = (tfrl_simple_q *)ctx;
    if (!buffer || buffer_len == 0 || !algo) return 0;
    snprintf(buffer, buffer_len, "simple_q eps=%.3f alpha=%.3f gamma=%.3f table=%d",
             algo->epsilon, algo->alpha, algo->gamma, algo->table_size);
    return 1;
}

static const tfrl_algo_vtable RANDOM_VTABLE = {
    .act = random_act,
    .act_env = NULL,
    .act_batch = random_act_batch,
    .update = random_update,
    .save = NULL,
    .destroy = random_destroy,
    .diagnostics = random_diagnostics,
};

static const tfrl_algo_vtable SIMPLE_Q_VTABLE = {
    .act = simple_q_act,
    .act_env = NULL,
    .act_batch = simple_q_act_batch,
    .update = simple_q_update,
    .save = NULL,
    .destroy = simple_q_destroy,
    .diagnostics = simple_q_diagnostics,
};

static tfrl_algo simple_q_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->action_n <= 0) return out;

    tfrl_simple_q *algo = (tfrl_simple_q *)calloc(1, sizeof(tfrl_simple_q));
    if (!algo) return out;

    int table_size = 4096;
    if (cfg->obs_type == TFRL_SPACE_DISCRETE && cfg->obs_n > 0) {
        table_size = cfg->obs_n;
        if (table_size > 200000) table_size = 200000;
    }

    algo->action_n = cfg->action_n;
    algo->table_size = table_size;
    algo->alpha = (cfg->lr > 0.0f) ? cfg->lr : 0.1f;
    algo->gamma = (cfg->gamma > 0.0f) ? cfg->gamma : 0.99f;
    algo->epsilon = (cfg->epsilon >= 0.0f) ? cfg->epsilon : 0.1f;
    algo->obs_type = cfg->obs_type;
    algo->obs_low = cfg->obs_low;
    algo->obs_high = cfg->obs_high;
    algo->action_bias_right = cfg->action_bias_right;
    algo->rng = cfg->seed ? (uint32_t)cfg->seed : 1u;
    algo->deterministic = cfg->deterministic;

    size_t q_count = (size_t)algo->table_size * (size_t)algo->action_n;
    algo->q = (float *)calloc(q_count, sizeof(float));
    if (!algo->q) {
        free(algo);
        return out;
    }

    out.ctx = algo;
    out.vtable = &SIMPLE_Q_VTABLE;
    return out;
}

tfrl_algo tfrl_algo_random_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->action_n <= 0) return out;
    if (cfg->name && strcmp(cfg->name, "random") != 0) {
        return simple_q_create(cfg);
    }
    tfrl_random_algo *algo = (tfrl_random_algo *)calloc(1, sizeof(tfrl_random_algo));
    if (!algo) return out;
    algo->action_n = cfg->action_n;
    algo->action_type = cfg->action_type;
    algo->action_dims = cfg->action_dims;
    algo->action_shape[0] = cfg->action_shape[0];
    algo->action_shape[1] = cfg->action_shape[1];
    algo->action_low = cfg->action_low;
    algo->action_high = cfg->action_high;
    out.ctx = algo;
    out.vtable = &RANDOM_VTABLE;
    return out;
}
