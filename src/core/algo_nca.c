#include "tfrl/nca.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int obs_type;
    int obs_n;
    int obs_dim;
    int cell_count;
    int action_n;
    int rollout_steps;
    int deterministic;
    float gamma;
    float lr;
    float epsilon;
    uint32_t rng;
    float kernel_left;
    float kernel_self;
    float kernel_right;
    float kernel_bias;
    float *readout;
    float *bias;
    float *value;
    long long updates;
    float last_td;
    float last_q;
    int last_action;
} tfrl_nca_algo;

static uint32_t nca_rng_next(tfrl_nca_algo *algo) {
    algo->rng = algo->rng * 1664525u + 1013904223u;
    return algo->rng;
}

static float nca_rand01(tfrl_nca_algo *algo) {
    uint32_t v = nca_rng_next(algo);
    return (float)((v >> 8) * (1.0 / 16777216.0));
}

static float nca_rand_signed(tfrl_nca_algo *algo) {
    return nca_rand01(algo) * 2.0f - 1.0f;
}

static int nca_obs_dim(const tfrl_algo_config *cfg) {
    if (!cfg) return 0;
    if (cfg->obs_type == TFRL_SPACE_DISCRETE) {
        if (cfg->obs_n > 0 && cfg->obs_n <= TFRL_NCA_MAX_CELLS) return cfg->obs_n;
        return 1;
    }
    if (cfg->obs_dims > 0) return cfg->obs_dims;
    return 1;
}

static void nca_obs_to_cells(const tfrl_nca_algo *algo, tfrl_obs obs, float *cells) {
    if (!algo || !cells) return;
    for (int i = 0; i < algo->cell_count; i++) cells[i] = 0.0f;

    if (algo->obs_type == TFRL_SPACE_DISCRETE) {
        if (algo->cell_count == 1) {
            float denom = algo->obs_n > 1 ? (float)(algo->obs_n - 1) : 1.0f;
            cells[0] = denom > 0.0f ? (float)obs.index / denom : 0.0f;
            return;
        }
        int idx = obs.index;
        if (idx < 0) idx = 0;
        if (idx >= algo->cell_count) idx = algo->cell_count - 1;
        cells[idx] = 1.0f;
        return;
    }

    int len = obs.data_len;
    if (len < 0) len = 0;
    if (len > algo->cell_count) len = algo->cell_count;
    for (int i = 0; i < len; i++) cells[i] = obs.data[i];
}

static void nca_rollout(const tfrl_nca_algo *algo, const float *input, float *state, float *tmp) {
    if (!algo || !input || !state || !tmp) return;
    memcpy(state, input, (size_t)algo->cell_count * sizeof(float));
    for (int step = 0; step < algo->rollout_steps; step++) {
        for (int i = 0; i < algo->cell_count; i++) {
            float left = i > 0 ? state[i - 1] : 0.0f;
            float self = state[i];
            float right = i + 1 < algo->cell_count ? state[i + 1] : 0.0f;
            float drive = input[i] + algo->kernel_left * left +
                          algo->kernel_self * self +
                          algo->kernel_right * right +
                          algo->kernel_bias;
            tmp[i] = tanhf(drive);
        }
        memcpy(state, tmp, (size_t)algo->cell_count * sizeof(float));
    }
}

static void nca_q_values(const tfrl_nca_algo *algo, const float *state, float *q) {
    if (!algo || !state || !q) return;
    for (int a = 0; a < algo->action_n; a++) {
        float v = algo->bias[a];
        for (int i = 0; i < algo->cell_count; i++) {
            v += state[i] * algo->readout[(size_t)i * (size_t)algo->action_n + (size_t)a];
        }
        q[a] = v;
    }
}

static int nca_best_action(const tfrl_nca_algo *algo, const float *q) {
    int best = 0;
    float best_v = q[0];
    for (int a = 1; a < algo->action_n; a++) {
        if (q[a] > best_v) {
            best_v = q[a];
            best = a;
        }
    }
    return best;
}

static tfrl_action nca_act(void *ctx, tfrl_obs obs) {
    tfrl_nca_algo *algo = (tfrl_nca_algo *)ctx;
    tfrl_action action = {0};
    if (!algo || algo->action_n <= 0) return action;

    float input[TFRL_NCA_MAX_CELLS];
    float state[TFRL_NCA_MAX_CELLS];
    float tmp[TFRL_NCA_MAX_CELLS];
    float q[TFRL_NCA_MAX_CELLS];
    nca_obs_to_cells(algo, obs, input);
    nca_rollout(algo, input, state, tmp);
    nca_q_values(algo, state, q);

    int act;
    if (!algo->deterministic && algo->epsilon > 0.0f && nca_rand01(algo) < algo->epsilon) {
        act = (int)(nca_rand01(algo) * (float)algo->action_n);
        if (act < 0) act = 0;
        if (act >= algo->action_n) act = algo->action_n - 1;
    } else {
        act = nca_best_action(algo, q);
    }
    algo->last_action = act;
    algo->last_q = q[act];
    action.index = act;
    return action;
}

static void nca_act_batch(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions) {
    if (!ctx || !out_actions || count <= 0) return;
    for (int i = 0; i < count; i++) {
        out_actions[i] = nca_act(ctx, obs ? obs[i] : (tfrl_obs){0});
    }
}

static void nca_update(void *ctx, const tfrl_transition *transition) {
    tfrl_nca_algo *algo = (tfrl_nca_algo *)ctx;
    if (!algo || !transition || algo->action_n <= 0) return;

    int action = transition->action.index;
    if (action < 0) action = 0;
    if (action >= algo->action_n) action = algo->action_n - 1;

    float input[TFRL_NCA_MAX_CELLS];
    float state[TFRL_NCA_MAX_CELLS];
    float tmp[TFRL_NCA_MAX_CELLS];
    float q[TFRL_NCA_MAX_CELLS];
    float next_input[TFRL_NCA_MAX_CELLS];
    float next_state[TFRL_NCA_MAX_CELLS];
    float next_q[TFRL_NCA_MAX_CELLS];

    nca_obs_to_cells(algo, transition->obs, input);
    nca_rollout(algo, input, state, tmp);
    nca_q_values(algo, state, q);
    nca_obs_to_cells(algo, transition->next_obs, next_input);
    nca_rollout(algo, next_input, next_state, tmp);
    nca_q_values(algo, next_state, next_q);

    float max_next = next_q[0];
    for (int a = 1; a < algo->action_n; a++) {
        if (next_q[a] > max_next) max_next = next_q[a];
    }
    float target = (float)transition->reward;
    if (!transition->done) target += algo->gamma * max_next;
    float td = target - q[action];

    float step = algo->lr * td;
    for (int i = 0; i < algo->cell_count; i++) {
        size_t idx = (size_t)i * (size_t)algo->action_n + (size_t)action;
        algo->readout[idx] += step * state[i];
        algo->value[i] += algo->lr * 0.25f * td * state[i];
    }
    algo->bias[action] += step;

    float corr_left = 0.0f;
    float corr_self = 0.0f;
    float corr_right = 0.0f;
    float corr_bias = 0.0f;
    for (int i = 0; i < algo->cell_count; i++) {
        float grad = algo->readout[(size_t)i * (size_t)algo->action_n + (size_t)action];
        float left = i > 0 ? input[i - 1] : 0.0f;
        float self = input[i];
        float right = i + 1 < algo->cell_count ? input[i + 1] : 0.0f;
        corr_left += grad * left;
        corr_self += grad * self;
        corr_right += grad * right;
        corr_bias += grad;
    }
    float kstep = algo->lr * td * 0.001f / (float)algo->cell_count;
    algo->kernel_left += kstep * corr_left;
    algo->kernel_self += kstep * corr_self;
    algo->kernel_right += kstep * corr_right;
    algo->kernel_bias += kstep * corr_bias;

    algo->last_td = td;
    algo->updates += 1;
}

static void nca_save(void *ctx, const char *path) {
    tfrl_nca_algo *algo = (tfrl_nca_algo *)ctx;
    if (!algo || !path) return;
    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    fprintf(fp, "TFRL_NCA %d\n", TFRL_NCA_VERSION);
    fprintf(fp, "%d %d %d %d %d %.9g %.9g %.9g %u\n",
            algo->obs_type, algo->obs_n, algo->cell_count, algo->action_n,
            algo->rollout_steps, algo->gamma, algo->lr, algo->epsilon,
            (unsigned)algo->rng);
    fprintf(fp, "%.9g %.9g %.9g %.9g\n",
            algo->kernel_left, algo->kernel_self, algo->kernel_right, algo->kernel_bias);
    for (int i = 0; i < algo->cell_count * algo->action_n; i++) fprintf(fp, "%.9g\n", algo->readout[i]);
    for (int i = 0; i < algo->action_n; i++) fprintf(fp, "%.9g\n", algo->bias[i]);
    for (int i = 0; i < algo->cell_count; i++) fprintf(fp, "%.9g\n", algo->value[i]);
    fclose(fp);
}

static int nca_load(tfrl_nca_algo *algo, const char *path) {
    if (!algo || !path) return 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    char magic[32];
    int version = 0;
    if (fscanf(fp, "%31s %d", magic, &version) != 2 ||
        strcmp(magic, "TFRL_NCA") != 0 || version != TFRL_NCA_VERSION) {
        fclose(fp);
        return 0;
    }
    unsigned rng = 0;
    if (fscanf(fp, "%d %d %d %d %d %f %f %f %u",
               &algo->obs_type, &algo->obs_n, &algo->cell_count, &algo->action_n,
               &algo->rollout_steps, &algo->gamma, &algo->lr, &algo->epsilon,
               &rng) != 9) {
        fclose(fp);
        return 0;
    }
    algo->rng = (uint32_t)rng;
    if (algo->cell_count <= 0 || algo->cell_count > TFRL_NCA_MAX_CELLS ||
        algo->action_n <= 0 || algo->action_n > TFRL_NCA_MAX_CELLS) {
        fclose(fp);
        return 0;
    }
    if (fscanf(fp, "%f %f %f %f",
               &algo->kernel_left, &algo->kernel_self,
               &algo->kernel_right, &algo->kernel_bias) != 4) {
        fclose(fp);
        return 0;
    }
    size_t readout_count = (size_t)algo->cell_count * (size_t)algo->action_n;
    algo->readout = (float *)calloc(readout_count, sizeof(float));
    algo->bias = (float *)calloc((size_t)algo->action_n, sizeof(float));
    algo->value = (float *)calloc((size_t)algo->cell_count, sizeof(float));
    if (!algo->readout || !algo->bias || !algo->value) {
        fclose(fp);
        return 0;
    }
    for (size_t i = 0; i < readout_count; i++) {
        if (fscanf(fp, "%f", &algo->readout[i]) != 1) { fclose(fp); return 0; }
    }
    for (int i = 0; i < algo->action_n; i++) {
        if (fscanf(fp, "%f", &algo->bias[i]) != 1) { fclose(fp); return 0; }
    }
    for (int i = 0; i < algo->cell_count; i++) {
        if (fscanf(fp, "%f", &algo->value[i]) != 1) { fclose(fp); return 0; }
    }
    fclose(fp);
    return 1;
}

static void nca_destroy(void *ctx) {
    tfrl_nca_algo *algo = (tfrl_nca_algo *)ctx;
    if (!algo) return;
    free(algo->readout);
    free(algo->bias);
    free(algo->value);
    free(algo);
}

static int nca_diagnostics(void *ctx, char *buffer, size_t buffer_len) {
    tfrl_nca_algo *algo = (tfrl_nca_algo *)ctx;
    if (!algo || !buffer || buffer_len == 0) return 0;
    snprintf(buffer, buffer_len,
             "nca version=%d cells=%d rollout=%d updates=%lld td=%.5f q=%.5f eps=%.3f",
             TFRL_NCA_VERSION, algo->cell_count, algo->rollout_steps,
             algo->updates, algo->last_td, algo->last_q, algo->epsilon);
    return 1;
}

static const tfrl_algo_vtable NCA_VTABLE = {
    .act = nca_act,
    .act_env = NULL,
    .act_batch = nca_act_batch,
    .update = nca_update,
    .save = nca_save,
    .destroy = nca_destroy,
    .diagnostics = nca_diagnostics,
};

tfrl_algo tfrl_algo_nca_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->action_type != TFRL_SPACE_DISCRETE || cfg->action_n <= 0) return out;

    tfrl_nca_algo *algo = (tfrl_nca_algo *)calloc(1, sizeof(tfrl_nca_algo));
    if (!algo) return out;
    algo->obs_type = cfg->obs_type;
    algo->obs_n = cfg->obs_n;
    algo->obs_dim = nca_obs_dim(cfg);
    algo->cell_count = algo->obs_dim;
    if (algo->cell_count < 4) algo->cell_count = 4;
    if (algo->cell_count > TFRL_NCA_MAX_CELLS) algo->cell_count = TFRL_NCA_MAX_CELLS;
    algo->action_n = cfg->action_n;
    if (algo->action_n > TFRL_NCA_MAX_CELLS) algo->action_n = TFRL_NCA_MAX_CELLS;
    algo->rollout_steps = TFRL_NCA_DEFAULT_ROLLOUT_STEPS;
    algo->deterministic = cfg->deterministic;
    algo->gamma = cfg->gamma > 0.0f ? cfg->gamma : 0.99f;
    algo->lr = cfg->lr > 0.0f ? cfg->lr : 0.02f;
    algo->epsilon = cfg->epsilon >= 0.0f ? cfg->epsilon : 0.1f;
    algo->rng = cfg->seed ? (uint32_t)cfg->seed : 1u;

    if (cfg->load_path && nca_load(algo, cfg->load_path)) {
        algo->deterministic = cfg->deterministic;
        out.ctx = algo;
        out.vtable = &NCA_VTABLE;
        return out;
    }

    size_t readout_count = (size_t)algo->cell_count * (size_t)algo->action_n;
    algo->readout = (float *)calloc(readout_count, sizeof(float));
    algo->bias = (float *)calloc((size_t)algo->action_n, sizeof(float));
    algo->value = (float *)calloc((size_t)algo->cell_count, sizeof(float));
    if (!algo->readout || !algo->bias || !algo->value) {
        nca_destroy(algo);
        return out;
    }

    algo->kernel_left = 0.35f + 0.05f * nca_rand_signed(algo);
    algo->kernel_self = 0.65f + 0.05f * nca_rand_signed(algo);
    algo->kernel_right = 0.35f + 0.05f * nca_rand_signed(algo);
    algo->kernel_bias = 0.0f;
    for (size_t i = 0; i < readout_count; i++) {
        algo->readout[i] = 0.01f * nca_rand_signed(algo);
    }

    out.ctx = algo;
    out.vtable = &NCA_VTABLE;
    return out;
}
