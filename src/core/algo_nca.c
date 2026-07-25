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
    int ppo_mode;
    float clip_eps;
    float entropy_coef;
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
    float last_logprob;
    int ppo_batch_size;
    int ppo_batch_count;
    int ppo_epochs;
    tfrl_transition *ppo_batch;
    float *ppo_old_logprobs;
    int population_size, population_current, population_best;
    float population_episode_return, population_best_return;
    float *population_readout, *population_bias, *population_scores;
    int population_episodes;
    int population_episode_count;
    float population_score_sum;
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

static size_t nca_param_count(const tfrl_nca_algo *a) {
    return (size_t)a->cell_count * (size_t)a->action_n;
}

static void nca_population_load(tfrl_nca_algo *a, int index) {
    size_t n = nca_param_count(a);
    memcpy(a->readout, a->population_readout + (size_t)index * n, n * sizeof(float));
    memcpy(a->bias, a->population_bias + (size_t)index * (size_t)a->action_n,
           (size_t)a->action_n * sizeof(float));
}

static void nca_population_store(tfrl_nca_algo *a, int index) {
    size_t n = nca_param_count(a);
    memcpy(a->population_readout + (size_t)index * n, a->readout, n * sizeof(float));
    memcpy(a->population_bias + (size_t)index * (size_t)a->action_n, a->bias,
           (size_t)a->action_n * sizeof(float));
}

static void nca_population_finish_episode(tfrl_nca_algo *a) {
    if (!a->ppo_mode || a->population_size <= 1) return;
    a->population_score_sum += a->population_episode_return;
    a->population_episode_count++;
    a->population_episode_return = 0.0f;
    if (a->population_episode_count < a->population_episodes) return;
    int current = a->population_current;
    nca_population_store(a, current);
    a->population_scores[current] = a->population_score_sum / (float)a->population_episodes;
    if (a->population_scores[current] > a->population_best_return) {
        a->population_best_return = a->population_scores[current];
        a->population_best = current;
    }
    a->population_current = (current + 1) % a->population_size;
    nca_population_load(a, a->population_best);
    size_t n = nca_param_count(a);
    for (size_t i = 0; i < n; i++) a->readout[i] += 0.01f * nca_rand_signed(a);
    for (int i = 0; i < a->action_n; i++) a->bias[i] += 0.01f * nca_rand_signed(a);
    nca_population_store(a, a->population_current);
    a->population_episode_count = 0;
    a->population_score_sum = 0.0f;
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

static void nca_softmax(const tfrl_nca_algo *algo, const float *q, float *p) {
    float max_q = q[0];
    for (int a = 1; a < algo->action_n; a++) if (q[a] > max_q) max_q = q[a];
    float total = 0.0f;
    for (int a = 0; a < algo->action_n; a++) {
        p[a] = expf(q[a] - max_q);
        total += p[a];
    }
    if (total <= 0.0f) total = 1.0f;
    for (int a = 0; a < algo->action_n; a++) p[a] /= total;
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
    float probs[TFRL_NCA_MAX_CELLS];
    nca_softmax(algo, q, probs);

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
    algo->last_logprob = logf(probs[act] > 1e-8f ? probs[act] : 1e-8f);
    action.index = act;
    return action;
}

static tfrl_action nca_act_env(void *ctx, tfrl_env *env, tfrl_obs obs) {
    tfrl_nca_algo *algo = (tfrl_nca_algo *)ctx;
    if (!algo || !env) return nca_act(ctx, obs);
    const tfrl_env_spec *spec = tfrl_env_get_spec(env);
    if (!spec || !spec->name || strcmp(spec->name, "tetris") != 0) return nca_act(ctx, obs);
    size_t state_size = tfrl_env_state_size(env);
    unsigned char *saved = (unsigned char *)malloc(state_size);
    if (!saved || !tfrl_env_state_save(env, saved, state_size)) { free(saved); return nca_act(ctx, obs); }
    int best = 0;
    double best_score = -1.0e30;
    for (int action = 0; action < algo->action_n; action++) {
        if (!tfrl_env_state_load(env, saved, state_size)) continue;
        tfrl_step_result step = tfrl_env_step(env, (tfrl_action){.index = action});
        float input[TFRL_NCA_MAX_CELLS], state[TFRL_NCA_MAX_CELLS], tmp[TFRL_NCA_MAX_CELLS];
        float q[TFRL_NCA_MAX_CELLS];
        nca_obs_to_cells(algo, step.observation, input);
        nca_rollout(algo, input, state, tmp);
        nca_q_values(algo, state, q);
        float value = q[0];
        for (int a = 1; a < algo->action_n; a++) if (q[a] > value) value = q[a];
        double score = step.reward + (step.done ? 0.0 : algo->gamma * (double)value);
        if (score > best_score) { best_score = score; best = action; }
    }
    tfrl_env_state_load(env, saved, state_size);
    free(saved);
    algo->last_action = best;
    algo->last_logprob = 0.0f;
    return (tfrl_action){.index = best};
}

static void nca_act_batch(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions) {
    if (!ctx || !out_actions || count <= 0) return;
    for (int i = 0; i < count; i++) {
        out_actions[i] = nca_act(ctx, obs ? obs[i] : (tfrl_obs){0});
    }
}

static void nca_update_one(void *ctx, const tfrl_transition *transition) {
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

    /* PPO clips policy ratios, not rewards or advantages. Keep the complete
     * TD signal here; clipping it to clip_eps made sparse-reward coin_maze
     * learning effectively stop. */

    float step = algo->lr * td;
    float ppo_probs[TFRL_NCA_MAX_CELLS] = {0};
    float ppo_advantage = td;
    if (algo->ppo_mode && action == algo->last_action) {
        nca_softmax(algo, q, ppo_probs);
        float current_logprob = logf(ppo_probs[action] > 1e-8f ? ppo_probs[action] : 1e-8f);
        float ratio = expf(current_logprob - algo->last_logprob);
        float clipped_ratio = ratio;
        float lo = 1.0f - algo->clip_eps;
        float hi = 1.0f + algo->clip_eps;
        if (clipped_ratio < lo) clipped_ratio = lo;
        if (clipped_ratio > hi) clipped_ratio = hi;
        ppo_advantage = td * clipped_ratio;
        step = algo->lr * ppo_advantage;
    }
    if (algo->ppo_mode && action == algo->last_action) {
        for (int i = 0; i < algo->cell_count; i++) {
            for (int a = 0; a < algo->action_n; a++) {
                float grad = ((a == action) ? 1.0f : 0.0f) - ppo_probs[a];
                size_t idx = (size_t)i * (size_t)algo->action_n + (size_t)a;
                algo->readout[idx] += step * grad * state[i];
            }
            algo->value[i] += algo->lr * 0.25f * td * state[i];
        }
        for (int a = 0; a < algo->action_n; a++) {
            float grad = ((a == action) ? 1.0f : 0.0f) - ppo_probs[a];
            algo->bias[a] += step * grad;
        }
    } else {
        for (int i = 0; i < algo->cell_count; i++) {
            size_t idx = (size_t)i * (size_t)algo->action_n + (size_t)action;
            algo->readout[idx] += step * state[i];
            algo->value[i] += algo->lr * 0.25f * td * state[i];
        }
        algo->bias[action] += step;
    }

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
    if (algo->ppo_mode && algo->epsilon > 0.02f) algo->epsilon *= 0.9999f;
}

static void nca_update(void *ctx, const tfrl_transition *transition) {
    tfrl_nca_algo *algo = (tfrl_nca_algo *)ctx;
    if (!algo || !transition) return;
    if (!algo->ppo_mode) { nca_update_one(ctx, transition); return; }
    int slot = algo->ppo_batch_count;
    if (slot < algo->ppo_batch_size) {
        algo->ppo_batch[slot] = *transition;
        algo->ppo_old_logprobs[slot] = algo->last_logprob;
        algo->ppo_batch_count++;
    }
    algo->population_episode_return += transition->reward;
    if (algo->ppo_batch_count < algo->ppo_batch_size && !transition->done) return;
    int order[256];
    for (int i = 0; i < algo->ppo_batch_count; i++) order[i] = i;
    int epochs = algo->ppo_epochs > 0 ? algo->ppo_epochs : 1;
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = algo->ppo_batch_count - 1; i > 0; i--) {
            int j = (int)(nca_rand01(algo) * (float)(i + 1));
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }
        for (int k = 0; k < algo->ppo_batch_count; k++) {
            int i = order[k];
            algo->last_action = algo->ppo_batch[i].action.index;
            algo->last_logprob = algo->ppo_old_logprobs[i];
            nca_update_one(ctx, &algo->ppo_batch[i]);
        }
    }
    algo->ppo_batch_count = 0;
    if (transition->done) nca_population_finish_episode(algo);
}

static void nca_save(void *ctx, const char *path) {
    tfrl_nca_algo *algo = (tfrl_nca_algo *)ctx;
    if (!algo || !path) return;
    if (algo->ppo_mode && algo->population_readout && algo->population_best >= 0)
        nca_population_load(algo, algo->population_best);
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
    free(algo->ppo_batch);
    free(algo->ppo_old_logprobs);
    free(algo->population_readout);
    free(algo->population_bias);
    free(algo->population_scores);
    free(algo);
}

static int nca_diagnostics(void *ctx, char *buffer, size_t buffer_len) {
    tfrl_nca_algo *algo = (tfrl_nca_algo *)ctx;
    if (!algo || !buffer || buffer_len == 0) return 0;
    snprintf(buffer, buffer_len,
             "%s version=%d cells=%d rollout=%d updates=%lld td=%.5f q=%.5f eps=%.3f clip=%.3f",
             algo->ppo_mode ? "nca-ppo-experimental" : "nca",
             TFRL_NCA_VERSION, algo->cell_count, algo->rollout_steps,
             algo->updates, algo->last_td, algo->last_q, algo->epsilon, algo->clip_eps);
    return 1;
}

static const tfrl_algo_vtable NCA_VTABLE = {
    .act = nca_act,
    .act_env = nca_act_env,
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
    algo->ppo_mode = cfg->name && strcmp(cfg->name, "nca-ppo") == 0;
    algo->clip_eps = cfg->clip_eps > 0.0f ? cfg->clip_eps : 0.2f;
    algo->entropy_coef = cfg->entropy_coef > 0.0f ? cfg->entropy_coef : 0.01f;
    algo->ppo_batch_size = cfg->steps_per_batch > 0 ? cfg->steps_per_batch : 32;
    algo->ppo_epochs = cfg->epochs > 0 ? cfg->epochs : 4;
    if (algo->ppo_batch_size > 256) algo->ppo_batch_size = 256;
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
    if (algo->ppo_mode) {
        algo->ppo_batch = (tfrl_transition *)calloc((size_t)algo->ppo_batch_size, sizeof(tfrl_transition));
        algo->ppo_old_logprobs = (float *)calloc((size_t)algo->ppo_batch_size, sizeof(float));
    }
    if (!algo->readout || !algo->bias || !algo->value ||
        (algo->ppo_mode && (!algo->ppo_batch || !algo->ppo_old_logprobs))) {
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

    if (algo->ppo_mode) {
        algo->population_size = 4;
        algo->population_episodes = 8;
        algo->population_current = 0;
        algo->population_best = 0;
        algo->population_best_return = -1.0e30f;
        algo->population_readout = calloc((size_t)algo->population_size * readout_count, sizeof(float));
        algo->population_bias = calloc((size_t)algo->population_size * (size_t)algo->action_n, sizeof(float));
        algo->population_scores = calloc((size_t)algo->population_size, sizeof(float));
        if (!algo->population_readout || !algo->population_bias || !algo->population_scores) {
            nca_destroy(algo);
            return out;
        }
        for (int p = 0; p < algo->population_size; p++) {
            for (size_t i = 0; i < readout_count; i++) algo->readout[i] += p == 0 ? 0.0f : 0.02f * nca_rand_signed(algo);
            for (int i = 0; i < algo->action_n; i++) algo->bias[i] = 0.02f * nca_rand_signed(algo);
            nca_population_store(algo, p);
            algo->population_scores[p] = -1.0e30f;
        }
        nca_population_load(algo, 0);
    }

    out.ctx = algo;
    out.vtable = &NCA_VTABLE;
    return out;
}
