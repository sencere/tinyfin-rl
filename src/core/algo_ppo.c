#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyfin/autograd.h"
#include "tinyfin/nn.h"
#include "tinyfin/optim.h"
#include "tinyfin/ops_add.h"
#include "tinyfin/ops_clamp.h"
#include "tinyfin/ops_exp.h"
#include "tinyfin/ops_log.h"
#include "tinyfin/ops_mul.h"
#include "tinyfin/ops_reduce.h"
#include "tinyfin/ops_softmax.h"
#include "tinyfin/ops_sub.h"
#include "tinyfin/tensor.h"

#include "algo_api.h"

#define PPO_VALUE_COEF 0.5f

typedef struct {
    Linear *policy;
    Linear *value;
    Adam *opt;
    int obs_n;
    int obs_dim;
    tfrl_space_type obs_type;
    int action_n;
    float gamma;
    float clip_eps;
    float entropy_coef;
    float gae_lambda;
    float kl_target;
    int batch;
    int epochs;
    int count;
    int *obs_idx;
    int *next_obs_idx;
    float *obs_buf;
    float *next_obs_buf;
    int *actions;
    float *rewards;
    int *dones;
    float *old_logp;
    float *old_values;
    float *old_next_values;
    double kl_sum;
    double kl_sum_sq;
    float kl_min;
    float kl_max;
    float kl_last;
    int kl_updates;
} tfrl_ppo_algo;

static Tensor *one_hot(int n, int idx) {
    int shape[2] = {1, n};
    Tensor *t = tensor_zeros(2, shape);
    if (!t) return NULL;
    if (idx >= 0 && idx < n) {
        tensor_set_f32_at(t, (size_t)idx, 1.0f);
    }
    return t;
}

static void init_linear(Linear *layer) {
    if (!layer) return;
    tensor_set_requires_grad(layer->weight, 1);
    tensor_set_requires_grad(layer->bias, 1);
    float scale = 0.1f;
    for (size_t i = 0; i < layer->weight->size; i++) {
        float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        tensor_set_f32_at(layer->weight, i, r * scale);
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        tensor_set_f32_at(layer->bias, i, 0.0f);
    }
}

static Tensor *obs_to_tensor(const tfrl_ppo_algo *algo, const tfrl_obs *obs) {
    if (!algo || !obs) return NULL;
    if (algo->obs_type == TFRL_SPACE_BOX) {
        int shape[2] = {1, algo->obs_dim};
        Tensor *t = tensor_zeros(2, shape);
        if (!t) return NULL;
        int n = obs->data_len < algo->obs_dim ? obs->data_len : algo->obs_dim;
        for (int i = 0; i < n; i++) {
            tensor_set_f32_at(t, (size_t)i, obs->data[i]);
        }
        return t;
    }
    return one_hot(algo->obs_n, obs->index);
}

static Tensor *obs_from_buffer(const tfrl_ppo_algo *algo, const float *buf, int idx) {
    if (!algo || !buf || algo->obs_dim <= 0) return NULL;
    int shape[2] = {1, algo->obs_dim};
    Tensor *t = tensor_zeros(2, shape);
    if (!t) return NULL;
    const float *src = &buf[(size_t)idx * (size_t)algo->obs_dim];
    for (int i = 0; i < algo->obs_dim; i++) {
        tensor_set_f32_at(t, (size_t)i, src[i]);
    }
    return t;
}

static int sample_action(Tensor *probs) {
    float r = (float)rand() / (float)RAND_MAX;
    float acc = 0.0f;
    for (size_t i = 0; i < probs->size; i++) {
        acc += tensor_get_f32_at(probs, i);
        if (r <= acc) return (int)i;
    }
    return (int)(probs->size > 0 ? probs->size - 1 : 0);
}

static int save_linear(Linear *layer, const char *prefix, const char *tag) {
    if (!prefix || !tag) return 0;
    char path_w[256];
    char path_b[256];
    snprintf(path_w, sizeof(path_w), "%s.%s.w.tensor", prefix, tag);
    snprintf(path_b, sizeof(path_b), "%s.%s.b.tensor", prefix, tag);
    if (!tensor_save(layer->weight, path_w)) return 0;
    if (!tensor_save(layer->bias, path_b)) return 0;
    return 1;
}

static int load_linear(Linear *layer, const char *prefix, const char *tag) {
    if (!prefix || !tag) return 0;
    char path_w[256];
    char path_b[256];
    snprintf(path_w, sizeof(path_w), "%s.%s.w.tensor", prefix, tag);
    snprintf(path_b, sizeof(path_b), "%s.%s.b.tensor", prefix, tag);
    Tensor *w = tensor_load(path_w);
    Tensor *b = tensor_load(path_b);
    if (!w || !b) {
        if (w) tensor_free(w);
        if (b) tensor_free(b);
        return 0;
    }
    if (w->size == layer->weight->size) {
        memcpy(layer->weight->data, w->data, w->size * sizeof(float));
    }
    if (b->size == layer->bias->size) {
        memcpy(layer->bias->data, b->data, b->size * sizeof(float));
    }
    tensor_free(w);
    tensor_free(b);
    return 1;
}

static tfrl_action ppo_act(void *ctx, tfrl_obs obs) {
    tfrl_ppo_algo *algo = (tfrl_ppo_algo *)ctx;
    tfrl_action action = {0};
    Tensor *x = obs_to_tensor(algo, &obs);
    Tensor *logits = linear_forward(algo->policy, x);
    Tensor *probs = tensor_softmax_autograd(logits);
    action.index = sample_action(probs);
    tensor_free(probs);
    tensor_free(logits);
    tensor_free(x);
    return action;
}

static void ppo_update(void *ctx, const tfrl_transition *transition) {
    tfrl_ppo_algo *algo = (tfrl_ppo_algo *)ctx;
    if (!transition) return;
    if (algo->count >= algo->batch) return;
    algo->obs_idx[algo->count] = transition->obs.index;
    algo->next_obs_idx[algo->count] = transition->next_obs.index;
    if (algo->obs_type == TFRL_SPACE_BOX && algo->obs_buf && algo->next_obs_buf) {
        float *dst = &algo->obs_buf[(size_t)algo->count * (size_t)algo->obs_dim];
        float *dst_next = &algo->next_obs_buf[(size_t)algo->count * (size_t)algo->obs_dim];
        int n = transition->obs.data_len < algo->obs_dim ? transition->obs.data_len : algo->obs_dim;
        int n_next = transition->next_obs.data_len < algo->obs_dim ? transition->next_obs.data_len : algo->obs_dim;
        for (int i = 0; i < algo->obs_dim; i++) {
            dst[i] = (i < n) ? transition->obs.data[i] : 0.0f;
            dst_next[i] = (i < n_next) ? transition->next_obs.data[i] : 0.0f;
        }
    }
    algo->actions[algo->count] = transition->action.index;
    algo->rewards[algo->count] = (float)transition->reward;
    algo->dones[algo->count] = transition->done;

    Tensor *x = obs_to_tensor(algo, &transition->obs);
    Tensor *logits = linear_forward(algo->policy, x);
    Tensor *probs = tensor_softmax_autograd(logits);
    Tensor *logp = tensor_log(probs);
    float lp = tensor_get_f32_at(logp, (size_t)transition->action.index);
    algo->old_logp[algo->count] = lp;
    tensor_free(logp);
    tensor_free(probs);
    tensor_free(logits);
    tensor_free(x);

    Tensor *vx = obs_to_tensor(algo, &transition->obs);
    Tensor *v = linear_forward(algo->value, vx);
    algo->old_values[algo->count] = v ? tensor_get_f32_at(v, 0) : 0.0f;
    tensor_free(v);
    tensor_free(vx);
    float next_v = 0.0f;
    if (!transition->done) {
        Tensor *vx_next = obs_to_tensor(algo, &transition->next_obs);
        Tensor *v_next = linear_forward(algo->value, vx_next);
        next_v = v_next ? tensor_get_f32_at(v_next, 0) : 0.0f;
        tensor_free(v_next);
        tensor_free(vx_next);
    }
    algo->old_next_values[algo->count] = next_v;

    algo->count++;
    if (algo->count < algo->batch) return;

    float *returns = (float *)calloc((size_t)algo->batch, sizeof(float));
    float *advantages = (float *)calloc((size_t)algo->batch, sizeof(float));
    float *values = (float *)calloc((size_t)algo->batch, sizeof(float));
    float *next_values = (float *)calloc((size_t)algo->batch, sizeof(float));
    if (!returns || !advantages || !values || !next_values) {
        free(returns);
        free(advantages);
        free(values);
        free(next_values);
        return;
    }

    for (int i = 0; i < algo->batch; i++) {
        values[i] = algo->old_values[i];
        next_values[i] = algo->old_next_values[i];
    }

    float gae = 0.0f;
    for (int i = algo->batch - 1; i >= 0; i--) {
        float not_done = algo->dones[i] ? 0.0f : 1.0f;
        float delta = algo->rewards[i] + algo->gamma * next_values[i] * not_done - values[i];
        gae = delta + algo->gamma * algo->gae_lambda * not_done * gae;
        advantages[i] = gae;
        returns[i] = advantages[i] + values[i];
    }

    // Advantage normalization can collapse to ~0 variance in these tiny envs,
    // which effectively zeroes the policy gradient. Skip it for now.

    int epochs = algo->epochs > 0 ? algo->epochs : 2;
    int stop_early = 0;
    for (int e = 0; e < epochs && !stop_early; e++) {
        float kl_sum = 0.0f;
        int kl_count = 0;
        float policy_loss_sum_f = 0.0f;
        float value_loss_sum_f = 0.0f;

        int trash_cap = algo->batch * 64;
        if (trash_cap < 1024) trash_cap = 1024;
        Tensor **trash = (Tensor **)calloc((size_t)trash_cap, sizeof(Tensor *));
        int trash_n = 0;
        #define TRACK(t) do { \
            if (t) { \
                if (trash_n >= trash_cap) { \
                    int new_cap = trash_cap * 2; \
                    Tensor **grown = (Tensor **)realloc(trash, (size_t)new_cap * sizeof(Tensor *)); \
                    if (grown) { \
                        memset(grown + trash_cap, 0, (size_t)(new_cap - trash_cap) * sizeof(Tensor *)); \
                        trash = grown; \
                        trash_cap = new_cap; \
                    } \
                } \
                if (trash_n < trash_cap) trash[trash_n++] = (t); \
            } \
        } while (0)

        Tensor *total_batch = NULL;
        adam_zero_grad(algo->opt);
        for (int i = 0; i < algo->batch; i++) {
            Tensor *x_i = NULL;
            if (algo->obs_type == TFRL_SPACE_BOX && algo->obs_buf) {
                x_i = obs_from_buffer(algo, algo->obs_buf, i);
            } else {
                x_i = one_hot(algo->obs_n, algo->obs_idx[i]);
            }
            TRACK(x_i);
            Tensor *logits_i = linear_forward(algo->policy, x_i); TRACK(logits_i);
            Tensor *probs_i = tensor_softmax_autograd(logits_i); TRACK(probs_i);
            Tensor *logp_i = tensor_log(probs_i); TRACK(logp_i);
            Tensor *action_one = one_hot(algo->action_n, algo->actions[i]); TRACK(action_one);
            Tensor *logp_sel = tensor_mul(logp_i, action_one); TRACK(logp_sel);
            Tensor *logp = tensor_sum(logp_sel); TRACK(logp);
            float new_lp = tensor_get_f32_at(logp, 0);
            kl_sum += (algo->old_logp[i] - new_lp);
            kl_count += 1;
            Tensor *old_lp_t = tensor_new(1, (int[1]){1}); TRACK(old_lp_t);
            tensor_set_f32_at(old_lp_t, 0, algo->old_logp[i]);
            Tensor *ratio = tensor_exp(tensor_sub(logp, old_lp_t)); TRACK(ratio);
            Tensor *clipped = tensor_clamp(ratio, 1.0f - algo->clip_eps, 1.0f + algo->clip_eps); TRACK(clipped);

            Tensor *adv_t = tensor_new(1, (int[1]){1}); TRACK(adv_t);
            tensor_set_f32_at(adv_t, 0, advantages[i]);
            float ratio_v = tensor_get_f32_at(ratio, 0);
            float clipped_v = tensor_get_f32_at(clipped, 0);
            Tensor *ratio_used = NULL;
            if (advantages[i] >= 0.0f) {
                ratio_used = ratio_v < clipped_v ? ratio : clipped;
            } else {
                ratio_used = ratio_v > clipped_v ? ratio : clipped;
            }
            Tensor *policy_loss = tensor_mul(ratio_used, adv_t); TRACK(policy_loss);
            Tensor *neg = tensor_new(1, (int[1]){1}); TRACK(neg);
            tensor_set_f32_at(neg, 0, -1.0f);
            Tensor *policy_loss_neg = tensor_mul(policy_loss, neg); TRACK(policy_loss_neg);

            Tensor *v = linear_forward(algo->value, x_i); TRACK(v);
            Tensor *target = tensor_new(1, (int[1]){1}); TRACK(target);
            tensor_set_f32_at(target, 0, returns[i]);
            Tensor *vdiff = tensor_sub(v, target); TRACK(vdiff);
            Tensor *value_loss = tensor_mul(vdiff, vdiff); TRACK(value_loss);
            Tensor *old_v = tensor_new(1, (int[1]){1}); TRACK(old_v);
            tensor_set_f32_at(old_v, 0, algo->old_values[i]);
            Tensor *v_delta = tensor_sub(v, old_v); TRACK(v_delta);
            Tensor *v_delta_clip = tensor_clamp(v_delta, -algo->clip_eps, algo->clip_eps); TRACK(v_delta_clip);
            Tensor *v_clipped = tensor_add(old_v, v_delta_clip); TRACK(v_clipped);
            Tensor *vdiff_clip = tensor_sub(v_clipped, target); TRACK(vdiff_clip);
            Tensor *value_loss_clip = tensor_mul(vdiff_clip, vdiff_clip); TRACK(value_loss_clip);
            float loss_unclipped = tensor_get_f32_at(value_loss, 0);
            float loss_clipped = tensor_get_f32_at(value_loss_clip, 0);
            Tensor *value_loss_used = loss_unclipped > loss_clipped ? value_loss : value_loss_clip;
            Tensor *value_coef = tensor_new(1, (int[1]){1}); TRACK(value_coef);
            tensor_set_f32_at(value_coef, 0, PPO_VALUE_COEF);
            Tensor *value_loss_scaled = tensor_mul(value_loss_used, value_coef); TRACK(value_loss_scaled);

            Tensor *total_i = tensor_add(policy_loss_neg, value_loss_scaled); TRACK(total_i);
            if (algo->entropy_coef > 0.0f) {
                Tensor *entropy_raw = tensor_mul(probs_i, logp_i); TRACK(entropy_raw);
                Tensor *entropy_sum = tensor_sum(entropy_raw); TRACK(entropy_sum);
                Tensor *neg_one = tensor_new(1, (int[1]){1}); TRACK(neg_one);
                tensor_set_f32_at(neg_one, 0, -1.0f);
                Tensor *entropy = tensor_mul(entropy_sum, neg_one); TRACK(entropy);
                Tensor *coef = tensor_new(1, (int[1]){1}); TRACK(coef);
                tensor_set_f32_at(coef, 0, -algo->entropy_coef);
                Tensor *entropy_term = tensor_mul(entropy, coef); TRACK(entropy_term);
                total_i = tensor_add(total_i, entropy_term); TRACK(total_i);
            }

            policy_loss_sum_f += tensor_get_f32_at(policy_loss_neg, 0);
            value_loss_sum_f += tensor_get_f32_at(value_loss_scaled, 0);

            if (!total_batch) {
                total_batch = total_i;
            } else {
                total_batch = tensor_add(total_batch, total_i);
                TRACK(total_batch);
            }
        }

        if (total_batch) {
            tensor_backward(total_batch);
            adam_step(algo->opt, 0.0f);
        }

        for (int ti = 0; ti < trash_n; ti++) {
            tensor_free(trash[ti]);
        }
        free(trash);
        #undef TRACK
        if (kl_count > 0) {
            float avg_kl = kl_sum / (float)kl_count;
            algo->kl_last = avg_kl;
            algo->kl_sum += avg_kl;
            algo->kl_sum_sq += (double)avg_kl * (double)avg_kl;
            algo->kl_updates += 1;
            if (algo->kl_updates == 1) {
                algo->kl_min = avg_kl;
                algo->kl_max = avg_kl;
            } else {
                if (avg_kl < algo->kl_min) algo->kl_min = avg_kl;
                if (avg_kl > algo->kl_max) algo->kl_max = avg_kl;
            }
            if (algo->kl_target > 0.0f) {
                if (avg_kl > algo->kl_target) {
                    stop_early = 1;
                }
            }
        }
    }

    free(returns);
    free(advantages);
    free(values);
    free(next_values);
    algo->count = 0;
}

static void ppo_destroy(void *ctx) {
    tfrl_ppo_algo *algo = (tfrl_ppo_algo *)ctx;
    if (!algo) return;
    adam_free(algo->opt);
    tensor_free(algo->policy->weight);
    tensor_free(algo->policy->bias);
    tensor_free(algo->value->weight);
    tensor_free(algo->value->bias);
    free(algo->policy);
    free(algo->value);
    free(algo->obs_idx);
    free(algo->next_obs_idx);
    free(algo->obs_buf);
    free(algo->next_obs_buf);
    free(algo->actions);
    free(algo->rewards);
    free(algo->dones);
    free(algo->old_logp);
    free(algo->old_values);
    free(algo->old_next_values);
    free(algo);
}

static void ppo_save(void *ctx, const char *path) {
    tfrl_ppo_algo *algo = (tfrl_ppo_algo *)ctx;
    if (!algo) return;
    save_linear(algo->policy, path, "pi");
    save_linear(algo->value, path, "v");
}

static int ppo_diagnostics(void *ctx, char *buffer, size_t buffer_len) {
    tfrl_ppo_algo *algo = (tfrl_ppo_algo *)ctx;
    if (!algo || !buffer || buffer_len == 0) return 0;
    double mean = 0.0;
    double std = 0.0;
    float kl_min = 0.0f;
    float kl_max = 0.0f;
    float kl_last = 0.0f;
    int updates = algo->kl_updates;
    if (updates > 0) {
        mean = algo->kl_sum / (double)updates;
        double var = (algo->kl_sum_sq / (double)updates) - mean * mean;
        if (var < 0.0) var = 0.0;
        std = sqrt(var);
        kl_min = algo->kl_min;
        kl_max = algo->kl_max;
        kl_last = algo->kl_last;
    }
    int written = snprintf(buffer,
                           buffer_len,
                           "ppo_kl_mean=%+012.6f ppo_kl_std=%+012.6f ppo_kl_min=%+012.6f "
                           "ppo_kl_max=%+012.6f ppo_kl_last=%+012.6f ppo_kl_updates=%08d",
                           (float)mean,
                           (float)std,
                           kl_min,
                           kl_max,
                           kl_last,
                           updates);
    return written > 0 && (size_t)written < buffer_len;
}

static const tfrl_algo_vtable PPO_VTABLE = {
    .act = ppo_act,
    .act_batch = NULL,
    .update = ppo_update,
    .save = ppo_save,
    .destroy = ppo_destroy,
    .diagnostics = ppo_diagnostics,
};

tfrl_algo tfrl_algo_ppo_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->obs_n <= 0 || cfg->action_n <= 0) return out;
    autograd_set_enabled(1);
    tfrl_ppo_algo *algo = (tfrl_ppo_algo *)calloc(1, sizeof(tfrl_ppo_algo));
    if (!algo) return out;
    algo->obs_n = cfg->obs_n;
    algo->obs_type = cfg->obs_type;
    algo->obs_dim = cfg->obs_dims > 0 ? cfg->obs_dims : cfg->obs_n;
    algo->action_n = cfg->action_n;
    algo->gamma = cfg->gamma > 0.0f ? cfg->gamma : 0.99f;
    algo->clip_eps = cfg->clip_eps > 0.0f ? cfg->clip_eps : 0.2f;
    algo->entropy_coef = cfg->entropy_coef > 0.0f ? cfg->entropy_coef : 0.0f;
    algo->gae_lambda = cfg->gae_lambda > 0.0f ? cfg->gae_lambda : 0.95f;
    algo->kl_target = cfg->kl_target > 0.0f ? cfg->kl_target : 0.0f;
    algo->batch = cfg->steps_per_batch > 0 ? cfg->steps_per_batch : 64;
    algo->epochs = cfg->epochs > 0 ? cfg->epochs : 2;
    int input_dim = algo->obs_type == TFRL_SPACE_BOX ? algo->obs_dim : algo->obs_n;
    algo->policy = linear_create(input_dim, algo->action_n);
    algo->value = linear_create(input_dim, 1);
    if (!algo->policy || !algo->value) {
        ppo_destroy(algo);
        return out;
    }
    init_linear(algo->policy);
    init_linear(algo->value);
    if (cfg->load_path) {
        load_linear(algo->policy, cfg->load_path, "pi");
        load_linear(algo->value, cfg->load_path, "v");
    }
    Tensor *params[4] = {algo->policy->weight, algo->policy->bias, algo->value->weight, algo->value->bias};
    algo->opt = adam_create(params, 4, cfg->lr > 0.0f ? cfg->lr : 0.0003f, 0.9f, 0.999f, 1e-8f, 0.0f);
    if (!algo->opt) {
        ppo_destroy(algo);
        return out;
    }
    algo->obs_idx = (int *)calloc((size_t)algo->batch, sizeof(int));
    algo->next_obs_idx = (int *)calloc((size_t)algo->batch, sizeof(int));
    if (algo->obs_type == TFRL_SPACE_BOX && algo->obs_dim > 0) {
        algo->obs_buf = (float *)calloc((size_t)algo->batch * (size_t)algo->obs_dim, sizeof(float));
        algo->next_obs_buf = (float *)calloc((size_t)algo->batch * (size_t)algo->obs_dim, sizeof(float));
    }
    algo->actions = (int *)calloc((size_t)algo->batch, sizeof(int));
    algo->rewards = (float *)calloc((size_t)algo->batch, sizeof(float));
    algo->dones = (int *)calloc((size_t)algo->batch, sizeof(int));
    algo->old_logp = (float *)calloc((size_t)algo->batch, sizeof(float));
    algo->old_values = (float *)calloc((size_t)algo->batch, sizeof(float));
    algo->old_next_values = (float *)calloc((size_t)algo->batch, sizeof(float));
    int need_buf = (algo->obs_type == TFRL_SPACE_BOX);
    if (!algo->obs_idx || !algo->next_obs_idx || !algo->actions || !algo->rewards || !algo->dones ||
        !algo->old_logp || !algo->old_values || !algo->old_next_values ||
        (need_buf && (!algo->obs_buf || !algo->next_obs_buf))) {
        ppo_destroy(algo);
        return out;
    }
    out.ctx = algo;
    out.vtable = &PPO_VTABLE;
    return out;
}
