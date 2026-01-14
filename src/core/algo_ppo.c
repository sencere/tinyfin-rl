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
    int *actions;
    float *rewards;
    int *dones;
    float *old_logp;
    float *old_values;
    float *old_next_values;
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
    Tensor *x = one_hot(algo->obs_n, obs.index);
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
    algo->actions[algo->count] = transition->action.index;
    algo->rewards[algo->count] = (float)transition->reward;
    algo->dones[algo->count] = transition->done;

    Tensor *x = one_hot(algo->obs_n, transition->obs.index);
    Tensor *logits = linear_forward(algo->policy, x);
    Tensor *probs = tensor_softmax_autograd(logits);
    Tensor *logp = tensor_log(probs);
    float lp = tensor_get_f32_at(logp, (size_t)transition->action.index);
    algo->old_logp[algo->count] = lp;
    tensor_free(logp);
    tensor_free(probs);
    tensor_free(logits);
    tensor_free(x);

    Tensor *vx = one_hot(algo->obs_n, transition->obs.index);
    Tensor *v = linear_forward(algo->value, vx);
    algo->old_values[algo->count] = v ? tensor_get_f32_at(v, 0) : 0.0f;
    tensor_free(v);
    tensor_free(vx);
    float next_v = 0.0f;
    if (!transition->done) {
        Tensor *vx_next = one_hot(algo->obs_n, transition->next_obs.index);
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

    float mean = 0.0f;
    float gae = 0.0f;
    for (int i = algo->batch - 1; i >= 0; i--) {
        float not_done = algo->dones[i] ? 0.0f : 1.0f;
        float delta = algo->rewards[i] + algo->gamma * next_values[i] * not_done - values[i];
        gae = delta + algo->gamma * algo->gae_lambda * not_done * gae;
        advantages[i] = gae;
        returns[i] = advantages[i] + values[i];
        mean += advantages[i];
    }

    mean /= (float)algo->batch;
    float var = 0.0f;
    for (int i = 0; i < algo->batch; i++) {
        float d = advantages[i] - mean;
        var += d * d;
    }
    float std = sqrtf(var / (float)algo->batch + 1e-8f);
    for (int i = 0; i < algo->batch; i++) {
        advantages[i] = (advantages[i] - mean) / std;
    }

    int epochs = algo->epochs > 0 ? algo->epochs : 2;
    int stop_early = 0;
    for (int e = 0; e < epochs && !stop_early; e++) {
        float kl_sum = 0.0f;
        int kl_count = 0;
        for (int i = 0; i < algo->batch; i++) {
            Tensor *x_i = one_hot(algo->obs_n, algo->obs_idx[i]);
            Tensor *logits_i = linear_forward(algo->policy, x_i);
            Tensor *probs_i = tensor_softmax_autograd(logits_i);
            Tensor *logp_i = tensor_log(probs_i);
            Tensor *action_one = one_hot(algo->action_n, algo->actions[i]);
            Tensor *logp_sel = tensor_mul(logp_i, action_one);
            Tensor *logp = tensor_sum(logp_sel);
            float new_lp = tensor_get_f32_at(logp, 0);
            kl_sum += (algo->old_logp[i] - new_lp);
            kl_count += 1;
            Tensor *old_lp_t = tensor_new(1, (int[1]){1});
            tensor_set_f32_at(old_lp_t, 0, algo->old_logp[i]);
            Tensor *ratio = tensor_exp(tensor_sub(logp, old_lp_t));
            Tensor *clipped = tensor_clamp(ratio, 1.0f - algo->clip_eps, 1.0f + algo->clip_eps);

            Tensor *adv_t = tensor_new(1, (int[1]){1});
            tensor_set_f32_at(adv_t, 0, advantages[i]);
            float ratio_v = tensor_get_f32_at(ratio, 0);
            float clipped_v = tensor_get_f32_at(clipped, 0);
            Tensor *ratio_used = NULL;
            if (advantages[i] >= 0.0f) {
                ratio_used = ratio_v < clipped_v ? ratio : clipped;
            } else {
                ratio_used = ratio_v > clipped_v ? ratio : clipped;
            }
            Tensor *policy_loss = tensor_mul(ratio_used, adv_t);
            Tensor *neg = tensor_new(1, (int[1]){1});
            tensor_set_f32_at(neg, 0, -1.0f);
            Tensor *policy_loss_neg = tensor_mul(policy_loss, neg);

            Tensor *v = linear_forward(algo->value, x_i);
            Tensor *target = tensor_new(1, (int[1]){1});
            tensor_set_f32_at(target, 0, returns[i]);
            Tensor *vdiff = tensor_sub(v, target);
            Tensor *value_loss = tensor_mul(vdiff, vdiff);
            Tensor *old_v = tensor_new(1, (int[1]){1});
            tensor_set_f32_at(old_v, 0, algo->old_values[i]);
            Tensor *v_delta = tensor_sub(v, old_v);
            Tensor *v_delta_clip = tensor_clamp(v_delta, -algo->clip_eps, algo->clip_eps);
            Tensor *v_clipped = tensor_add(old_v, v_delta_clip);
            Tensor *vdiff_clip = tensor_sub(v_clipped, target);
            Tensor *value_loss_clip = tensor_mul(vdiff_clip, vdiff_clip);
            float loss_unclipped = tensor_get_f32_at(value_loss, 0);
            float loss_clipped = tensor_get_f32_at(value_loss_clip, 0);
            Tensor *value_loss_used = loss_unclipped > loss_clipped ? value_loss : value_loss_clip;
            Tensor *value_coef = tensor_new(1, (int[1]){1});
            tensor_set_f32_at(value_coef, 0, PPO_VALUE_COEF);
            Tensor *value_loss_scaled = tensor_mul(value_loss_used, value_coef);

            Tensor *total = tensor_add(policy_loss_neg, value_loss_scaled);
            if (algo->entropy_coef > 0.0f) {
                Tensor *entropy_raw = tensor_mul(probs_i, logp_i);
                Tensor *entropy_sum = tensor_sum(entropy_raw);
                Tensor *neg_one = tensor_new(1, (int[1]){1});
                tensor_set_f32_at(neg_one, 0, -1.0f);
                Tensor *entropy = tensor_mul(entropy_sum, neg_one);
                Tensor *coef = tensor_new(1, (int[1]){1});
                tensor_set_f32_at(coef, 0, -algo->entropy_coef);
                Tensor *entropy_term = tensor_mul(entropy, coef);
                Tensor *total_with_entropy = tensor_add(total, entropy_term);
                tensor_free(total);
                total = total_with_entropy;
                tensor_free(entropy_term);
                tensor_free(coef);
                tensor_free(entropy);
                tensor_free(neg_one);
                tensor_free(entropy_sum);
                tensor_free(entropy_raw);
            }

            adam_zero_grad(algo->opt);
            tensor_backward(total);
            adam_step(algo->opt, 0.0f);

            tensor_free(total);
            tensor_free(value_loss_scaled);
            tensor_free(value_coef);
            tensor_free(value_loss);
            tensor_free(value_loss_clip);
            tensor_free(vdiff_clip);
            tensor_free(v_clipped);
            tensor_free(v_delta_clip);
            tensor_free(v_delta);
            tensor_free(old_v);
            tensor_free(vdiff);
            tensor_free(target);
            tensor_free(v);
            tensor_free(policy_loss_neg);
            tensor_free(neg);
            tensor_free(policy_loss);
            tensor_free(adv_t);
            tensor_free(clipped);
            tensor_free(ratio);
            tensor_free(old_lp_t);
            tensor_free(logp);
            tensor_free(logp_sel);
            tensor_free(action_one);
            tensor_free(logp_i);
            tensor_free(probs_i);
            tensor_free(logits_i);
            tensor_free(x_i);
        }
        if (algo->kl_target > 0.0f && kl_count > 0) {
            float avg_kl = kl_sum / (float)kl_count;
            fprintf(stdout, "ppo kl: epoch=%d avg=%.6f target=%.6f\n", e, avg_kl, algo->kl_target);
            if (avg_kl > algo->kl_target) {
                fprintf(stdout, "ppo early stop: kl=%.6f target=%.6f\n", avg_kl, algo->kl_target);
                stop_early = 1;
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

static const tfrl_algo_vtable PPO_VTABLE = {
    .act = ppo_act,
    .act_batch = NULL,
    .update = ppo_update,
    .save = ppo_save,
    .destroy = ppo_destroy,
};

tfrl_algo tfrl_algo_ppo_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->obs_n <= 0 || cfg->action_n <= 0) return out;
    autograd_set_enabled(1);
    tfrl_ppo_algo *algo = (tfrl_ppo_algo *)calloc(1, sizeof(tfrl_ppo_algo));
    if (!algo) return out;
    algo->obs_n = cfg->obs_n;
    algo->action_n = cfg->action_n;
    algo->gamma = cfg->gamma > 0.0f ? cfg->gamma : 0.99f;
    algo->clip_eps = cfg->clip_eps > 0.0f ? cfg->clip_eps : 0.2f;
    algo->entropy_coef = cfg->entropy_coef > 0.0f ? cfg->entropy_coef : 0.0f;
    algo->gae_lambda = cfg->gae_lambda > 0.0f ? cfg->gae_lambda : 0.95f;
    algo->kl_target = cfg->kl_target > 0.0f ? cfg->kl_target : 0.0f;
    algo->batch = cfg->steps_per_batch > 0 ? cfg->steps_per_batch : 64;
    algo->epochs = cfg->epochs > 0 ? cfg->epochs : 2;
    algo->policy = linear_create(algo->obs_n, algo->action_n);
    algo->value = linear_create(algo->obs_n, 1);
    if (!algo->policy || !algo->value) {
        ppo_destroy(algo);
        return out;
    }
    if (cfg->load_path) {
        load_linear(algo->policy, cfg->load_path, "pi");
        load_linear(algo->value, cfg->load_path, "v");
    }
    tensor_set_requires_grad(algo->policy->weight, 1);
    tensor_set_requires_grad(algo->policy->bias, 1);
    tensor_set_requires_grad(algo->value->weight, 1);
    tensor_set_requires_grad(algo->value->bias, 1);
    Tensor *params[4] = {algo->policy->weight, algo->policy->bias, algo->value->weight, algo->value->bias};
    algo->opt = adam_create(params, 4, cfg->lr > 0.0f ? cfg->lr : 0.0003f, 0.9f, 0.999f, 1e-8f, 0.0f);
    if (!algo->opt) {
        ppo_destroy(algo);
        return out;
    }
    algo->obs_idx = (int *)calloc((size_t)algo->batch, sizeof(int));
    algo->next_obs_idx = (int *)calloc((size_t)algo->batch, sizeof(int));
    algo->actions = (int *)calloc((size_t)algo->batch, sizeof(int));
    algo->rewards = (float *)calloc((size_t)algo->batch, sizeof(float));
    algo->dones = (int *)calloc((size_t)algo->batch, sizeof(int));
    algo->old_logp = (float *)calloc((size_t)algo->batch, sizeof(float));
    algo->old_values = (float *)calloc((size_t)algo->batch, sizeof(float));
    algo->old_next_values = (float *)calloc((size_t)algo->batch, sizeof(float));
    if (!algo->obs_idx || !algo->next_obs_idx || !algo->actions || !algo->rewards || !algo->dones || !algo->old_logp || !algo->old_values || !algo->old_next_values) {
        ppo_destroy(algo);
        return out;
    }
    out.ctx = algo;
    out.vtable = &PPO_VTABLE;
    return out;
}
