#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyfin/autograd.h"
#include "tinyfin/nn.h"
#include "tinyfin/optim.h"
#include "tinyfin/ops_add.h"
#include "tinyfin/ops_log.h"
#include "tinyfin/ops_mul.h"
#include "tinyfin/ops_reduce.h"
#include "tinyfin/ops_softmax.h"
#include "tinyfin/ops_sub.h"
#include "tinyfin/tensor.h"

#include "algo_api.h"

typedef struct {
    Linear *policy;
    Linear *value;
    Adam *opt;
    int obs_n;
    int action_n;
    float gamma;
    float entropy_coef;
    int batch;
    int count;
    int *obs_idx;
    int *actions;
    float *rewards;
    int *dones;
    pthread_mutex_t lock;
} tfrl_a3c_algo;

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

static tfrl_action a3c_act(void *ctx, tfrl_obs obs) {
    tfrl_a3c_algo *algo = (tfrl_a3c_algo *)ctx;
    tfrl_action action = {0};
    pthread_mutex_lock(&algo->lock);
    Tensor *x = one_hot(algo->obs_n, obs.index);
    Tensor *logits = linear_forward(algo->policy, x);
    Tensor *probs = tensor_softmax_autograd(logits);
    action.index = sample_action(probs);
    tensor_free(probs);
    tensor_free(logits);
    tensor_free(x);
    pthread_mutex_unlock(&algo->lock);
    return action;
}

static void a3c_update(void *ctx, const tfrl_transition *transition) {
    tfrl_a3c_algo *algo = (tfrl_a3c_algo *)ctx;
    if (!transition) return;
    pthread_mutex_lock(&algo->lock);
    if (algo->count >= algo->batch) {
        pthread_mutex_unlock(&algo->lock);
        return;
    }
    algo->obs_idx[algo->count] = transition->obs.index;
    algo->actions[algo->count] = transition->action.index;
    algo->rewards[algo->count] = (float)transition->reward;
    algo->dones[algo->count] = transition->done;
    algo->count++;

    if (algo->count < algo->batch && !transition->done) {
        pthread_mutex_unlock(&algo->lock);
        return;
    }

    int n = algo->count;
    float *returns = (float *)calloc((size_t)n, sizeof(float));
    if (!returns) {
        algo->count = 0;
        pthread_mutex_unlock(&algo->lock);
        return;
    }
    float g = 0.0f;
    for (int i = n - 1; i >= 0; i--) {
        if (algo->dones[i]) g = 0.0f;
        g = algo->rewards[i] + algo->gamma * g;
        returns[i] = g;
    }

    adam_zero_grad(algo->opt);
    for (int i = 0; i < n; i++) {
        Tensor *x = one_hot(algo->obs_n, algo->obs_idx[i]);
        Tensor *logits = linear_forward(algo->policy, x);
        Tensor *probs = tensor_softmax_autograd(logits);
        Tensor *logp = tensor_log(probs);
        Tensor *action_one = one_hot(algo->action_n, algo->actions[i]);
        Tensor *logp_sel = tensor_mul(logp, action_one);
        Tensor *logp_i = tensor_sum(logp_sel);

        Tensor *v = linear_forward(algo->value, x);
        float v_scalar = tensor_get_f32_at(v, 0);
        Tensor *adv_t = tensor_new(1, (int[1]){1});
        tensor_set_f32_at(adv_t, 0, returns[i] - v_scalar);

        Tensor *policy_loss = tensor_mul(logp_i, adv_t);
        Tensor *neg = tensor_new(1, (int[1]){1});
        tensor_set_f32_at(neg, 0, -1.0f);
        Tensor *policy_loss_neg = tensor_mul(policy_loss, neg);

        Tensor *ret_t = tensor_new(1, (int[1]){1});
        tensor_set_f32_at(ret_t, 0, returns[i]);
        Tensor *vdiff = tensor_sub(v, ret_t);
        Tensor *value_loss = tensor_mul(vdiff, vdiff);

        Tensor *entropy = NULL;
        if (algo->entropy_coef > 0.0f) {
            Tensor *logp_probs = tensor_mul(probs, logp);
            Tensor *ent_sum = tensor_sum(logp_probs);
            Tensor *entropy_scale = tensor_new(1, (int[1]){1});
            tensor_set_f32_at(entropy_scale, 0, -algo->entropy_coef);
            entropy = tensor_mul(ent_sum, entropy_scale);
            tensor_free(entropy_scale);
            tensor_free(ent_sum);
            tensor_free(logp_probs);
        }

        Tensor *loss = tensor_add(policy_loss_neg, value_loss);
        if (entropy) {
            Tensor *tmp = tensor_add(loss, entropy);
            tensor_free(loss);
            loss = tmp;
            tensor_free(entropy);
        }

        tensor_backward(loss);
        tensor_free(loss);

        tensor_free(value_loss);
        tensor_free(vdiff);
        tensor_free(ret_t);
        tensor_free(policy_loss_neg);
        tensor_free(neg);
        tensor_free(policy_loss);
        tensor_free(adv_t);
        tensor_free(v);
        tensor_free(logp_i);
        tensor_free(logp_sel);
        tensor_free(action_one);
        tensor_free(logp);
        tensor_free(probs);
        tensor_free(logits);
        tensor_free(x);
    }
    adam_step(algo->opt, 0.0f);

    free(returns);
    algo->count = 0;
    pthread_mutex_unlock(&algo->lock);
}

static void a3c_destroy(void *ctx) {
    tfrl_a3c_algo *algo = (tfrl_a3c_algo *)ctx;
    if (!algo) return;
    adam_free(algo->opt);
    tensor_free(algo->policy->weight);
    tensor_free(algo->policy->bias);
    tensor_free(algo->value->weight);
    tensor_free(algo->value->bias);
    free(algo->policy);
    free(algo->value);
    free(algo->obs_idx);
    free(algo->actions);
    free(algo->rewards);
    free(algo->dones);
    pthread_mutex_destroy(&algo->lock);
    free(algo);
}

static void a3c_save(void *ctx, const char *path) {
    tfrl_a3c_algo *algo = (tfrl_a3c_algo *)ctx;
    if (!algo) return;
    save_linear(algo->policy, path, "pi");
    save_linear(algo->value, path, "v");
}

static const tfrl_algo_vtable A3C_VTABLE = {
    .act = a3c_act,
    .act_batch = NULL,
    .update = a3c_update,
    .save = a3c_save,
    .destroy = a3c_destroy,
};

tfrl_algo tfrl_algo_a3c_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->obs_n <= 0 || cfg->action_n <= 0) return out;
    autograd_set_enabled(1);
    tfrl_a3c_algo *algo = (tfrl_a3c_algo *)calloc(1, sizeof(tfrl_a3c_algo));
    if (!algo) return out;
    algo->obs_n = cfg->obs_n;
    algo->action_n = cfg->action_n;
    algo->gamma = cfg->gamma > 0.0f ? cfg->gamma : 0.99f;
    algo->entropy_coef = cfg->entropy_coef > 0.0f ? cfg->entropy_coef : 0.0f;
    algo->batch = cfg->steps_per_batch > 0 ? cfg->steps_per_batch : 64;
    algo->policy = linear_create(algo->obs_n, algo->action_n);
    algo->value = linear_create(algo->obs_n, 1);
    if (!algo->policy || !algo->value) {
        a3c_destroy(algo);
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
    algo->opt = adam_create(params, 4, cfg->lr > 0.0f ? cfg->lr : 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);
    if (!algo->opt) {
        a3c_destroy(algo);
        return out;
    }
    algo->obs_idx = (int *)calloc((size_t)algo->batch, sizeof(int));
    algo->actions = (int *)calloc((size_t)algo->batch, sizeof(int));
    algo->rewards = (float *)calloc((size_t)algo->batch, sizeof(float));
    algo->dones = (int *)calloc((size_t)algo->batch, sizeof(int));
    if (!algo->obs_idx || !algo->actions || !algo->rewards || !algo->dones) {
        a3c_destroy(algo);
        return out;
    }
    pthread_mutex_init(&algo->lock, NULL);
    out.ctx = algo;
    out.vtable = &A3C_VTABLE;
    return out;
}
