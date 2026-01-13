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
#include "tinyfin/tensor.h"

#include "algo_api.h"

typedef struct {
    Linear *policy;
    Adam *opt;
    int obs_n;
    int action_n;
    float gamma;
    int max_steps;
    int step_count;
    int *obs_idx;
    int *actions;
    float *rewards;
} tfrl_reinforce_algo;

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

static tfrl_action reinforce_act(void *ctx, tfrl_obs obs) {
    tfrl_reinforce_algo *algo = (tfrl_reinforce_algo *)ctx;
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

static void reinforce_update(void *ctx, const tfrl_transition *transition) {
    tfrl_reinforce_algo *algo = (tfrl_reinforce_algo *)ctx;
    if (!transition) return;
    if (algo->step_count >= algo->max_steps) return;
    algo->obs_idx[algo->step_count] = transition->obs.index;
    algo->actions[algo->step_count] = transition->action.index;
    algo->rewards[algo->step_count] = (float)transition->reward;
    algo->step_count++;

    if (!transition->done) return;

    float g = 0.0f;
    for (int i = algo->step_count - 1; i >= 0; i--) {
        g = algo->rewards[i] + algo->gamma * g;
        algo->rewards[i] = g;
    }

    Tensor *loss_sum = NULL;
    for (int i = 0; i < algo->step_count; i++) {
        Tensor *x = one_hot(algo->obs_n, algo->obs_idx[i]);
        Tensor *logits = linear_forward(algo->policy, x);
        Tensor *probs = tensor_softmax_autograd(logits);
        Tensor *logp = tensor_log(probs);
        Tensor *action_one = one_hot(algo->action_n, algo->actions[i]);
        Tensor *logp_sel = tensor_mul(logp, action_one);
        Tensor *logp_i = tensor_sum(logp_sel);
        Tensor *ret_t = tensor_new(1, (int[1]){1});
        tensor_set_f32_at(ret_t, 0, algo->rewards[i]);
        Tensor *loss = tensor_mul(logp_i, ret_t);
        Tensor *neg = tensor_new(1, (int[1]){1});
        tensor_set_f32_at(neg, 0, -1.0f);
        Tensor *loss_neg = tensor_mul(loss, neg);

        if (!loss_sum) {
            loss_sum = loss_neg;
        } else {
            Tensor *tmp = tensor_add(loss_sum, loss_neg);
            tensor_free(loss_sum);
            loss_sum = tmp;
        }

        tensor_free(neg);
        tensor_free(loss);
        tensor_free(ret_t);
        tensor_free(logp_i);
        tensor_free(logp_sel);
        tensor_free(action_one);
        tensor_free(logp);
        tensor_free(probs);
        tensor_free(logits);
        tensor_free(x);
    }

    if (loss_sum) {
        adam_zero_grad(algo->opt);
        tensor_backward(loss_sum);
        adam_step(algo->opt, 0.0f);
        tensor_free(loss_sum);
    }

    algo->step_count = 0;
}

static void reinforce_destroy(void *ctx) {
    tfrl_reinforce_algo *algo = (tfrl_reinforce_algo *)ctx;
    if (!algo) return;
    adam_free(algo->opt);
    tensor_free(algo->policy->weight);
    tensor_free(algo->policy->bias);
    free(algo->policy);
    free(algo->obs_idx);
    free(algo->actions);
    free(algo->rewards);
    free(algo);
}

static void reinforce_save(void *ctx, const char *path) {
    tfrl_reinforce_algo *algo = (tfrl_reinforce_algo *)ctx;
    if (!algo) return;
    save_linear(algo->policy, path, "pi");
}

static const tfrl_algo_vtable REINFORCE_VTABLE = {
    .act = reinforce_act,
    .update = reinforce_update,
    .save = reinforce_save,
    .destroy = reinforce_destroy,
};

tfrl_algo tfrl_algo_reinforce_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->obs_n <= 0 || cfg->action_n <= 0) return out;
    autograd_set_enabled(1);
    tfrl_reinforce_algo *algo = (tfrl_reinforce_algo *)calloc(1, sizeof(tfrl_reinforce_algo));
    if (!algo) return out;
    algo->obs_n = cfg->obs_n;
    algo->action_n = cfg->action_n;
    algo->gamma = cfg->gamma > 0.0f ? cfg->gamma : 0.99f;
    algo->max_steps = cfg->steps_per_batch > 0 ? cfg->steps_per_batch : 512;
    algo->policy = linear_create(algo->obs_n, algo->action_n);
    if (!algo->policy) {
        free(algo);
        return out;
    }
    if (cfg->load_path) {
        load_linear(algo->policy, cfg->load_path, "pi");
    }
    tensor_set_requires_grad(algo->policy->weight, 1);
    tensor_set_requires_grad(algo->policy->bias, 1);
    Tensor *params[2] = {algo->policy->weight, algo->policy->bias};
    algo->opt = adam_create(params, 2, cfg->lr > 0.0f ? cfg->lr : 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);
    if (!algo->opt) {
        tensor_free(algo->policy->weight);
        tensor_free(algo->policy->bias);
        free(algo->policy);
        free(algo);
        return out;
    }
    algo->obs_idx = (int *)calloc((size_t)algo->max_steps, sizeof(int));
    algo->actions = (int *)calloc((size_t)algo->max_steps, sizeof(int));
    algo->rewards = (float *)calloc((size_t)algo->max_steps, sizeof(float));
    if (!algo->obs_idx || !algo->actions || !algo->rewards) {
        reinforce_destroy(algo);
        return out;
    }
    out.ctx = algo;
    out.vtable = &REINFORCE_VTABLE;
    return out;
}
