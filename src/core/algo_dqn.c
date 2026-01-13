#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyfin/autograd.h"
#include "tinyfin/nn.h"
#include "tinyfin/optim.h"
#include "tinyfin/ops_add.h"
#include "tinyfin/ops_mul.h"
#include "tinyfin/ops_reduce.h"
#include "tinyfin/ops_sub.h"
#include "tinyfin/tensor.h"

#include "algo_api.h"

typedef struct {
    Linear *q;
    SGD *opt;
    int obs_n;
    int action_n;
    float gamma;
    float epsilon;
} tfrl_dqn_algo;

static Tensor *one_hot(int n, int idx) {
    int shape[2] = {1, n};
    Tensor *t = tensor_zeros(2, shape);
    if (!t) return NULL;
    if (idx >= 0 && idx < n) {
        tensor_set_f32_at(t, (size_t)idx, 1.0f);
    }
    return t;
}

static int argmax_tensor(Tensor *t) {
    int best = 0;
    float best_v = tensor_get_f32_at(t, 0);
    for (size_t i = 1; i < t->size; i++) {
        float v = tensor_get_f32_at(t, i);
        if (v > best_v) {
            best_v = v;
            best = (int)i;
        }
    }
    return best;
}

static void init_linear(Linear *layer) {
    if (!layer) return;
    tensor_set_requires_grad(layer->weight, 1);
    tensor_set_requires_grad(layer->bias, 1);
    float scale = 0.01f;
    for (size_t i = 0; i < layer->weight->size; i++) {
        float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        tensor_set_f32_at(layer->weight, i, r * scale);
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        tensor_set_f32_at(layer->bias, i, 0.0f);
    }
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

static tfrl_action dqn_act(void *ctx, tfrl_obs obs) {
    tfrl_dqn_algo *algo = (tfrl_dqn_algo *)ctx;
    tfrl_action action = {0};
    float r = (float)rand() / (float)RAND_MAX;
    if (r < algo->epsilon) {
        action.index = rand() % algo->action_n;
        return action;
    }
    Tensor *x = one_hot(algo->obs_n, obs.index);
    Tensor *q_values = linear_forward(algo->q, x);
    action.index = argmax_tensor(q_values);
    tensor_free(q_values);
    tensor_free(x);
    return action;
}

static void dqn_act_batch(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions) {
    tfrl_dqn_algo *algo = (tfrl_dqn_algo *)ctx;
    if (!algo || !obs || !out_actions || count <= 0) return;

    int shape[2] = {count, algo->obs_n};
    Tensor *x = tensor_zeros(2, shape);
    if (!x) return;
    for (int i = 0; i < count; i++) {
        if (obs[i].index >= 0 && obs[i].index < algo->obs_n) {
            size_t off = (size_t)i * (size_t)algo->obs_n + (size_t)obs[i].index;
            tensor_set_f32_at(x, off, 1.0f);
        }
    }

    Tensor *q_values = linear_forward(algo->q, x);
    for (int i = 0; i < count; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        if (r < algo->epsilon) {
            out_actions[i].index = rand() % algo->action_n;
            continue;
        }
        int best = 0;
        float best_v = tensor_get_f32_at(q_values, (size_t)i * (size_t)algo->action_n);
        for (int j = 1; j < algo->action_n; j++) {
            float v = tensor_get_f32_at(q_values, (size_t)i * (size_t)algo->action_n + (size_t)j);
            if (v > best_v) {
                best_v = v;
                best = j;
            }
        }
        out_actions[i].index = best;
    }
    tensor_free(q_values);
    tensor_free(x);
}

static void dqn_update(void *ctx, const tfrl_transition *transition) {
    tfrl_dqn_algo *algo = (tfrl_dqn_algo *)ctx;
    if (!transition) return;

    Tensor *x = one_hot(algo->obs_n, transition->obs.index);
    Tensor *q_values = linear_forward(algo->q, x);
    int action = transition->action.index;

    float target = (float)transition->reward;
    if (!transition->done) {
        Tensor *x_next = one_hot(algo->obs_n, transition->next_obs.index);
        Tensor *q_next = linear_forward(algo->q, x_next);
        int next_a = argmax_tensor(q_next);
        float max_q = tensor_get_f32_at(q_next, (size_t)next_a);
        target = (float)transition->reward + algo->gamma * max_q;
        tensor_free(x_next);
        tensor_free(q_next);
    }

    Tensor *action_one = one_hot(algo->action_n, action);
    Tensor *q_sel = tensor_mul(q_values, action_one);
    Tensor *q_val = tensor_sum(q_sel);
    int target_shape[1] = {1};
    Tensor *target_t = tensor_new(1, target_shape);
    tensor_set_f32_at(target_t, 0, target);
    Tensor *diff = tensor_sub(q_val, target_t);
    Tensor *loss = tensor_mul(diff, diff);

    sgd_zero_grad(algo->opt);
    tensor_backward(loss);
    sgd_step(algo->opt, 0.0f);

    tensor_free(action_one);
    tensor_free(q_sel);
    tensor_free(q_val);
    tensor_free(target_t);
    tensor_free(diff);
    tensor_free(loss);
    tensor_free(q_values);
    tensor_free(x);
}

static void dqn_destroy(void *ctx) {
    tfrl_dqn_algo *algo = (tfrl_dqn_algo *)ctx;
    if (!algo) return;
    sgd_free(algo->opt);
    tensor_free(algo->q->weight);
    tensor_free(algo->q->bias);
    free(algo->q);
    free(algo);
}

static void dqn_save(void *ctx, const char *path) {
    tfrl_dqn_algo *algo = (tfrl_dqn_algo *)ctx;
    if (!algo) return;
    save_linear(algo->q, path, "q");
}

static const tfrl_algo_vtable DQN_VTABLE = {
    .act = dqn_act,
    .act_batch = dqn_act_batch,
    .update = dqn_update,
    .save = dqn_save,
    .destroy = dqn_destroy,
};

tfrl_algo tfrl_algo_dqn_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->obs_n <= 0 || cfg->action_n <= 0) return out;
    autograd_set_enabled(1);
    tfrl_dqn_algo *algo = (tfrl_dqn_algo *)calloc(1, sizeof(tfrl_dqn_algo));
    if (!algo) return out;
    algo->obs_n = cfg->obs_n;
    algo->action_n = cfg->action_n;
    algo->gamma = cfg->gamma > 0.0f ? cfg->gamma : 0.99f;
    algo->epsilon = cfg->epsilon;
    algo->q = linear_create(algo->obs_n, algo->action_n);
    if (!algo->q) {
        free(algo);
        return out;
    }
    init_linear(algo->q);
    if (cfg->load_path) {
        load_linear(algo->q, cfg->load_path, "q");
    }
    Tensor *params[2] = {algo->q->weight, algo->q->bias};
    algo->opt = sgd_create(params, 2, cfg->lr > 0.0f ? cfg->lr : 0.05f, 0.0f, 0.0f);
    if (!algo->opt) {
        tensor_free(algo->q->weight);
        tensor_free(algo->q->bias);
        free(algo->q);
        free(algo);
        return out;
    }
    out.ctx = algo;
    out.vtable = &DQN_VTABLE;
    return out;
}
