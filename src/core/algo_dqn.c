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
#include "replay_buffer.h"

typedef struct {
    Linear *q;
    SGD *opt;
    tfrl_replay_buffer *replay;
    int obs_n;
    int obs_dim;
    tfrl_space_type obs_type;
    int action_n;
    float gamma;
    float epsilon;
    int batch_size;
    float per_alpha;
    float per_beta;
    int seed;
    int deterministic;
    int step_count;
    unsigned int rng_state;
} tfrl_dqn_algo;

static unsigned int dqn_lcg_next(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float dqn_rand_uniform(tfrl_dqn_algo *algo) {
    if (algo->deterministic) {
        return (dqn_lcg_next(&algo->rng_state) & 0x00FFFFFFu) / 16777215.0f;
    }
    return (float)rand() / (float)RAND_MAX;
}

static int dqn_rand_int(tfrl_dqn_algo *algo, int max) {
    if (max <= 0) return 0;
    if (algo->deterministic) {
        return (int)(dqn_lcg_next(&algo->rng_state) % (unsigned int)max);
    }
    return rand() % max;
}

static Tensor *one_hot(int n, int idx) {
    int shape[2] = {1, n};
    Tensor *t = tensor_zeros(2, shape);
    if (!t) return NULL;
    if (idx >= 0 && idx < n) {
        tensor_set_f32_at(t, (size_t)idx, 1.0f);
    }
    return t;
}

static Tensor *obs_to_tensor(const tfrl_dqn_algo *algo, tfrl_obs obs) {
    if (algo->obs_type == TFRL_SPACE_BOX) {
        int shape[2] = {1, algo->obs_dim};
        Tensor *t = tensor_zeros(2, shape);
        if (!t) return NULL;
        int n = obs.data_len < algo->obs_dim ? obs.data_len : algo->obs_dim;
        for (int i = 0; i < n; i++) {
            tensor_set_f32_at(t, (size_t)i, obs.data[i]);
        }
        return t;
    }
    return one_hot(algo->obs_n, obs.index);
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
    float r = dqn_rand_uniform(algo);
    if (r < algo->epsilon) {
        action.index = dqn_rand_int(algo, algo->action_n);
        return action;
    }
    Tensor *x = obs_to_tensor(algo, obs);
    Tensor *q_values = linear_forward(algo->q, x);
    action.index = argmax_tensor(q_values);
    tensor_free(q_values);
    tensor_free(x);
    return action;
}

static void dqn_act_batch(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions) {
    tfrl_dqn_algo *algo = (tfrl_dqn_algo *)ctx;
    if (!algo || !obs || !out_actions || count <= 0) return;

    int obs_dim = algo->obs_type == TFRL_SPACE_BOX ? algo->obs_dim : algo->obs_n;
    int shape[2] = {count, obs_dim};
    Tensor *x = tensor_zeros(2, shape);
    if (!x) return;
    for (int i = 0; i < count; i++) {
        if (algo->obs_type == TFRL_SPACE_BOX) {
            int n = obs[i].data_len < obs_dim ? obs[i].data_len : obs_dim;
            for (int j = 0; j < n; j++) {
                size_t off = (size_t)i * (size_t)obs_dim + (size_t)j;
                tensor_set_f32_at(x, off, obs[i].data[j]);
            }
        } else if (obs[i].index >= 0 && obs[i].index < algo->obs_n) {
            size_t off = (size_t)i * (size_t)obs_dim + (size_t)obs[i].index;
            tensor_set_f32_at(x, off, 1.0f);
        }
    }

    Tensor *q_values = linear_forward(algo->q, x);
    for (int i = 0; i < count; i++) {
        float r = dqn_rand_uniform(algo);
        if (r < algo->epsilon) {
            out_actions[i].index = dqn_rand_int(algo, algo->action_n);
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

    if (algo->replay) {
        tfrl_replay_push(algo->replay, transition, 1.0f);
        if (tfrl_replay_size(algo->replay) < algo->batch_size) return;

        int *idx = (int *)calloc((size_t)algo->batch_size, sizeof(int));
        float *weights = (float *)calloc((size_t)algo->batch_size, sizeof(float));
        if (!idx || !weights) {
            free(idx);
            free(weights);
            return;
        }
        if (algo->deterministic) {
            tfrl_replay_sample_deterministic(algo->replay, algo->batch_size, idx, weights, algo->per_beta, (unsigned int)(algo->seed + algo->step_count));
        } else {
            tfrl_replay_sample(algo->replay, algo->batch_size, idx, weights, algo->per_beta);
        }

        Tensor *loss_sum = NULL;
        for (int i = 0; i < algo->batch_size; i++) {
            const tfrl_transition *tr = tfrl_replay_get(algo->replay, idx[i]);
            if (!tr) continue;

            Tensor *x = obs_to_tensor(algo, tr->obs);
            Tensor *q_values = linear_forward(algo->q, x);
            int action = tr->action.index;

            float target = (float)tr->reward;
            if (!tr->done) {
                Tensor *x_next = obs_to_tensor(algo, tr->next_obs);
                Tensor *q_next = linear_forward(algo->q, x_next);
                int next_a = argmax_tensor(q_next);
                float max_q = tensor_get_f32_at(q_next, (size_t)next_a);
                target = (float)tr->reward + algo->gamma * max_q;
                tensor_free(x_next);
                tensor_free(q_next);
            }

            float q_pred = tensor_get_f32_at(q_values, (size_t)action);
            float td_err = target - q_pred;
            float priority = td_err >= 0.0f ? td_err : -td_err;
            tfrl_replay_update_priority(algo->replay, idx[i], priority + 1e-3f);

            Tensor *action_one = one_hot(algo->action_n, action);
            Tensor *q_sel = tensor_mul(q_values, action_one);
            Tensor *q_val = tensor_sum(q_sel);
            Tensor *target_t = tensor_new(1, (int[1]){1});
            tensor_set_f32_at(target_t, 0, target);
            Tensor *diff = tensor_sub(q_val, target_t);
            Tensor *loss = tensor_mul(diff, diff);
            Tensor *weight_t = tensor_new(1, (int[1]){1});
            tensor_set_f32_at(weight_t, 0, weights[i]);
            Tensor *weighted = tensor_mul(loss, weight_t);

            if (!loss_sum) {
                loss_sum = weighted;
            } else {
                Tensor *tmp = tensor_add(loss_sum, weighted);
                tensor_free(loss_sum);
                tensor_free(weighted);
                loss_sum = tmp;
            }

            tensor_free(weight_t);
            tensor_free(loss);
            tensor_free(diff);
            tensor_free(target_t);
            tensor_free(q_val);
            tensor_free(q_sel);
            tensor_free(action_one);
            tensor_free(q_values);
            tensor_free(x);
        }

        if (loss_sum) {
            sgd_zero_grad(algo->opt);
            tensor_backward(loss_sum);
            sgd_step(algo->opt, 0.0f);
            tensor_free(loss_sum);
        }
        free(idx);
        free(weights);
        algo->step_count++;
        return;
    }

    Tensor *x = obs_to_tensor(algo, transition->obs);
    Tensor *q_values = linear_forward(algo->q, x);
    int action = transition->action.index;

    float target = (float)transition->reward;
    if (!transition->done) {
        Tensor *x_next = obs_to_tensor(algo, transition->next_obs);
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
    Tensor *target_t = tensor_new(1, (int[1]){1});
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
    tfrl_replay_free(algo->replay);
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
    algo->obs_type = cfg->obs_type;
    algo->obs_dim = cfg->obs_dims > 0 ? cfg->obs_dims : cfg->obs_n;
    algo->action_n = cfg->action_n;
    algo->gamma = cfg->gamma > 0.0f ? cfg->gamma : 0.99f;
    algo->epsilon = cfg->epsilon;
    algo->batch_size = cfg->batch_size > 0 ? cfg->batch_size : 32;
    algo->per_alpha = cfg->per_alpha;
    algo->per_beta = cfg->per_beta;
    algo->seed = cfg->seed;
    algo->deterministic = cfg->deterministic;
    algo->rng_state = (unsigned int)(cfg->seed > 0 ? cfg->seed : 1);
    if (cfg->replay_size > 0) {
        algo->replay = tfrl_replay_create(cfg->replay_size, algo->per_alpha);
    }
    int input_dim = algo->obs_type == TFRL_SPACE_BOX ? algo->obs_dim : algo->obs_n;
    algo->q = linear_create(input_dim, algo->action_n);
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
