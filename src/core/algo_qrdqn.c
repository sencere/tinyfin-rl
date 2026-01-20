#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyfin/autograd.h"
#include "tinyfin/nn.h"
#include "tinyfin/optim.h"
#include "tinyfin/ops_add.h"
#include "tinyfin/ops_huber.h"
#include "tinyfin/ops_mul.h"
#include "tinyfin/ops_reduce.h"
#include "tinyfin/ops_sub.h"
#include "tinyfin/tensor.h"

#include "algo_api.h"
#include "replay_buffer.h"

#define QRDQN_QUANTILES 16
#define QRDQN_TARGET_UPDATE 200

typedef struct {
    Linear *q;
    Linear *target_q;
    Adam *opt;
    tfrl_replay_buffer *replay;
    int obs_n;
    int obs_dim;
    tfrl_space_type obs_type;
    int action_n;
    float gamma;
    float epsilon;
    int batch_size;
    int train_every;
    int learning_starts;
    int grad_steps;
    float per_alpha;
    float per_beta;
    int seed;
    int deterministic;
    int step_count;
    int update_count;
    unsigned int rng_state;
    int *idx;
    float *weights;
} tfrl_qrdqn_algo;

static unsigned int qrdqn_lcg_next(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float qrdqn_rand_uniform(tfrl_qrdqn_algo *algo) {
    if (algo->deterministic) {
        return (qrdqn_lcg_next(&algo->rng_state) & 0x00FFFFFFu) / 16777215.0f;
    }
    return (float)rand() / (float)RAND_MAX;
}

static int qrdqn_rand_int(tfrl_qrdqn_algo *algo, int max) {
    if (max <= 0) return 0;
    if (algo->deterministic) {
        return (int)(qrdqn_lcg_next(&algo->rng_state) % (unsigned int)max);
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

static Tensor *obs_to_tensor(const tfrl_qrdqn_algo *algo, tfrl_obs obs) {
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

static void copy_linear(Linear *dst, const Linear *src) {
    if (!dst || !src) return;
    if (dst->weight->size == src->weight->size) {
        memcpy(dst->weight->data, src->weight->data, src->weight->size * sizeof(float));
    }
    if (dst->bias->size == src->bias->size) {
        memcpy(dst->bias->data, src->bias->data, src->bias->size * sizeof(float));
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

static int qrdqn_argmax(Tensor *q_values, int action_n) {
    int best = 0;
    float best_v = -1e30f;
    for (int a = 0; a < action_n; a++) {
        float sum = 0.0f;
        for (int k = 0; k < QRDQN_QUANTILES; k++) {
            size_t idx = (size_t)a * (size_t)QRDQN_QUANTILES + (size_t)k;
            sum += tensor_get_f32_at(q_values, idx);
        }
        float mean = sum / (float)QRDQN_QUANTILES;
        if (mean > best_v) {
            best_v = mean;
            best = a;
        }
    }
    return best;
}

static tfrl_action qrdqn_act(void *ctx, tfrl_obs obs) {
    tfrl_qrdqn_algo *algo = (tfrl_qrdqn_algo *)ctx;
    tfrl_action action = {0};
    float r = qrdqn_rand_uniform(algo);
    if (r < algo->epsilon) {
        action.index = qrdqn_rand_int(algo, algo->action_n);
        return action;
    }
    Tensor *x = obs_to_tensor(algo, obs);
    Tensor *q_values = linear_forward(algo->q, x);
    action.index = qrdqn_argmax(q_values, algo->action_n);
    tensor_free(q_values);
    tensor_free(x);
    return action;
}

static void qrdqn_act_batch(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions) {
    tfrl_qrdqn_algo *algo = (tfrl_qrdqn_algo *)ctx;
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
        float r = qrdqn_rand_uniform(algo);
        if (r < algo->epsilon) {
            out_actions[i].index = qrdqn_rand_int(algo, algo->action_n);
            continue;
        }
        Tensor view = *q_values;
        view.data = q_values->data + (size_t)i * (size_t)algo->action_n * (size_t)QRDQN_QUANTILES;
        view.size = (size_t)algo->action_n * (size_t)QRDQN_QUANTILES;
        out_actions[i].index = qrdqn_argmax(&view, algo->action_n);
    }
    tensor_free(q_values);
    tensor_free(x);
}

static float qrdqn_apply_update(tfrl_qrdqn_algo *algo, const tfrl_transition *transition, float weight) {
    Tensor *x = obs_to_tensor(algo, transition->obs);
    Tensor *q_values = linear_forward(algo->q, x);

    int action = transition->action.index;
    float targets[QRDQN_QUANTILES];
    for (int k = 0; k < QRDQN_QUANTILES; k++) {
        targets[k] = (float)transition->reward;
    }

    if (!transition->done) {
        Tensor *x_next = obs_to_tensor(algo, transition->next_obs);
        Tensor *q_next_online = linear_forward(algo->q, x_next);
        int next_a = qrdqn_argmax(q_next_online, algo->action_n);
        Tensor *q_next_target = linear_forward(algo->target_q, x_next);
        for (int k = 0; k < QRDQN_QUANTILES; k++) {
            size_t idx = (size_t)next_a * (size_t)QRDQN_QUANTILES + (size_t)k;
            targets[k] = (float)transition->reward + algo->gamma * tensor_get_f32_at(q_next_target, idx);
        }
        tensor_free(q_next_target);
        tensor_free(q_next_online);
        tensor_free(x_next);
    }

    float td_err = 0.0f;
    Tensor *loss_sum = NULL;
    for (int k = 0; k < QRDQN_QUANTILES; k++) {
        int sel = action * QRDQN_QUANTILES + k;
        Tensor *mask = one_hot(algo->action_n * QRDQN_QUANTILES, sel);
        Tensor *q_sel = tensor_mul(q_values, mask);
        Tensor *q_val = tensor_sum(q_sel);
        Tensor *target_t = tensor_new(1, (int[1]){1});
        tensor_set_f32_at(target_t, 0, targets[k]);
        Tensor *loss = tensor_huber_loss(q_val, target_t, 1.0f, 0);
        float q_pred = tensor_get_f32_at(q_val, 0);
        float td = targets[k] - q_pred;
        td_err += td >= 0.0f ? td : -td;

        if (weight > 0.0f && weight != 1.0f) {
            Tensor *w_t = tensor_new(1, (int[1]){1});
            tensor_set_f32_at(w_t, 0, weight);
            Tensor *weighted = tensor_mul(loss, w_t);
            tensor_free(loss);
            tensor_free(w_t);
            loss = weighted;
        }

        if (!loss_sum) {
            loss_sum = loss;
        } else {
            Tensor *tmp = tensor_add(loss_sum, loss);
            tensor_free(loss_sum);
            tensor_free(loss);
            loss_sum = tmp;
        }

        tensor_free(target_t);
        tensor_free(q_val);
        tensor_free(q_sel);
        tensor_free(mask);
    }

    if (loss_sum) {
        adam_zero_grad(algo->opt);
        tensor_backward(loss_sum);
        adam_step(algo->opt, 0.0f);
        tensor_free(loss_sum);
    }

    tensor_free(q_values);
    tensor_free(x);

    algo->update_count++;
    if (algo->update_count % QRDQN_TARGET_UPDATE == 0) {
        copy_linear(algo->target_q, algo->q);
    }
    return td_err / (float)QRDQN_QUANTILES;
}

static void qrdqn_update(void *ctx, const tfrl_transition *transition) {
    tfrl_qrdqn_algo *algo = (tfrl_qrdqn_algo *)ctx;
    if (!transition) return;
    algo->step_count++;
    int step_id = algo->step_count;

    if (algo->replay) {
        tfrl_replay_push(algo->replay, transition, 1.0f);
        if (tfrl_replay_size(algo->replay) < algo->batch_size) return;
        if (step_id < algo->learning_starts) return;
        if (algo->train_every > 1 && (step_id % algo->train_every) != 0) return;
        if (!algo->idx || !algo->weights) {
            algo->idx = (int *)calloc((size_t)algo->batch_size, sizeof(int));
            algo->weights = (float *)calloc((size_t)algo->batch_size, sizeof(float));
            if (!algo->idx || !algo->weights) return;
        }
        for (int g = 0; g < algo->grad_steps; g++) {
            if (algo->deterministic) {
                tfrl_replay_sample_deterministic(algo->replay, algo->batch_size, algo->idx, algo->weights, algo->per_beta,
                                                 (unsigned int)(algo->seed + step_id + g));
            } else {
                tfrl_replay_sample(algo->replay, algo->batch_size, algo->idx, algo->weights, algo->per_beta);
            }
            for (int i = 0; i < algo->batch_size; i++) {
                const tfrl_transition *tr = tfrl_replay_get(algo->replay, algo->idx[i]);
                if (!tr) continue;
                float td = qrdqn_apply_update(algo, tr, algo->weights[i]);
                tfrl_replay_update_priority(algo->replay, algo->idx[i], td + 1e-3f);
            }
        }
        return;
    }

    qrdqn_apply_update(algo, transition, 1.0f);
}

static void qrdqn_destroy(void *ctx) {
    tfrl_qrdqn_algo *algo = (tfrl_qrdqn_algo *)ctx;
    if (!algo) return;
    adam_free(algo->opt);
    tfrl_replay_free(algo->replay);
    free(algo->idx);
    free(algo->weights);
    tensor_free(algo->q->weight);
    tensor_free(algo->q->bias);
    tensor_free(algo->target_q->weight);
    tensor_free(algo->target_q->bias);
    free(algo->q);
    free(algo->target_q);
    free(algo);
}

static void qrdqn_save(void *ctx, const char *path) {
    tfrl_qrdqn_algo *algo = (tfrl_qrdqn_algo *)ctx;
    if (!algo) return;
    save_linear(algo->q, path, "q");
}

static const tfrl_algo_vtable QRDQN_VTABLE = {
    .act = qrdqn_act,
    .act_batch = qrdqn_act_batch,
    .update = qrdqn_update,
    .save = qrdqn_save,
    .destroy = qrdqn_destroy,
};

tfrl_algo tfrl_algo_qrdqn_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->obs_n <= 0 || cfg->action_n <= 0) return out;
    autograd_set_enabled(1);
    tfrl_qrdqn_algo *algo = (tfrl_qrdqn_algo *)calloc(1, sizeof(tfrl_qrdqn_algo));
    if (!algo) return out;
    algo->obs_n = cfg->obs_n;
    algo->obs_type = cfg->obs_type;
    algo->obs_dim = cfg->obs_dims > 0 ? cfg->obs_dims : cfg->obs_n;
    algo->action_n = cfg->action_n;
    algo->gamma = cfg->gamma > 0.0f ? cfg->gamma : 0.99f;
    algo->epsilon = cfg->epsilon;
    algo->batch_size = cfg->batch_size > 0 ? cfg->batch_size : 32;
    algo->train_every = cfg->train_every > 0 ? cfg->train_every : 1;
    algo->learning_starts = cfg->learning_starts > 0 ? cfg->learning_starts : 0;
    algo->grad_steps = cfg->grad_steps > 0 ? cfg->grad_steps : 1;
    algo->per_alpha = cfg->per_alpha;
    algo->per_beta = cfg->per_beta;
    algo->seed = cfg->seed;
    algo->deterministic = cfg->deterministic;
    algo->rng_state = (unsigned int)(cfg->seed > 0 ? cfg->seed : 1);
    if (cfg->replay_size > 0) {
        algo->replay = tfrl_replay_create(cfg->replay_size, algo->per_alpha);
    }
    int input_dim = algo->obs_type == TFRL_SPACE_BOX ? algo->obs_dim : algo->obs_n;
    algo->q = linear_create(input_dim, algo->action_n * QRDQN_QUANTILES);
    algo->target_q = linear_create(input_dim, algo->action_n * QRDQN_QUANTILES);
    if (!algo->q || !algo->target_q) {
        qrdqn_destroy(algo);
        return out;
    }
    init_linear(algo->q);
    init_linear(algo->target_q);
    copy_linear(algo->target_q, algo->q);
    if (cfg->load_path) {
        load_linear(algo->q, cfg->load_path, "q");
        copy_linear(algo->target_q, algo->q);
    }
    Tensor *params[2] = {algo->q->weight, algo->q->bias};
    algo->opt = adam_create(params, 2, cfg->lr > 0.0f ? cfg->lr : 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);
    if (!algo->opt) {
        qrdqn_destroy(algo);
        return out;
    }
    if (algo->batch_size > 0) {
        algo->idx = (int *)calloc((size_t)algo->batch_size, sizeof(int));
        algo->weights = (float *)calloc((size_t)algo->batch_size, sizeof(float));
        if (!algo->idx || !algo->weights) {
            qrdqn_destroy(algo);
            return out;
        }
    }
    out.ctx = algo;
    out.vtable = &QRDQN_VTABLE;
    return out;
}
