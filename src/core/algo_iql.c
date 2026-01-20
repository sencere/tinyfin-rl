#include <math.h>
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

#define IQL_EXPECTILE 0.7f

typedef struct {
    Linear *q;
    Linear *v;
    Adam *q_opt;
    Adam *v_opt;
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
    unsigned int rng_state;
} tfrl_iql_algo;

static unsigned int iql_lcg_next(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float iql_rand_uniform(tfrl_iql_algo *algo) {
    if (algo->deterministic) {
        return (iql_lcg_next(&algo->rng_state) & 0x00FFFFFFu) / 16777215.0f;
    }
    return (float)rand() / (float)RAND_MAX;
}

static int iql_rand_int(tfrl_iql_algo *algo, int max) {
    if (max <= 0) return 0;
    if (algo->deterministic) {
        return (int)(iql_lcg_next(&algo->rng_state) % (unsigned int)max);
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

static Tensor *obs_to_tensor(const tfrl_iql_algo *algo, tfrl_obs obs) {
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

static tfrl_action iql_act(void *ctx, tfrl_obs obs) {
    tfrl_iql_algo *algo = (tfrl_iql_algo *)ctx;
    tfrl_action action = {0};
    float r = iql_rand_uniform(algo);
    if (r < algo->epsilon) {
        action.index = iql_rand_int(algo, algo->action_n);
        return action;
    }
    Tensor *x = obs_to_tensor(algo, obs);
    Tensor *q_values = linear_forward(algo->q, x);
    action.index = argmax_tensor(q_values);
    tensor_free(q_values);
    tensor_free(x);
    return action;
}

static void iql_act_batch(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions) {
    tfrl_iql_algo *algo = (tfrl_iql_algo *)ctx;
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
        float r = iql_rand_uniform(algo);
        if (r < algo->epsilon) {
            out_actions[i].index = iql_rand_int(algo, algo->action_n);
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

static void iql_update(void *ctx, const tfrl_transition *transition) {
    tfrl_iql_algo *algo = (tfrl_iql_algo *)ctx;
    if (!transition) return;
    algo->step_count++;
    int step_id = algo->step_count;

    if (!algo->replay) return;
    tfrl_replay_push(algo->replay, transition, 1.0f);
    if (tfrl_replay_size(algo->replay) < algo->batch_size) return;
    if (step_id < algo->learning_starts) return;
    if (algo->train_every > 1 && (step_id % algo->train_every) != 0) return;

    int *idx = (int *)calloc((size_t)algo->batch_size, sizeof(int));
    float *weights = (float *)calloc((size_t)algo->batch_size, sizeof(float));
    if (!idx || !weights) {
        free(idx);
        free(weights);
        return;
    }

    for (int g = 0; g < algo->grad_steps; g++) {
        if (algo->deterministic) {
            tfrl_replay_sample_deterministic(algo->replay, algo->batch_size, idx, weights, algo->per_beta,
                                             (unsigned int)(algo->seed + step_id + g));
        } else {
            tfrl_replay_sample(algo->replay, algo->batch_size, idx, weights, algo->per_beta);
        }

        int obs_dim = algo->obs_type == TFRL_SPACE_BOX ? algo->obs_dim : algo->obs_n;
        int x_shape[2] = {algo->batch_size, obs_dim};
        int q_shape[2] = {algo->batch_size, algo->action_n};
        int v_shape[2] = {algo->batch_size, 1};
        Tensor *x = tensor_zeros(2, x_shape);
        Tensor *x_next = tensor_zeros(2, x_shape);
        Tensor *target_mat = tensor_zeros(2, q_shape);
        Tensor *weight_mat = tensor_zeros(2, q_shape);
        Tensor *q_sa = tensor_zeros(2, v_shape);
        Tensor *weight_v = tensor_zeros(2, v_shape);
        if (!x || !x_next || !target_mat || !weight_mat || !q_sa || !weight_v) {
            tensor_free(x);
            tensor_free(x_next);
            tensor_free(target_mat);
            tensor_free(weight_mat);
            tensor_free(q_sa);
            tensor_free(weight_v);
            continue;
        }

        for (int i = 0; i < algo->batch_size; i++) {
            const tfrl_transition *tr = tfrl_replay_get(algo->replay, idx[i]);
            if (!tr) continue;

            if (algo->obs_type == TFRL_SPACE_BOX) {
                int n = tr->obs.data_len < obs_dim ? tr->obs.data_len : obs_dim;
                for (int j = 0; j < n; j++) {
                    size_t off = (size_t)i * (size_t)obs_dim + (size_t)j;
                    tensor_set_f32_at(x, off, tr->obs.data[j]);
                }
                n = tr->next_obs.data_len < obs_dim ? tr->next_obs.data_len : obs_dim;
                for (int j = 0; j < n; j++) {
                    size_t off = (size_t)i * (size_t)obs_dim + (size_t)j;
                    tensor_set_f32_at(x_next, off, tr->next_obs.data[j]);
                }
            } else {
                if (tr->obs.index >= 0 && tr->obs.index < algo->obs_n) {
                    size_t off = (size_t)i * (size_t)obs_dim + (size_t)tr->obs.index;
                    tensor_set_f32_at(x, off, 1.0f);
                }
                if (tr->next_obs.index >= 0 && tr->next_obs.index < algo->obs_n) {
                    size_t off = (size_t)i * (size_t)obs_dim + (size_t)tr->next_obs.index;
                    tensor_set_f32_at(x_next, off, 1.0f);
                }
            }
        }

        Tensor *q_values = linear_forward(algo->q, x);
        Tensor *v_values = linear_forward(algo->v, x);
        Tensor *v_next = linear_forward(algo->v, x_next);
        if (!q_values || !v_values || !v_next) {
            tensor_free(q_values);
            tensor_free(v_values);
            tensor_free(v_next);
            tensor_free(x);
            tensor_free(x_next);
            tensor_free(target_mat);
            tensor_free(weight_mat);
            tensor_free(q_sa);
            tensor_free(weight_v);
            continue;
        }

        for (int i = 0; i < algo->batch_size; i++) {
            const tfrl_transition *tr = tfrl_replay_get(algo->replay, idx[i]);
            if (!tr) continue;
            int action = tr->action.index;
            if (action < 0 || action >= algo->action_n) continue;

            float v_next_val = tr->done ? 0.0f : tensor_get_f32_at(v_next, (size_t)i);
            float target = (float)tr->reward + algo->gamma * v_next_val;
            size_t off = (size_t)i * (size_t)algo->action_n + (size_t)action;
            tensor_set_f32_at(target_mat, off, target);
            tensor_set_f32_at(weight_mat, off, weights[i]);

            float q_sa_val = tensor_get_f32_at(q_values, off);
            tensor_set_f32_at(q_sa, (size_t)i, q_sa_val);
            float v_val = tensor_get_f32_at(v_values, (size_t)i);
            float diff = q_sa_val - v_val;
            float w = diff > 0.0f ? IQL_EXPECTILE : (1.0f - IQL_EXPECTILE);
            tensor_set_f32_at(weight_v, (size_t)i, w);

            float td_err = target - q_sa_val;
            float priority = td_err >= 0.0f ? td_err : -td_err;
            tfrl_replay_update_priority(algo->replay, idx[i], priority + 1e-3f);
        }

        Tensor *diff_q = tensor_sub(q_values, target_mat);
        Tensor *diff_q_sq = diff_q ? tensor_mul(diff_q, diff_q) : NULL;
        Tensor *weighted_q = diff_q_sq ? tensor_mul(diff_q_sq, weight_mat) : NULL;
        Tensor *loss_q = weighted_q ? tensor_sum(weighted_q) : NULL;

        if (loss_q) {
            adam_zero_grad(algo->q_opt);
            tensor_backward(loss_q);
            adam_step(algo->q_opt, 0.0f);
        }

        Tensor *diff_v = tensor_sub(q_sa, v_values);
        Tensor *diff_v_sq = diff_v ? tensor_mul(diff_v, diff_v) : NULL;
        Tensor *weighted_v = diff_v_sq ? tensor_mul(diff_v_sq, weight_v) : NULL;
        Tensor *loss_v = weighted_v ? tensor_sum(weighted_v) : NULL;

        if (loss_v) {
            adam_zero_grad(algo->v_opt);
            tensor_backward(loss_v);
            adam_step(algo->v_opt, 0.0f);
        }

        tensor_free(loss_v);
        tensor_free(weighted_v);
        tensor_free(diff_v_sq);
        tensor_free(diff_v);
        tensor_free(loss_q);
        tensor_free(weighted_q);
        tensor_free(diff_q_sq);
        tensor_free(diff_q);
        tensor_free(v_next);
        tensor_free(v_values);
        tensor_free(q_values);
        tensor_free(weight_v);
        tensor_free(q_sa);
        tensor_free(weight_mat);
        tensor_free(target_mat);
        tensor_free(x_next);
        tensor_free(x);
    }

    free(idx);
    free(weights);
}

static void iql_destroy(void *ctx) {
    tfrl_iql_algo *algo = (tfrl_iql_algo *)ctx;
    if (!algo) return;
    adam_free(algo->q_opt);
    adam_free(algo->v_opt);
    tfrl_replay_free(algo->replay);
    tensor_free(algo->q->weight);
    tensor_free(algo->q->bias);
    free(algo->q);
    tensor_free(algo->v->weight);
    tensor_free(algo->v->bias);
    free(algo->v);
    free(algo);
}

static void iql_save(void *ctx, const char *path) {
    tfrl_iql_algo *algo = (tfrl_iql_algo *)ctx;
    if (!algo) return;
    save_linear(algo->q, path, "q");
    save_linear(algo->v, path, "v");
}

static const tfrl_algo_vtable IQL_VTABLE = {
    .act = iql_act,
    .act_batch = iql_act_batch,
    .update = iql_update,
    .save = iql_save,
    .destroy = iql_destroy,
};

tfrl_algo tfrl_algo_iql_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->obs_n <= 0 || cfg->action_n <= 0) return out;
    autograd_set_enabled(1);
    tfrl_iql_algo *algo = (tfrl_iql_algo *)calloc(1, sizeof(tfrl_iql_algo));
    if (!algo) return out;
    algo->obs_n = cfg->obs_n;
    algo->obs_type = cfg->obs_type;
    algo->obs_dim = cfg->obs_dims > 0 ? cfg->obs_dims : cfg->obs_n;
    algo->action_n = cfg->action_n;
    algo->gamma = cfg->gamma > 0.0f ? cfg->gamma : 0.99f;
    algo->epsilon = cfg->epsilon;
    algo->batch_size = cfg->batch_size > 0 ? cfg->batch_size : 64;
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
    algo->q = linear_create(input_dim, algo->action_n);
    algo->v = linear_create(input_dim, 1);
    if (!algo->q || !algo->v) {
        if (algo->q) {
            tensor_free(algo->q->weight);
            tensor_free(algo->q->bias);
            free(algo->q);
        }
        if (algo->v) {
            tensor_free(algo->v->weight);
            tensor_free(algo->v->bias);
            free(algo->v);
        }
        free(algo);
        return out;
    }
    init_linear(algo->q);
    init_linear(algo->v);
    if (cfg->load_path) {
        load_linear(algo->q, cfg->load_path, "q");
        load_linear(algo->v, cfg->load_path, "v");
    }
    Tensor *q_params[2] = {algo->q->weight, algo->q->bias};
    Tensor *v_params[2] = {algo->v->weight, algo->v->bias};
    float lr = cfg->lr > 0.0f ? cfg->lr : 0.0003f;
    algo->q_opt = adam_create(q_params, 2, lr, 0.9f, 0.999f, 1e-8f, 0.0f);
    algo->v_opt = adam_create(v_params, 2, lr, 0.9f, 0.999f, 1e-8f, 0.0f);
    if (!algo->q_opt || !algo->v_opt) {
        adam_free(algo->q_opt);
        adam_free(algo->v_opt);
        tensor_free(algo->q->weight);
        tensor_free(algo->q->bias);
        free(algo->q);
        tensor_free(algo->v->weight);
        tensor_free(algo->v->bias);
        free(algo->v);
        free(algo);
        return out;
    }
    out.ctx = algo;
    out.vtable = &IQL_VTABLE;
    return out;
}
