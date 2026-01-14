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
#include "tinyfin/tensor.h"

#include "algo_api.h"
#include "replay_buffer.h"

#define IQN_TARGET_UPDATE 200

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
    float per_alpha;
    float per_beta;
    int seed;
    int deterministic;
    int step_count;
    unsigned int rng_state;
    int quantiles;
    int tau_samples;
} tfrl_iqn_algo;

static Tensor *one_hot(int n, int idx) {
    int shape[2] = {1, n};
    Tensor *t = tensor_zeros(2, shape);
    if (!t) return NULL;
    if (idx >= 0 && idx < n) {
        tensor_set_f32_at(t, (size_t)idx, 1.0f);
    }
    return t;
}

static unsigned int iqn_lcg_next(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float iqn_rand_uniform(tfrl_iqn_algo *algo) {
    if (algo->deterministic) {
        return (iqn_lcg_next(&algo->rng_state) & 0x00FFFFFFu) / 16777215.0f;
    }
    return (float)rand() / (float)RAND_MAX;
}

static int iqn_rand_int(tfrl_iqn_algo *algo, int max) {
    if (max <= 0) return 0;
    if (algo->deterministic) {
        return (int)(iqn_lcg_next(&algo->rng_state) % (unsigned int)max);
    }
    return rand() % max;
}

static Tensor *obs_tau_to_tensor(const tfrl_iqn_algo *algo, tfrl_obs obs, float tau) {
    int base_dim = algo->obs_type == TFRL_SPACE_BOX ? algo->obs_dim : algo->obs_n;
    int shape[2] = {1, base_dim + 1};
    Tensor *t = tensor_zeros(2, shape);
    if (!t) return NULL;
    if (algo->obs_type == TFRL_SPACE_BOX) {
        int n = obs.data_len < algo->obs_dim ? obs.data_len : algo->obs_dim;
        for (int i = 0; i < n; i++) {
            tensor_set_f32_at(t, (size_t)i, obs.data[i]);
        }
    } else if (obs.index >= 0 && obs.index < algo->obs_n) {
        tensor_set_f32_at(t, (size_t)obs.index, 1.0f);
    }
    tensor_set_f32_at(t, (size_t)base_dim, tau);
    return t;
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

static int iqn_argmax(tfrl_iqn_algo *algo, tfrl_obs obs) {
    int best = 0;
    float best_v = -1e30f;
    for (int a = 0; a < algo->action_n; a++) {
        float sum = 0.0f;
        for (int k = 0; k < algo->quantiles; k++) {
            float tau = iqn_rand_uniform(algo);
            Tensor *x = obs_tau_to_tensor(algo, obs, tau);
            if (!x) continue;
            Tensor *q = linear_forward(algo->q, x);
            if (q) {
                sum += tensor_get_f32_at(q, (size_t)a);
                tensor_free(q);
            }
            tensor_free(x);
        }
        float mean = sum / (float)algo->quantiles;
        if (mean > best_v) {
            best_v = mean;
            best = a;
        }
    }
    return best;
}

static tfrl_action iqn_act(void *ctx, tfrl_obs obs) {
    tfrl_iqn_algo *algo = (tfrl_iqn_algo *)ctx;
    tfrl_action action = {0};
    float r = iqn_rand_uniform(algo);
    if (r < algo->epsilon) {
        action.index = iqn_rand_int(algo, algo->action_n);
        return action;
    }
    action.index = iqn_argmax(algo, obs);
    return action;
}

static void iqn_act_batch(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions) {
    tfrl_iqn_algo *algo = (tfrl_iqn_algo *)ctx;
    if (!algo || !obs || !out_actions || count <= 0) return;
    for (int i = 0; i < count; i++) {
        float r = iqn_rand_uniform(algo);
        if (r < algo->epsilon) {
            out_actions[i].index = iqn_rand_int(algo, algo->action_n);
        } else {
            out_actions[i].index = iqn_argmax(algo, obs[i]);
        }
    }
}

static float iqn_apply_update(tfrl_iqn_algo *algo, const tfrl_transition *transition, float weight) {
    int next_action = 0;
    if (!transition->done) {
        next_action = iqn_argmax(algo, transition->next_obs);
    }

    float td_err = 0.0f;
    adam_zero_grad(algo->opt);
    for (int k = 0; k < algo->tau_samples; k++) {
        float tau = iqn_rand_uniform(algo);
        Tensor *x = obs_tau_to_tensor(algo, transition->obs, tau);
        Tensor *q = linear_forward(algo->q, x);
        if (!q) {
            tensor_free(x);
            continue;
        }
        Tensor *action_one = one_hot(algo->action_n, transition->action.index);
        if (!action_one) {
            tensor_free(q);
            tensor_free(x);
            continue;
        }
        Tensor *q_sel = tensor_mul(q, action_one);
        if (!q_sel) {
            tensor_free(action_one);
            tensor_free(q);
            tensor_free(x);
            continue;
        }
        Tensor *q_val = tensor_sum(q_sel);
        if (!q_val) {
            tensor_free(q_sel);
            tensor_free(action_one);
            tensor_free(q);
            tensor_free(x);
            continue;
        }

        float target_v = (float)transition->reward;
        if (!transition->done) {
            Tensor *x_next = obs_tau_to_tensor(algo, transition->next_obs, tau);
            if (x_next) {
                Tensor *q_next = linear_forward(algo->target_q, x_next);
                if (q_next) {
                    target_v += algo->gamma * tensor_get_f32_at(q_next, (size_t)next_action);
                    tensor_free(q_next);
                }
                tensor_free(x_next);
            }
        }

        Tensor *target_t = tensor_new(1, (int[1]){1});
        tensor_set_f32_at(target_t, 0, target_v);
        Tensor *loss = tensor_huber_loss(q_val, target_t, 1.0f, 0);
        if (!loss) {
            tensor_free(target_t);
            tensor_free(q_val);
            tensor_free(q_sel);
            tensor_free(action_one);
            tensor_free(q);
            tensor_free(x);
            continue;
        }
        float q_pred = tensor_get_f32_at(q_val, 0);
        float td = target_v - q_pred;
        td_err += td >= 0.0f ? td : -td;

        if (weight > 0.0f && weight != 1.0f) {
            Tensor *w_t = tensor_new(1, (int[1]){1});
            tensor_set_f32_at(w_t, 0, weight);
            Tensor *weighted = tensor_mul(loss, w_t);
            tensor_free(loss);
            tensor_free(w_t);
            loss = weighted;
        }

        tensor_backward(loss);
        tensor_free(loss);
        tensor_free(target_t);
        tensor_free(q_val);
        tensor_free(q_sel);
        tensor_free(action_one);
        tensor_free(q);
        tensor_free(x);
    }

    adam_step(algo->opt, 0.0f);

    algo->step_count++;
    if (algo->step_count % IQN_TARGET_UPDATE == 0) {
        copy_linear(algo->target_q, algo->q);
    }
    return td_err / (float)algo->tau_samples;
}

static void iqn_update(void *ctx, const tfrl_transition *transition) {
    tfrl_iqn_algo *algo = (tfrl_iqn_algo *)ctx;
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
        for (int i = 0; i < algo->batch_size; i++) {
            const tfrl_transition *tr = tfrl_replay_get(algo->replay, idx[i]);
            if (!tr) continue;
            float td = iqn_apply_update(algo, tr, weights[i]);
            tfrl_replay_update_priority(algo->replay, idx[i], td + 1e-3f);
        }
        free(idx);
        free(weights);
        return;
    }

    iqn_apply_update(algo, transition, 1.0f);
}

static void iqn_destroy(void *ctx) {
    tfrl_iqn_algo *algo = (tfrl_iqn_algo *)ctx;
    if (!algo) return;
    adam_free(algo->opt);
    tfrl_replay_free(algo->replay);
    tensor_free(algo->q->weight);
    tensor_free(algo->q->bias);
    tensor_free(algo->target_q->weight);
    tensor_free(algo->target_q->bias);
    free(algo->q);
    free(algo->target_q);
    free(algo);
}

static void iqn_save(void *ctx, const char *path) {
    tfrl_iqn_algo *algo = (tfrl_iqn_algo *)ctx;
    if (!algo) return;
    save_linear(algo->q, path, "q");
}

static const tfrl_algo_vtable IQN_VTABLE = {
    .act = iqn_act,
    .act_batch = iqn_act_batch,
    .update = iqn_update,
    .save = iqn_save,
    .destroy = iqn_destroy,
};

tfrl_algo tfrl_algo_iqn_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->obs_n <= 0 || cfg->action_n <= 0) return out;
    autograd_set_enabled(1);
    tfrl_iqn_algo *algo = (tfrl_iqn_algo *)calloc(1, sizeof(tfrl_iqn_algo));
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
    algo->quantiles = cfg->iqn_quantiles > 0 ? cfg->iqn_quantiles : 32;
    algo->tau_samples = cfg->iqn_tau_samples > 0 ? cfg->iqn_tau_samples : 32;
    if (cfg->replay_size > 0) {
        algo->replay = tfrl_replay_create(cfg->replay_size, algo->per_alpha);
    }
    int base_dim = algo->obs_type == TFRL_SPACE_BOX ? algo->obs_dim : algo->obs_n;
    algo->q = linear_create(base_dim + 1, algo->action_n);
    algo->target_q = linear_create(base_dim + 1, algo->action_n);
    if (!algo->q || !algo->target_q) {
        iqn_destroy(algo);
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
        iqn_destroy(algo);
        return out;
    }
    out.ctx = algo;
    out.vtable = &IQN_VTABLE;
    return out;
}
