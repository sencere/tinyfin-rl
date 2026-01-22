#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyfin/autograd.h"
#include "tinyfin/nn.h"
#include "tinyfin/optim.h"
#include "tinyfin/ops_activation.h"
#include "tinyfin/ops_add.h"
#include "tinyfin/ops_mul.h"
#include "tinyfin/ops_reduce.h"
#include "tinyfin/ops_sub.h"
#include "tinyfin/tensor.h"

#include "algo_api.h"

enum {
    GRU_DQN_HIDDEN = 32
};

typedef struct {
    Linear *xz;
    Linear *hz;
    Linear *xr;
    Linear *hr;
    Linear *xh;
    Linear *hh;
    Linear *q;
    SGD *opt;
    int obs_n;
    int obs_dim;
    tfrl_space_type obs_type;
    int action_n;
    float gamma;
    float epsilon;
    int deterministic;
    int seed;
    unsigned int rng_state;
    float h_state[GRU_DQN_HIDDEN];
} tfrl_gru_dqn_algo;

typedef struct {
    Tensor **items;
    int count;
    int capacity;
} tfrl_tensor_list;

static void tensor_list_init(tfrl_tensor_list *list, int capacity) {
    if (!list) return;
    list->items = (Tensor **)calloc((size_t)capacity, sizeof(Tensor *));
    list->count = 0;
    list->capacity = list->items ? capacity : 0;
}

static void tensor_list_push(tfrl_tensor_list *list, Tensor *t) {
    if (!list || !t || list->count >= list->capacity) return;
    list->items[list->count++] = t;
}

static void tensor_list_free(tfrl_tensor_list *list) {
    if (!list || !list->items) return;
    for (int i = 0; i < list->count; i++) {
        tensor_free(list->items[i]);
    }
    free(list->items);
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

static unsigned int dqn_lcg_next(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float dqn_rand_uniform(tfrl_gru_dqn_algo *algo) {
    if (algo->deterministic) {
        return (dqn_lcg_next(&algo->rng_state) & 0x00FFFFFFu) / 16777215.0f;
    }
    return (float)rand() / (float)RAND_MAX;
}

static int dqn_rand_int(tfrl_gru_dqn_algo *algo, int max) {
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

static Tensor *obs_to_tensor(const tfrl_gru_dqn_algo *algo, tfrl_obs obs) {
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

static Tensor *state_to_tensor(const float *state) {
    int shape[2] = {1, GRU_DQN_HIDDEN};
    Tensor *t = tensor_zeros(2, shape);
    if (!t || !state) return t;
    for (int i = 0; i < GRU_DQN_HIDDEN; i++) {
        tensor_set_f32_at(t, (size_t)i, state[i]);
    }
    return t;
}

static void tensor_to_state(const Tensor *t, float *state) {
    if (!t || !state) return;
    for (int i = 0; i < GRU_DQN_HIDDEN; i++) {
        state[i] = tensor_get_f32_at((Tensor *)t, (size_t)i);
    }
}

static Tensor *tensor_ones_like(const Tensor *t) {
    if (!t) return NULL;
    Tensor *out = tensor_zeros(t->ndim, t->shape);
    if (!out) return NULL;
    for (size_t i = 0; i < out->size; i++) {
        tensor_set_f32_at(out, i, 1.0f);
    }
    return out;
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

static Tensor *gru_step(tfrl_gru_dqn_algo *algo, Tensor *x, Tensor *h_prev, tfrl_tensor_list *track) {
    Tensor *xz = linear_forward(algo->xz, x);
    Tensor *hz = linear_forward(algo->hz, h_prev);
    Tensor *zr = tensor_add(xz, hz);
    Tensor *z = zr ? tensor_sigmoid(zr) : NULL;

    Tensor *xr = linear_forward(algo->xr, x);
    Tensor *hr = linear_forward(algo->hr, h_prev);
    Tensor *rr = tensor_add(xr, hr);
    Tensor *r = rr ? tensor_sigmoid(rr) : NULL;

    Tensor *rh = r ? tensor_mul(r, h_prev) : NULL;
    Tensor *xh = linear_forward(algo->xh, x);
    Tensor *hh = rh ? linear_forward(algo->hh, rh) : NULL;
    Tensor *hh_sum = (xh && hh) ? tensor_add(xh, hh) : NULL;
    Tensor *h_tilde = hh_sum ? tensor_tanh(hh_sum) : NULL;

    Tensor *ones = tensor_ones_like(z);
    Tensor *one_minus_z = ones ? tensor_sub(ones, z) : NULL;
    Tensor *h_keep = one_minus_z ? tensor_mul(one_minus_z, h_prev) : NULL;
    Tensor *h_new = (z && h_tilde) ? tensor_mul(z, h_tilde) : NULL;
    Tensor *h = (h_keep && h_new) ? tensor_add(h_keep, h_new) : NULL;

    if (track) {
        tensor_list_push(track, xz);
        tensor_list_push(track, hz);
        tensor_list_push(track, zr);
        tensor_list_push(track, z);
        tensor_list_push(track, xr);
        tensor_list_push(track, hr);
        tensor_list_push(track, rr);
        tensor_list_push(track, r);
        tensor_list_push(track, rh);
        tensor_list_push(track, xh);
        tensor_list_push(track, hh);
        tensor_list_push(track, hh_sum);
        tensor_list_push(track, h_tilde);
        tensor_list_push(track, ones);
        tensor_list_push(track, one_minus_z);
        tensor_list_push(track, h_keep);
        tensor_list_push(track, h_new);
    } else {
        tensor_free(xz);
        tensor_free(hz);
        tensor_free(zr);
        tensor_free(xr);
        tensor_free(hr);
        tensor_free(rr);
        tensor_free(rh);
        tensor_free(xh);
        tensor_free(hh);
        tensor_free(hh_sum);
        tensor_free(ones);
        tensor_free(one_minus_z);
        tensor_free(h_keep);
        tensor_free(h_new);
        tensor_free(z);
        tensor_free(r);
        tensor_free(h_tilde);
    }
    return h;
}

static tfrl_action gru_dqn_act_env(void *ctx, tfrl_env *env, tfrl_obs obs) {
    (void)env;
    tfrl_gru_dqn_algo *algo = (tfrl_gru_dqn_algo *)ctx;
    tfrl_action action = {0};
    float r = dqn_rand_uniform(algo);
    if (r < algo->epsilon) {
        action.index = dqn_rand_int(algo, algo->action_n);
        return action;
    }
    Tensor *x = obs_to_tensor(algo, obs);
    Tensor *h_prev = state_to_tensor(algo->h_state);
    Tensor *h = gru_step(algo, x, h_prev, NULL);
    Tensor *q_values = h ? linear_forward(algo->q, h) : NULL;
    if (q_values) {
        action.index = argmax_tensor(q_values);
    }
    if (h) tensor_to_state(h, algo->h_state);
    tensor_free(q_values);
    tensor_free(h);
    tensor_free(h_prev);
    tensor_free(x);
    return action;
}

static void gru_dqn_update(void *ctx, const tfrl_transition *transition) {
    tfrl_gru_dqn_algo *algo = (tfrl_gru_dqn_algo *)ctx;
    if (!transition) return;

    Tensor *x = obs_to_tensor(algo, transition->obs);
    Tensor *h_prev = state_to_tensor(algo->h_state);
    tfrl_tensor_list track = {0};
    tensor_list_init(&track, 128);
    Tensor *h = gru_step(algo, x, h_prev, &track);
    Tensor *q_values = h ? linear_forward(algo->q, h) : NULL;

    float target = (float)transition->reward;
    if (!transition->done && h) {
        Tensor *x_next = obs_to_tensor(algo, transition->next_obs);
        Tensor *h_next = gru_step(algo, x_next, h, NULL);
        Tensor *q_next = h_next ? linear_forward(algo->q, h_next) : NULL;
        if (q_next) {
            int next_a = argmax_tensor(q_next);
            float max_q = tensor_get_f32_at(q_next, (size_t)next_a);
            target = (float)transition->reward + algo->gamma * max_q;
        }
        tensor_free(q_next);
        tensor_free(h_next);
        tensor_free(x_next);
    }

    int action = transition->action.index;
    if (q_values && action >= 0 && action < algo->action_n) {
        Tensor *action_one = one_hot(algo->action_n, action);
        Tensor *q_sel = tensor_mul(q_values, action_one);
        Tensor *q_val = tensor_sum(q_sel);
        Tensor *target_t = tensor_new(1, (int[1]){1});
        tensor_set_f32_at(target_t, 0, target);
        Tensor *diff = tensor_sub(q_val, target_t);
        Tensor *loss = tensor_mul(diff, diff);

        if (loss) {
            sgd_zero_grad(algo->opt);
            tensor_backward(loss);
            sgd_step(algo->opt, 0.0f);
        }

        tensor_free(action_one);
        tensor_free(q_sel);
        tensor_free(q_val);
        tensor_free(target_t);
        tensor_free(diff);
        tensor_free(loss);
    }

    if (transition->done || !h) {
        memset(algo->h_state, 0, sizeof(algo->h_state));
    } else {
        tensor_to_state(h, algo->h_state);
    }

    tensor_free(q_values);
    tensor_free(h);
    tensor_free(h_prev);
    tensor_free(x);
    tensor_list_free(&track);
}

static void gru_dqn_destroy(void *ctx) {
    tfrl_gru_dqn_algo *algo = (tfrl_gru_dqn_algo *)ctx;
    if (!algo) return;
    sgd_free(algo->opt);
    tensor_free(algo->xz->weight);
    tensor_free(algo->xz->bias);
    tensor_free(algo->hz->weight);
    tensor_free(algo->hz->bias);
    tensor_free(algo->xr->weight);
    tensor_free(algo->xr->bias);
    tensor_free(algo->hr->weight);
    tensor_free(algo->hr->bias);
    tensor_free(algo->xh->weight);
    tensor_free(algo->xh->bias);
    tensor_free(algo->hh->weight);
    tensor_free(algo->hh->bias);
    tensor_free(algo->q->weight);
    tensor_free(algo->q->bias);
    free(algo->xz);
    free(algo->hz);
    free(algo->xr);
    free(algo->hr);
    free(algo->xh);
    free(algo->hh);
    free(algo->q);
    free(algo);
}

static void gru_dqn_save(void *ctx, const char *path) {
    tfrl_gru_dqn_algo *algo = (tfrl_gru_dqn_algo *)ctx;
    if (!algo) return;
    save_linear(algo->xz, path, "xz");
    save_linear(algo->hz, path, "hz");
    save_linear(algo->xr, path, "xr");
    save_linear(algo->hr, path, "hr");
    save_linear(algo->xh, path, "xh");
    save_linear(algo->hh, path, "hh");
    save_linear(algo->q, path, "q");
}

static const tfrl_algo_vtable GRU_DQN_VTABLE = {
    .act = NULL,
    .act_env = gru_dqn_act_env,
    .act_batch = NULL,
    .update = gru_dqn_update,
    .save = gru_dqn_save,
    .destroy = gru_dqn_destroy,
    .diagnostics = NULL,
};

tfrl_algo tfrl_algo_gru_dqn_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->obs_n <= 0 || cfg->action_n <= 0) return out;
    autograd_set_enabled(1);
    tfrl_gru_dqn_algo *algo = (tfrl_gru_dqn_algo *)calloc(1, sizeof(tfrl_gru_dqn_algo));
    if (!algo) return out;
    algo->obs_n = cfg->obs_n;
    algo->obs_type = cfg->obs_type;
    algo->obs_dim = cfg->obs_dims > 0 ? cfg->obs_dims : cfg->obs_n;
    algo->action_n = cfg->action_n;
    algo->gamma = cfg->gamma > 0.0f ? cfg->gamma : 0.99f;
    algo->epsilon = cfg->epsilon;
    algo->deterministic = cfg->deterministic;
    algo->seed = cfg->seed;
    algo->rng_state = (unsigned int)(cfg->seed > 0 ? cfg->seed : 1);

    int input_dim = algo->obs_type == TFRL_SPACE_BOX ? algo->obs_dim : algo->obs_n;
    algo->xz = linear_create(input_dim, GRU_DQN_HIDDEN);
    algo->hz = linear_create(GRU_DQN_HIDDEN, GRU_DQN_HIDDEN);
    algo->xr = linear_create(input_dim, GRU_DQN_HIDDEN);
    algo->hr = linear_create(GRU_DQN_HIDDEN, GRU_DQN_HIDDEN);
    algo->xh = linear_create(input_dim, GRU_DQN_HIDDEN);
    algo->hh = linear_create(GRU_DQN_HIDDEN, GRU_DQN_HIDDEN);
    algo->q = linear_create(GRU_DQN_HIDDEN, algo->action_n);
    if (!algo->xz || !algo->hz || !algo->xr || !algo->hr || !algo->xh || !algo->hh || !algo->q) {
        gru_dqn_destroy(algo);
        return out;
    }
    init_linear(algo->xz);
    init_linear(algo->hz);
    init_linear(algo->xr);
    init_linear(algo->hr);
    init_linear(algo->xh);
    init_linear(algo->hh);
    init_linear(algo->q);
    if (cfg->load_path) {
        load_linear(algo->xz, cfg->load_path, "xz");
        load_linear(algo->hz, cfg->load_path, "hz");
        load_linear(algo->xr, cfg->load_path, "xr");
        load_linear(algo->hr, cfg->load_path, "hr");
        load_linear(algo->xh, cfg->load_path, "xh");
        load_linear(algo->hh, cfg->load_path, "hh");
        load_linear(algo->q, cfg->load_path, "q");
    }
    Tensor *params[14] = {
        algo->xz->weight, algo->xz->bias,
        algo->hz->weight, algo->hz->bias,
        algo->xr->weight, algo->xr->bias,
        algo->hr->weight, algo->hr->bias,
        algo->xh->weight, algo->xh->bias,
        algo->hh->weight, algo->hh->bias,
        algo->q->weight, algo->q->bias
    };
    algo->opt = sgd_create(params, 14, cfg->lr > 0.0f ? cfg->lr : 0.0005f, 0.0f, 0.0f);
    if (!algo->opt) {
        gru_dqn_destroy(algo);
        return out;
    }

    memset(algo->h_state, 0, sizeof(algo->h_state));
    out.ctx = algo;
    out.vtable = &GRU_DQN_VTABLE;
    return out;
}
