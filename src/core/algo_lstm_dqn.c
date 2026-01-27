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
    LSTM_DQN_HIDDEN = 32
};

typedef struct {
    Linear *xi;
    Linear *hi;
    Linear *xf;
    Linear *hf;
    Linear *xo;
    Linear *ho;
    Linear *xg;
    Linear *hg;
    Linear *q;
    SGD *opt;
    int obs_n;
    int obs_dim;
    tfrl_space_type obs_type;
    int action_n;
    float gamma;
    float epsilon;
    float epsilon_start;
    float epsilon_end;
    int epsilon_decay_steps;
    int step_count;
    int deterministic;
    int seed;
    unsigned int rng_state;
    float h_state[LSTM_DQN_HIDDEN];
    float c_state[LSTM_DQN_HIDDEN];
} tfrl_lstm_dqn_algo;

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

static float dqn_rand_uniform(tfrl_lstm_dqn_algo *algo) {
    if (algo->deterministic) {
        return (dqn_lcg_next(&algo->rng_state) & 0x00FFFFFFu) / 16777215.0f;
    }
    return (float)rand() / (float)RAND_MAX;
}

static int dqn_rand_int(tfrl_lstm_dqn_algo *algo, int max) {
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

static Tensor *obs_to_tensor(const tfrl_lstm_dqn_algo *algo, tfrl_obs obs) {
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
    int shape[2] = {1, LSTM_DQN_HIDDEN};
    Tensor *t = tensor_zeros(2, shape);
    if (!t || !state) return t;
    for (int i = 0; i < LSTM_DQN_HIDDEN; i++) {
        tensor_set_f32_at(t, (size_t)i, state[i]);
    }
    return t;
}

static void tensor_to_state(const Tensor *t, float *state) {
    if (!t || !state) return;
    for (int i = 0; i < LSTM_DQN_HIDDEN; i++) {
        state[i] = tensor_get_f32_at((Tensor *)t, (size_t)i);
    }
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

static void lstm_step(tfrl_lstm_dqn_algo *algo, Tensor *x, Tensor *h_prev, Tensor *c_prev,
                      Tensor **out_h, Tensor **out_c, tfrl_tensor_list *track) {
    Tensor *xi = linear_forward(algo->xi, x);
    Tensor *hi = linear_forward(algo->hi, h_prev);
    Tensor *i_sum = tensor_add(xi, hi);
    Tensor *i_gate = i_sum ? tensor_sigmoid(i_sum) : NULL;

    Tensor *xf = linear_forward(algo->xf, x);
    Tensor *hf = linear_forward(algo->hf, h_prev);
    Tensor *f_sum = tensor_add(xf, hf);
    Tensor *f_gate = f_sum ? tensor_sigmoid(f_sum) : NULL;

    Tensor *xo = linear_forward(algo->xo, x);
    Tensor *ho = linear_forward(algo->ho, h_prev);
    Tensor *o_sum = tensor_add(xo, ho);
    Tensor *o_gate = o_sum ? tensor_sigmoid(o_sum) : NULL;

    Tensor *xg = linear_forward(algo->xg, x);
    Tensor *hg = linear_forward(algo->hg, h_prev);
    Tensor *g_sum = tensor_add(xg, hg);
    Tensor *g_gate = g_sum ? tensor_tanh(g_sum) : NULL;

    Tensor *f_c = (f_gate && c_prev) ? tensor_mul(f_gate, c_prev) : NULL;
    Tensor *i_g = (i_gate && g_gate) ? tensor_mul(i_gate, g_gate) : NULL;
    Tensor *c_next = (f_c && i_g) ? tensor_add(f_c, i_g) : NULL;
    Tensor *c_tanh = c_next ? tensor_tanh(c_next) : NULL;
    Tensor *h_next = (o_gate && c_tanh) ? tensor_mul(o_gate, c_tanh) : NULL;

    if (track) {
        tensor_list_push(track, xi);
        tensor_list_push(track, hi);
        tensor_list_push(track, i_sum);
        tensor_list_push(track, i_gate);
        tensor_list_push(track, xf);
        tensor_list_push(track, hf);
        tensor_list_push(track, f_sum);
        tensor_list_push(track, f_gate);
        tensor_list_push(track, xo);
        tensor_list_push(track, ho);
        tensor_list_push(track, o_sum);
        tensor_list_push(track, o_gate);
        tensor_list_push(track, xg);
        tensor_list_push(track, hg);
        tensor_list_push(track, g_sum);
        tensor_list_push(track, g_gate);
        tensor_list_push(track, f_c);
        tensor_list_push(track, i_g);
        tensor_list_push(track, c_tanh);
    } else {
        tensor_free(xi);
        tensor_free(hi);
        tensor_free(i_sum);
        tensor_free(xf);
        tensor_free(hf);
        tensor_free(f_sum);
        tensor_free(xo);
        tensor_free(ho);
        tensor_free(o_sum);
        tensor_free(xg);
        tensor_free(hg);
        tensor_free(g_sum);
        tensor_free(f_c);
        tensor_free(i_g);
        tensor_free(c_tanh);
        tensor_free(i_gate);
        tensor_free(f_gate);
        tensor_free(o_gate);
        tensor_free(g_gate);
    }

    if (out_h) *out_h = h_next;
    else tensor_free(h_next);
    if (out_c) *out_c = c_next;
    else tensor_free(c_next);
}

static tfrl_action lstm_dqn_act_env(void *ctx, tfrl_env *env, tfrl_obs obs) {
    (void)env;
    tfrl_lstm_dqn_algo *algo = (tfrl_lstm_dqn_algo *)ctx;
    tfrl_action action = {0};
    algo->step_count++;
    if (algo->epsilon_decay_steps > 0) {
        float t = (float)algo->step_count / (float)algo->epsilon_decay_steps;
        if (t > 1.0f) t = 1.0f;
        algo->epsilon = algo->epsilon_start + t * (algo->epsilon_end - algo->epsilon_start);
    }
    float r = dqn_rand_uniform(algo);
    if (r < algo->epsilon) {
        action.index = dqn_rand_int(algo, algo->action_n);
        return action;
    }
    Tensor *x = obs_to_tensor(algo, obs);
    Tensor *h_prev = state_to_tensor(algo->h_state);
    Tensor *c_prev = state_to_tensor(algo->c_state);
    Tensor *h = NULL;
    Tensor *c = NULL;
    lstm_step(algo, x, h_prev, c_prev, &h, &c, NULL);
    Tensor *q_values = h ? linear_forward(algo->q, h) : NULL;
    if (q_values) {
        action.index = argmax_tensor(q_values);
    }
    if (h && c) {
        tensor_to_state(h, algo->h_state);
        tensor_to_state(c, algo->c_state);
    }
    tensor_free(q_values);
    tensor_free(h);
    tensor_free(c);
    tensor_free(h_prev);
    tensor_free(c_prev);
    tensor_free(x);
    return action;
}

static void lstm_dqn_update(void *ctx, const tfrl_transition *transition) {
    tfrl_lstm_dqn_algo *algo = (tfrl_lstm_dqn_algo *)ctx;
    if (!transition) return;

    Tensor *x = obs_to_tensor(algo, transition->obs);
    Tensor *h_prev = state_to_tensor(algo->h_state);
    Tensor *c_prev = state_to_tensor(algo->c_state);
    Tensor *h = NULL;
    Tensor *c = NULL;
    tfrl_tensor_list track = {0};
    tensor_list_init(&track, 128);
    lstm_step(algo, x, h_prev, c_prev, &h, &c, &track);
    Tensor *q_values = h ? linear_forward(algo->q, h) : NULL;

    float target = (float)transition->reward;
    if (!transition->done && h && c) {
        Tensor *x_next = obs_to_tensor(algo, transition->next_obs);
        Tensor *h_next = NULL;
        Tensor *c_next = NULL;
        lstm_step(algo, x_next, h, c, &h_next, &c_next, NULL);
        Tensor *q_next = h_next ? linear_forward(algo->q, h_next) : NULL;
        if (q_next) {
            int next_a = argmax_tensor(q_next);
            float max_q = tensor_get_f32_at(q_next, (size_t)next_a);
            target = (float)transition->reward + algo->gamma * max_q;
        }
        tensor_free(q_next);
        tensor_free(h_next);
        tensor_free(c_next);
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

    if (transition->done || !h || !c) {
        memset(algo->h_state, 0, sizeof(algo->h_state));
        memset(algo->c_state, 0, sizeof(algo->c_state));
    } else {
        tensor_to_state(h, algo->h_state);
        tensor_to_state(c, algo->c_state);
    }

    tensor_free(q_values);
    tensor_free(h);
    tensor_free(c);
    tensor_free(h_prev);
    tensor_free(c_prev);
    tensor_free(x);
    tensor_list_free(&track);
}

static void lstm_dqn_destroy(void *ctx) {
    tfrl_lstm_dqn_algo *algo = (tfrl_lstm_dqn_algo *)ctx;
    if (!algo) return;
    sgd_free(algo->opt);
    tensor_free(algo->xi->weight);
    tensor_free(algo->xi->bias);
    tensor_free(algo->hi->weight);
    tensor_free(algo->hi->bias);
    tensor_free(algo->xf->weight);
    tensor_free(algo->xf->bias);
    tensor_free(algo->hf->weight);
    tensor_free(algo->hf->bias);
    tensor_free(algo->xo->weight);
    tensor_free(algo->xo->bias);
    tensor_free(algo->ho->weight);
    tensor_free(algo->ho->bias);
    tensor_free(algo->xg->weight);
    tensor_free(algo->xg->bias);
    tensor_free(algo->hg->weight);
    tensor_free(algo->hg->bias);
    tensor_free(algo->q->weight);
    tensor_free(algo->q->bias);
    free(algo->xi);
    free(algo->hi);
    free(algo->xf);
    free(algo->hf);
    free(algo->xo);
    free(algo->ho);
    free(algo->xg);
    free(algo->hg);
    free(algo->q);
    free(algo);
}

static void lstm_dqn_save(void *ctx, const char *path) {
    tfrl_lstm_dqn_algo *algo = (tfrl_lstm_dqn_algo *)ctx;
    if (!algo) return;
    save_linear(algo->xi, path, "xi");
    save_linear(algo->hi, path, "hi");
    save_linear(algo->xf, path, "xf");
    save_linear(algo->hf, path, "hf");
    save_linear(algo->xo, path, "xo");
    save_linear(algo->ho, path, "ho");
    save_linear(algo->xg, path, "xg");
    save_linear(algo->hg, path, "hg");
    save_linear(algo->q, path, "q");
}

static const tfrl_algo_vtable LSTM_DQN_VTABLE = {
    .act = NULL,
    .act_env = lstm_dqn_act_env,
    .act_batch = NULL,
    .update = lstm_dqn_update,
    .save = lstm_dqn_save,
    .destroy = lstm_dqn_destroy,
    .diagnostics = NULL,
};

tfrl_algo tfrl_algo_lstm_dqn_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->obs_n <= 0 || cfg->action_n <= 0) return out;
    autograd_set_enabled(1);
    tfrl_lstm_dqn_algo *algo = (tfrl_lstm_dqn_algo *)calloc(1, sizeof(tfrl_lstm_dqn_algo));
    if (!algo) return out;
    algo->obs_n = cfg->obs_n;
    algo->obs_type = cfg->obs_type;
    algo->obs_dim = cfg->obs_dims > 0 ? cfg->obs_dims : cfg->obs_n;
    algo->action_n = cfg->action_n;
    algo->gamma = cfg->gamma > 0.0f ? cfg->gamma : 0.99f;
    algo->epsilon_start = cfg->epsilon > 0.0f ? cfg->epsilon : 0.1f;
    algo->epsilon_end = 0.02f;
    algo->epsilon_decay_steps = 200000;
    algo->epsilon = algo->epsilon_start;
    algo->deterministic = cfg->deterministic;
    algo->seed = cfg->seed;
    algo->rng_state = (unsigned int)(cfg->seed > 0 ? cfg->seed : 1);

    int input_dim = algo->obs_type == TFRL_SPACE_BOX ? algo->obs_dim : algo->obs_n;
    algo->xi = linear_create(input_dim, LSTM_DQN_HIDDEN);
    algo->hi = linear_create(LSTM_DQN_HIDDEN, LSTM_DQN_HIDDEN);
    algo->xf = linear_create(input_dim, LSTM_DQN_HIDDEN);
    algo->hf = linear_create(LSTM_DQN_HIDDEN, LSTM_DQN_HIDDEN);
    algo->xo = linear_create(input_dim, LSTM_DQN_HIDDEN);
    algo->ho = linear_create(LSTM_DQN_HIDDEN, LSTM_DQN_HIDDEN);
    algo->xg = linear_create(input_dim, LSTM_DQN_HIDDEN);
    algo->hg = linear_create(LSTM_DQN_HIDDEN, LSTM_DQN_HIDDEN);
    algo->q = linear_create(LSTM_DQN_HIDDEN, algo->action_n);
    if (!algo->xi || !algo->hi || !algo->xf || !algo->hf || !algo->xo ||
        !algo->ho || !algo->xg || !algo->hg || !algo->q) {
        lstm_dqn_destroy(algo);
        return out;
    }
    init_linear(algo->xi);
    init_linear(algo->hi);
    init_linear(algo->xf);
    init_linear(algo->hf);
    init_linear(algo->xo);
    init_linear(algo->ho);
    init_linear(algo->xg);
    init_linear(algo->hg);
    init_linear(algo->q);
    if (cfg->load_path) {
        load_linear(algo->xi, cfg->load_path, "xi");
        load_linear(algo->hi, cfg->load_path, "hi");
        load_linear(algo->xf, cfg->load_path, "xf");
        load_linear(algo->hf, cfg->load_path, "hf");
        load_linear(algo->xo, cfg->load_path, "xo");
        load_linear(algo->ho, cfg->load_path, "ho");
        load_linear(algo->xg, cfg->load_path, "xg");
        load_linear(algo->hg, cfg->load_path, "hg");
        load_linear(algo->q, cfg->load_path, "q");
    }
    Tensor *params[18] = {
        algo->xi->weight, algo->xi->bias,
        algo->hi->weight, algo->hi->bias,
        algo->xf->weight, algo->xf->bias,
        algo->hf->weight, algo->hf->bias,
        algo->xo->weight, algo->xo->bias,
        algo->ho->weight, algo->ho->bias,
        algo->xg->weight, algo->xg->bias,
        algo->hg->weight, algo->hg->bias,
        algo->q->weight, algo->q->bias
    };
    algo->opt = sgd_create(params, 18, cfg->lr > 0.0f ? cfg->lr : 0.0005f, 0.0f, 0.0f);
    if (!algo->opt) {
        lstm_dqn_destroy(algo);
        return out;
    }

    memset(algo->h_state, 0, sizeof(algo->h_state));
    memset(algo->c_state, 0, sizeof(algo->c_state));
    out.ctx = algo;
    out.vtable = &LSTM_DQN_VTABLE;
    return out;
}
