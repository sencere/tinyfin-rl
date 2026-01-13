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

#define RAINBOW_N_STEP 3
#define RAINBOW_TARGET_UPDATE 100

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
    int step_count;
    int buf_count;
    int obs_buf[RAINBOW_N_STEP];
    int action_buf[RAINBOW_N_STEP];
    float reward_buf[RAINBOW_N_STEP];
    int done_buf[RAINBOW_N_STEP];
    int next_obs_buf[RAINBOW_N_STEP];
} tfrl_rainbow_algo;

static Tensor *one_hot(int n, int idx) {
    int shape[2] = {1, n};
    Tensor *t = tensor_zeros(2, shape);
    if (!t) return NULL;
    if (idx >= 0 && idx < n) {
        tensor_set_f32_at(t, (size_t)idx, 1.0f);
    }
    return t;
}

static Tensor *obs_to_tensor(const tfrl_rainbow_algo *algo, tfrl_obs obs) {
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

static tfrl_action rainbow_act(void *ctx, tfrl_obs obs) {
    tfrl_rainbow_algo *algo = (tfrl_rainbow_algo *)ctx;
    tfrl_action action = {0};
    float r = (float)rand() / (float)RAND_MAX;
    if (r < algo->epsilon) {
        action.index = rand() % algo->action_n;
        return action;
    }
    Tensor *x = obs_to_tensor(algo, obs);
    Tensor *q_values = linear_forward(algo->q, x);
    action.index = argmax_tensor(q_values);
    tensor_free(q_values);
    tensor_free(x);
    return action;
}

static void rainbow_act_batch(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions) {
    tfrl_rainbow_algo *algo = (tfrl_rainbow_algo *)ctx;
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

static void rainbow_apply_update(tfrl_rainbow_algo *algo, tfrl_obs obs, int action_idx, float reward, int done, tfrl_obs next_obs, float gamma_n, float weight, float *out_td) {
    Tensor *x = obs_to_tensor(algo, obs);
    Tensor *q_values = linear_forward(algo->q, x);
    Tensor *action_one = one_hot(algo->action_n, action_idx);
    Tensor *q_sel = tensor_mul(q_values, action_one);
    Tensor *q_val = tensor_sum(q_sel);

    float target = reward;
    if (!done) {
        Tensor *x_next = obs_to_tensor(algo, next_obs);
        Tensor *q_next_online = linear_forward(algo->q, x_next);
        int next_a = argmax_tensor(q_next_online);
        Tensor *q_next_target = linear_forward(algo->target_q, x_next);
        float max_q = tensor_get_f32_at(q_next_target, (size_t)next_a);
        target = reward + gamma_n * max_q;
        tensor_free(q_next_target);
        tensor_free(q_next_online);
        tensor_free(x_next);
    }

    Tensor *target_t = tensor_new(1, (int[1]){1});
    tensor_set_f32_at(target_t, 0, target);
    Tensor *diff = tensor_sub(q_val, target_t);
    Tensor *loss = tensor_mul(diff, diff);
    if (out_td) {
        float q_pred = tensor_get_f32_at(q_val, 0);
        float td = target - q_pred;
        *out_td = td >= 0.0f ? td : -td;
    }
    if (weight > 0.0f && weight != 1.0f) {
        Tensor *w_t = tensor_new(1, (int[1]){1});
        tensor_set_f32_at(w_t, 0, weight);
        Tensor *weighted = tensor_mul(loss, w_t);
        tensor_free(loss);
        tensor_free(w_t);
        loss = weighted;
    }

    adam_zero_grad(algo->opt);
    tensor_backward(loss);
    adam_step(algo->opt, 0.0f);

    tensor_free(loss);
    tensor_free(diff);
    tensor_free(target_t);
    tensor_free(q_val);
    tensor_free(q_sel);
    tensor_free(action_one);
    tensor_free(q_values);
    tensor_free(x);

    algo->step_count++;
    if (algo->step_count % RAINBOW_TARGET_UPDATE == 0) {
        copy_linear(algo->target_q, algo->q);
    }
}

static void rainbow_update(void *ctx, const tfrl_transition *transition) {
    tfrl_rainbow_algo *algo = (tfrl_rainbow_algo *)ctx;
    if (!transition) return;

    if (algo->obs_type == TFRL_SPACE_BOX) {
        if (algo->replay) {
            tfrl_replay_push(algo->replay, transition, 1.0f);
        } else {
            rainbow_apply_update(algo, transition->obs, transition->action.index, (float)transition->reward, transition->done, transition->next_obs, algo->gamma, 1.0f, NULL);
        }
        if (algo->replay && tfrl_replay_size(algo->replay) >= algo->batch_size) {
            int *idx = (int *)calloc((size_t)algo->batch_size, sizeof(int));
            float *weights = (float *)calloc((size_t)algo->batch_size, sizeof(float));
            if (!idx || !weights) {
                free(idx);
                free(weights);
                return;
            }
            tfrl_replay_sample(algo->replay, algo->batch_size, idx, weights, algo->per_beta);
            for (int i = 0; i < algo->batch_size; i++) {
                const tfrl_transition *tr = tfrl_replay_get(algo->replay, idx[i]);
                if (!tr) continue;
                float td = 0.0f;
                rainbow_apply_update(algo, tr->obs, tr->action.index, (float)tr->reward, tr->done, tr->next_obs, algo->gamma, weights[i], &td);
                tfrl_replay_update_priority(algo->replay, idx[i], td + 1e-3f);
            }
            free(idx);
            free(weights);
        }
        return;
    }

    if (algo->buf_count < RAINBOW_N_STEP) {
        int idx = algo->buf_count;
        algo->obs_buf[idx] = transition->obs.index;
        algo->action_buf[idx] = transition->action.index;
        algo->reward_buf[idx] = (float)transition->reward;
        algo->done_buf[idx] = transition->done;
        algo->next_obs_buf[idx] = transition->next_obs.index;
        algo->buf_count++;
    }

    if (algo->buf_count == RAINBOW_N_STEP) {
        float g = 0.0f;
        float gamma_pow = 1.0f;
        int done = 0;
        int last_next_obs = algo->next_obs_buf[0];
        for (int i = 0; i < RAINBOW_N_STEP; i++) {
            g += gamma_pow * algo->reward_buf[i];
            last_next_obs = algo->next_obs_buf[i];
            done = algo->done_buf[i];
            if (done) break;
            gamma_pow *= algo->gamma;
        }
        if (algo->replay) {
            tfrl_transition tr = {
                .obs = {.index = algo->obs_buf[0]},
                .action = {.index = algo->action_buf[0]},
                .reward = g,
                .next_obs = {.index = last_next_obs},
                .done = done,
            };
            tfrl_replay_push(algo->replay, &tr, 1.0f);
        } else {
            tfrl_obs obs = {.index = algo->obs_buf[0]};
            tfrl_obs next_obs = {.index = last_next_obs};
            rainbow_apply_update(algo, obs, algo->action_buf[0], g, done, next_obs, gamma_pow, 1.0f, NULL);
        }
        for (int i = 1; i < algo->buf_count; i++) {
            algo->obs_buf[i - 1] = algo->obs_buf[i];
            algo->action_buf[i - 1] = algo->action_buf[i];
            algo->reward_buf[i - 1] = algo->reward_buf[i];
            algo->done_buf[i - 1] = algo->done_buf[i];
            algo->next_obs_buf[i - 1] = algo->next_obs_buf[i];
        }
        algo->buf_count = RAINBOW_N_STEP - 1;
    }

    if (transition->done && algo->buf_count > 0) {
        int upto = algo->buf_count;
        for (int start = 0; start < upto; start++) {
            float g = 0.0f;
            float gamma_pow = 1.0f;
            int done = 0;
            int last_next_obs = algo->next_obs_buf[start];
            for (int i = start; i < upto; i++) {
                g += gamma_pow * algo->reward_buf[i];
                last_next_obs = algo->next_obs_buf[i];
                done = algo->done_buf[i];
                if (done) break;
                gamma_pow *= algo->gamma;
            }
            if (algo->replay) {
                tfrl_transition tr = {
                    .obs = {.index = algo->obs_buf[start]},
                    .action = {.index = algo->action_buf[start]},
                    .reward = g,
                    .next_obs = {.index = last_next_obs},
                    .done = done,
                };
                tfrl_replay_push(algo->replay, &tr, 1.0f);
            } else {
                tfrl_obs obs = {.index = algo->obs_buf[start]};
                tfrl_obs next_obs = {.index = last_next_obs};
                rainbow_apply_update(algo, obs, algo->action_buf[start], g, done, next_obs, gamma_pow, 1.0f, NULL);
            }
        }
        algo->buf_count = 0;
    }

    if (algo->replay && tfrl_replay_size(algo->replay) >= algo->batch_size) {
        int *idx = (int *)calloc((size_t)algo->batch_size, sizeof(int));
        float *weights = (float *)calloc((size_t)algo->batch_size, sizeof(float));
        if (!idx || !weights) {
            free(idx);
            free(weights);
            return;
        }
        tfrl_replay_sample(algo->replay, algo->batch_size, idx, weights, algo->per_beta);
        float gamma_n = powf(algo->gamma, (float)RAINBOW_N_STEP);
        for (int i = 0; i < algo->batch_size; i++) {
            const tfrl_transition *tr = tfrl_replay_get(algo->replay, idx[i]);
            if (!tr) continue;
            float td = 0.0f;
            rainbow_apply_update(algo, tr->obs, tr->action.index, (float)tr->reward, tr->done, tr->next_obs, gamma_n, weights[i], &td);
            tfrl_replay_update_priority(algo->replay, idx[i], td + 1e-3f);
        }
        free(idx);
        free(weights);
    }
}

static void rainbow_destroy(void *ctx) {
    tfrl_rainbow_algo *algo = (tfrl_rainbow_algo *)ctx;
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

static void rainbow_save(void *ctx, const char *path) {
    tfrl_rainbow_algo *algo = (tfrl_rainbow_algo *)ctx;
    if (!algo) return;
    save_linear(algo->q, path, "q");
}

static const tfrl_algo_vtable RAINBOW_VTABLE = {
    .act = rainbow_act,
    .act_batch = rainbow_act_batch,
    .update = rainbow_update,
    .save = rainbow_save,
    .destroy = rainbow_destroy,
};

tfrl_algo tfrl_algo_rainbow_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->obs_n <= 0 || cfg->action_n <= 0) return out;
    autograd_set_enabled(1);
    tfrl_rainbow_algo *algo = (tfrl_rainbow_algo *)calloc(1, sizeof(tfrl_rainbow_algo));
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
    if (cfg->replay_size > 0) {
        algo->replay = tfrl_replay_create(cfg->replay_size, algo->per_alpha);
    }
    int input_dim = algo->obs_type == TFRL_SPACE_BOX ? algo->obs_dim : algo->obs_n;
    algo->q = linear_create(input_dim, algo->action_n);
    algo->target_q = linear_create(input_dim, algo->action_n);
    if (!algo->q || !algo->target_q) {
        rainbow_destroy(algo);
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
        rainbow_destroy(algo);
        return out;
    }
    out.ctx = algo;
    out.vtable = &RAINBOW_VTABLE;
    return out;
}
