#include <math.h>
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
#include "tinyfin/ops_reshape.h"
#include "tinyfin/ops_slice.h"
#include "tinyfin/ops_softmax.h"
#include "tinyfin/ops_sub.h"
#include "tinyfin/tensor.h"

#include "algo_api.h"
#include "replay_buffer.h"

#define RAINBOW_N_STEP 3
#define RAINBOW_TARGET_UPDATE 100

typedef struct {
    Linear *adv;
    Linear *val;
    Linear *target_adv;
    Linear *target_val;
    Adam *opt;
    tfrl_replay_buffer *replay;
    int obs_n;
    int obs_dim;
    tfrl_space_type obs_type;
    int action_n;
    int atoms;
    float v_min;
    float v_max;
    float delta_z;
    float *support;
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
    int buf_count;
    int obs_buf[RAINBOW_N_STEP];
    int action_buf[RAINBOW_N_STEP];
    float reward_buf[RAINBOW_N_STEP];
    int done_buf[RAINBOW_N_STEP];
    int next_obs_buf[RAINBOW_N_STEP];
} tfrl_rainbow_algo;

static unsigned int rainbow_lcg_next(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float rainbow_rand_uniform(tfrl_rainbow_algo *algo) {
    if (algo->deterministic) {
        return (rainbow_lcg_next(&algo->rng_state) & 0x00FFFFFFu) / 16777215.0f;
    }
    return (float)rand() / (float)RAND_MAX;
}

static int rainbow_rand_int(tfrl_rainbow_algo *algo, int max) {
    if (max <= 0) return 0;
    if (algo->deterministic) {
        return (int)(rainbow_lcg_next(&algo->rng_state) % (unsigned int)max);
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

static Tensor *rainbow_forward_logits(tfrl_rainbow_algo *algo, Tensor *x, int target) {
    Linear *adv = target ? algo->target_adv : algo->adv;
    Linear *val = target ? algo->target_val : algo->val;
    Tensor *adv_logits = linear_forward(adv, x);
    Tensor *val_logits = linear_forward(val, x);
    if (!adv_logits || !val_logits) {
        if (adv_logits) tensor_free(adv_logits);
        if (val_logits) tensor_free(val_logits);
        return NULL;
    }
    Tensor *adv_mean = tensor_mean(adv_logits);
    if (!adv_mean) {
        tensor_free(adv_logits);
        tensor_free(val_logits);
        return NULL;
    }
    Tensor *adv_centered = tensor_sub(adv_logits, adv_mean);
    if (!adv_centered) {
        tensor_free(adv_mean);
        tensor_free(adv_logits);
        tensor_free(val_logits);
        return NULL;
    }
    Tensor *q_logits = tensor_add(val_logits, adv_centered);
    tensor_free(adv_centered);
    tensor_free(adv_mean);
    tensor_free(val_logits);
    tensor_free(adv_logits);
    return q_logits;
}

static float rainbow_expected_q(Tensor *q_logits, const tfrl_rainbow_algo *algo, int action_idx) {
    int shape[2] = {algo->action_n, algo->atoms};
    Tensor *view = tensor_reshape(q_logits, 2, shape);
    if (!view) return 0.0f;
    Tensor *row = tensor_slice(view, 0, action_idx, action_idx + 1);
    if (!row) {
        tensor_free(view);
        return 0.0f;
    }
    Tensor *probs = tensor_softmax_autograd(row);
    if (!probs) {
        tensor_free(row);
        tensor_free(view);
        return 0.0f;
    }
    int sshape[2] = {1, algo->atoms};
    Tensor *support = tensor_new(2, sshape);
    if (!support) {
        tensor_free(probs);
        tensor_free(row);
        tensor_free(view);
        return 0.0f;
    }
    for (int i = 0; i < algo->atoms; i++) {
        tensor_set_f32_at(support, (size_t)i, algo->support[i]);
    }
    Tensor *weighted = tensor_mul(probs, support);
    if (!weighted) {
        tensor_free(support);
        tensor_free(probs);
        tensor_free(row);
        tensor_free(view);
        return 0.0f;
    }
    Tensor *sum = tensor_sum(weighted);
    if (!sum) {
        tensor_free(weighted);
        tensor_free(support);
        tensor_free(probs);
        tensor_free(row);
        tensor_free(view);
        return 0.0f;
    }
    float value = tensor_get_f32_at(sum, 0);
    tensor_free(sum);
    tensor_free(weighted);
    tensor_free(support);
    tensor_free(probs);
    tensor_free(row);
    tensor_free(view);
    return value;
}

static int rainbow_argmax_action(Tensor *q_logits, const tfrl_rainbow_algo *algo) {
    int best = 0;
    float best_v = rainbow_expected_q(q_logits, algo, 0);
    for (int a = 1; a < algo->action_n; a++) {
        float v = rainbow_expected_q(q_logits, algo, a);
        if (v > best_v) {
            best_v = v;
            best = a;
        }
    }
    return best;
}

static tfrl_action rainbow_act(void *ctx, tfrl_obs obs) {
    tfrl_rainbow_algo *algo = (tfrl_rainbow_algo *)ctx;
    tfrl_action action = {0};
    float r = rainbow_rand_uniform(algo);
    if (r < algo->epsilon) {
        action.index = rainbow_rand_int(algo, algo->action_n);
        return action;
    }
    Tensor *x = obs_to_tensor(algo, obs);
    if (x) {
        Tensor *q_logits = rainbow_forward_logits(algo, x, 0);
        if (q_logits) {
            action.index = rainbow_argmax_action(q_logits, algo);
            tensor_free(q_logits);
        }
        tensor_free(x);
    }
    return action;
}

static void rainbow_act_batch(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions) {
    tfrl_rainbow_algo *algo = (tfrl_rainbow_algo *)ctx;
    if (!algo || !obs || !out_actions || count <= 0) return;

    for (int i = 0; i < count; i++) {
        float r = rainbow_rand_uniform(algo);
        if (r < algo->epsilon) {
            out_actions[i].index = rainbow_rand_int(algo, algo->action_n);
            continue;
        }
        Tensor *x = obs_to_tensor(algo, obs[i]);
        if (x) {
            Tensor *q_logits = rainbow_forward_logits(algo, x, 0);
            if (q_logits) {
                out_actions[i].index = rainbow_argmax_action(q_logits, algo);
                tensor_free(q_logits);
            } else {
                out_actions[i].index = rainbow_rand_int(algo, algo->action_n);
            }
            tensor_free(x);
        } else {
            out_actions[i].index = rainbow_rand_int(algo, algo->action_n);
        }
    }
}

static void rainbow_apply_update(tfrl_rainbow_algo *algo,
                                 tfrl_obs obs,
                                 int action_idx,
                                 float reward,
                                 int done,
                                 tfrl_obs next_obs,
                                 float gamma_n,
                                 float weight,
                                 float *out_td) {
    Tensor *x = obs_to_tensor(algo, obs);
    if (!x) return;
    Tensor *adv_logits = linear_forward(algo->adv, x);
    Tensor *val_logits = linear_forward(algo->val, x);
    if (!adv_logits || !val_logits) {
        if (adv_logits) tensor_free(adv_logits);
        if (val_logits) tensor_free(val_logits);
        tensor_free(x);
        return;
    }
    Tensor *adv_mean = tensor_mean(adv_logits);
    if (!adv_mean) {
        tensor_free(val_logits);
        tensor_free(adv_logits);
        tensor_free(x);
        return;
    }
    Tensor *adv_centered = tensor_sub(adv_logits, adv_mean);
    if (!adv_centered) {
        tensor_free(adv_mean);
        tensor_free(val_logits);
        tensor_free(adv_logits);
        tensor_free(x);
        return;
    }
    Tensor *q_logits = tensor_add(val_logits, adv_centered);
    if (!q_logits) {
        tensor_free(adv_centered);
        tensor_free(adv_mean);
        tensor_free(val_logits);
        tensor_free(adv_logits);
        tensor_free(x);
        return;
    }

    int atoms = algo->atoms;
    float target_dist[atoms];
    for (int i = 0; i < atoms; i++) target_dist[i] = 0.0f;

    if (done) {
        float z = reward;
        if (z < algo->v_min) z = algo->v_min;
        if (z > algo->v_max) z = algo->v_max;
        float b = (z - algo->v_min) / algo->delta_z;
        int l = (int)floorf(b);
        int u = (int)ceilf(b);
        if (l < 0) l = 0;
        if (u < 0) u = 0;
        if (l > atoms - 1) l = atoms - 1;
        if (u > atoms - 1) u = atoms - 1;
        if (l == u) {
            target_dist[l] = 1.0f;
        } else {
            target_dist[l] = (float)(u - b);
            target_dist[u] = (float)(b - l);
        }
    } else {
        Tensor *x_next = obs_to_tensor(algo, next_obs);
        if (!x_next) {
            tensor_free(q_logits);
            tensor_free(x);
            return;
        }
        Tensor *q_next_online = rainbow_forward_logits(algo, x_next, 0);
        if (!q_next_online) {
            tensor_free(x_next);
            tensor_free(q_logits);
            tensor_free(x);
            return;
        }
        int next_a = rainbow_argmax_action(q_next_online, algo);
        Tensor *q_next_target = rainbow_forward_logits(algo, x_next, 1);
        if (!q_next_target) {
            tensor_free(q_next_online);
            tensor_free(x_next);
            tensor_free(q_logits);
            tensor_free(x);
            return;
        }

        int shape[2] = {algo->action_n, algo->atoms};
        Tensor *q_next_view = tensor_reshape(q_next_target, 2, shape);
        if (!q_next_view) {
            tensor_free(q_next_target);
            tensor_free(q_next_online);
            tensor_free(x_next);
            tensor_free(q_logits);
            tensor_free(x);
            return;
        }
        Tensor *q_next_row = tensor_slice(q_next_view, 0, next_a, next_a + 1);
        if (!q_next_row) {
            tensor_free(q_next_view);
            tensor_free(q_next_target);
            tensor_free(q_next_online);
            tensor_free(x_next);
            tensor_free(q_logits);
            tensor_free(x);
            return;
        }
        Tensor *p_next = tensor_softmax_autograd(q_next_row);
        if (!p_next) {
            tensor_free(q_next_row);
            tensor_free(q_next_view);
            tensor_free(q_next_target);
            tensor_free(q_next_online);
            tensor_free(x_next);
            tensor_free(q_logits);
            tensor_free(x);
            return;
        }

        for (int j = 0; j < atoms; j++) {
            float p = tensor_get_f32_at(p_next, (size_t)j);
            float z = reward + gamma_n * algo->support[j];
            if (z < algo->v_min) z = algo->v_min;
            if (z > algo->v_max) z = algo->v_max;
            float b = (z - algo->v_min) / algo->delta_z;
            int l = (int)floorf(b);
            int u = (int)ceilf(b);
            if (l < 0) l = 0;
            if (u < 0) u = 0;
            if (l > atoms - 1) l = atoms - 1;
            if (u > atoms - 1) u = atoms - 1;
            if (l == u) {
                target_dist[l] += p;
            } else {
                target_dist[l] += p * (float)(u - b);
                target_dist[u] += p * (float)(b - l);
            }
        }

        tensor_free(p_next);
        tensor_free(q_next_row);
        tensor_free(q_next_view);
        tensor_free(q_next_target);
        tensor_free(q_next_online);
        tensor_free(x_next);
    }

    int shape[2] = {algo->action_n, algo->atoms};
    Tensor *q_view = tensor_reshape(q_logits, 2, shape);
    if (!q_view) {
        tensor_free(q_logits);
        tensor_free(x);
        return;
    }
    Tensor *q_row = tensor_slice(q_view, 0, action_idx, action_idx + 1);
    if (!q_row) {
        tensor_free(q_view);
        tensor_free(q_logits);
        tensor_free(x);
        return;
    }
    Tensor *probs = tensor_softmax_autograd(q_row);
    if (!probs) {
        tensor_free(q_row);
        tensor_free(q_view);
        tensor_free(q_logits);
        tensor_free(x);
        return;
    }
    Tensor *logp = tensor_log(probs);
    if (!logp) {
        tensor_free(probs);
        tensor_free(q_row);
        tensor_free(q_view);
        tensor_free(q_logits);
        tensor_free(x);
        return;
    }
    int tshape[2] = {1, algo->atoms};
    Tensor *target_t = tensor_new(2, tshape);
    if (!target_t) {
        tensor_free(logp);
        tensor_free(probs);
        tensor_free(q_row);
        tensor_free(q_view);
        tensor_free(q_logits);
        tensor_free(x);
        return;
    }
    for (int i = 0; i < atoms; i++) {
        tensor_set_f32_at(target_t, (size_t)i, target_dist[i]);
    }
    Tensor *mul = tensor_mul(target_t, logp);
    if (!mul) {
        tensor_free(target_t);
        tensor_free(logp);
        tensor_free(probs);
        tensor_free(q_row);
        tensor_free(q_view);
        tensor_free(q_logits);
        tensor_free(x);
        return;
    }
    Tensor *sum = tensor_sum(mul);
    if (!sum) {
        tensor_free(mul);
        tensor_free(target_t);
        tensor_free(logp);
        tensor_free(probs);
        tensor_free(q_row);
        tensor_free(q_view);
        tensor_free(q_logits);
        tensor_free(x);
        return;
    }
    Tensor *neg = tensor_new(1, (int[1]){1});
    if (!neg) {
        tensor_free(sum);
        tensor_free(mul);
        tensor_free(target_t);
        tensor_free(logp);
        tensor_free(probs);
        tensor_free(q_row);
        tensor_free(q_view);
        tensor_free(q_logits);
        tensor_free(x);
        return;
    }
    tensor_set_f32_at(neg, 0, -1.0f);
    Tensor *loss = tensor_mul(sum, neg);
    if (!loss) {
        tensor_free(neg);
        tensor_free(sum);
        tensor_free(mul);
        tensor_free(target_t);
        tensor_free(logp);
        tensor_free(probs);
        tensor_free(q_row);
        tensor_free(q_view);
        tensor_free(q_logits);
        tensor_free(x);
        return;
    }

    if (out_td) {
        float pred_mean = 0.0f;
        float target_mean = 0.0f;
        for (int i = 0; i < atoms; i++) {
            float p = tensor_get_f32_at(probs, (size_t)i);
            pred_mean += p * algo->support[i];
            target_mean += target_dist[i] * algo->support[i];
        }
        float td = target_mean - pred_mean;
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
    tensor_free(neg);
    tensor_free(sum);
    tensor_free(mul);
    tensor_free(target_t);
    tensor_free(logp);
    tensor_free(probs);
    tensor_free(q_row);
    tensor_free(q_view);
    tensor_free(q_logits);
    tensor_free(adv_centered);
    tensor_free(adv_mean);
    tensor_free(val_logits);
    tensor_free(adv_logits);
    tensor_free(x);

    algo->update_count++;
    if (algo->update_count % RAINBOW_TARGET_UPDATE == 0) {
        copy_linear(algo->target_adv, algo->adv);
        copy_linear(algo->target_val, algo->val);
    }
}

static void rainbow_update(void *ctx, const tfrl_transition *transition) {
    tfrl_rainbow_algo *algo = (tfrl_rainbow_algo *)ctx;
    if (!transition) return;
    algo->step_count++;
    int step_id = algo->step_count;

    if (algo->obs_type == TFRL_SPACE_BOX) {
        if (algo->replay) {
            tfrl_replay_push(algo->replay, transition, 1.0f);
            if (tfrl_replay_size(algo->replay) < algo->batch_size) return;
            if (step_id < algo->learning_starts) return;
            if (algo->train_every > 1 && (step_id % algo->train_every) != 0) return;
        }
        if (!algo->replay) {
            rainbow_apply_update(algo, transition->obs, transition->action.index, (float)transition->reward, transition->done, transition->next_obs, algo->gamma, 1.0f, NULL);
        }
        if (algo->replay && tfrl_replay_size(algo->replay) >= algo->batch_size) {
            for (int g = 0; g < algo->grad_steps; g++) {
                int *idx = (int *)calloc((size_t)algo->batch_size, sizeof(int));
                float *weights = (float *)calloc((size_t)algo->batch_size, sizeof(float));
                if (!idx || !weights) {
                    free(idx);
                    free(weights);
                    return;
                }
                if (algo->deterministic) {
                    tfrl_replay_sample_deterministic(algo->replay, algo->batch_size, idx, weights, algo->per_beta,
                                                     (unsigned int)(algo->seed + step_id + g));
                } else {
                    tfrl_replay_sample(algo->replay, algo->batch_size, idx, weights, algo->per_beta);
                }
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
        if (step_id < algo->learning_starts) return;
        if (algo->train_every > 1 && (step_id % algo->train_every) != 0) return;
        for (int g = 0; g < algo->grad_steps; g++) {
            int *idx = (int *)calloc((size_t)algo->batch_size, sizeof(int));
            float *weights = (float *)calloc((size_t)algo->batch_size, sizeof(float));
            if (!idx || !weights) {
                free(idx);
                free(weights);
                return;
            }
            if (algo->deterministic) {
                tfrl_replay_sample_deterministic(algo->replay, algo->batch_size, idx, weights, algo->per_beta,
                                                 (unsigned int)(algo->seed + step_id + g));
            } else {
                tfrl_replay_sample(algo->replay, algo->batch_size, idx, weights, algo->per_beta);
            }
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
}

static void rainbow_destroy(void *ctx) {
    tfrl_rainbow_algo *algo = (tfrl_rainbow_algo *)ctx;
    if (!algo) return;
    adam_free(algo->opt);
    tfrl_replay_free(algo->replay);
    tensor_free(algo->adv->weight);
    tensor_free(algo->adv->bias);
    tensor_free(algo->val->weight);
    tensor_free(algo->val->bias);
    tensor_free(algo->target_adv->weight);
    tensor_free(algo->target_adv->bias);
    tensor_free(algo->target_val->weight);
    tensor_free(algo->target_val->bias);
    free(algo->adv);
    free(algo->val);
    free(algo->target_adv);
    free(algo->target_val);
    free(algo->support);
    free(algo);
}

static void rainbow_save(void *ctx, const char *path) {
    tfrl_rainbow_algo *algo = (tfrl_rainbow_algo *)ctx;
    if (!algo) return;
    save_linear(algo->adv, path, "adv");
    save_linear(algo->val, path, "val");
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
    algo->atoms = cfg->c51_atoms > 0 ? cfg->c51_atoms : 51;
    algo->v_min = cfg->c51_vmin;
    algo->v_max = cfg->c51_vmax;
    if (algo->v_max <= algo->v_min) {
        algo->v_min = -10.0f;
        algo->v_max = 10.0f;
    }
    algo->delta_z = (algo->v_max - algo->v_min) / (float)(algo->atoms - 1);
    algo->support = (float *)calloc((size_t)algo->atoms, sizeof(float));
    if (!algo->support) {
        rainbow_destroy(algo);
        return out;
    }
    for (int i = 0; i < algo->atoms; i++) {
        algo->support[i] = algo->v_min + (float)i * algo->delta_z;
    }
    int out_dim = algo->action_n * algo->atoms;
    algo->adv = linear_create(input_dim, out_dim);
    algo->val = linear_create(input_dim, out_dim);
    algo->target_adv = linear_create(input_dim, out_dim);
    algo->target_val = linear_create(input_dim, out_dim);
    if (!algo->adv || !algo->val || !algo->target_adv || !algo->target_val) {
        rainbow_destroy(algo);
        return out;
    }
    init_linear(algo->adv);
    init_linear(algo->val);
    init_linear(algo->target_adv);
    init_linear(algo->target_val);
    copy_linear(algo->target_adv, algo->adv);
    copy_linear(algo->target_val, algo->val);
    if (cfg->load_path) {
        load_linear(algo->adv, cfg->load_path, "adv");
        load_linear(algo->val, cfg->load_path, "val");
        copy_linear(algo->target_adv, algo->adv);
        copy_linear(algo->target_val, algo->val);
    }
    Tensor *params[4] = {algo->adv->weight, algo->adv->bias, algo->val->weight, algo->val->bias};
    algo->opt = adam_create(params, 4, cfg->lr > 0.0f ? cfg->lr : 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);
    if (!algo->opt) {
        rainbow_destroy(algo);
        return out;
    }
    out.ctx = algo;
    out.vtable = &RAINBOW_VTABLE;
    return out;
}
