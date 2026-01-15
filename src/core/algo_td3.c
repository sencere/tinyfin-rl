#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyfin/autograd.h"
#include "tinyfin/nn.h"
#include "tinyfin/optim.h"
#include "tinyfin/ops_activation.h"
#include "tinyfin/ops_add.h"
#include "tinyfin/ops_log.h"
#include "tinyfin/ops_mul.h"
#include "tinyfin/ops_reduce.h"
#include "tinyfin/ops_softmax.h"
#include "tinyfin/ops_sub.h"
#include "tinyfin/tensor.h"

#include "algo_api.h"
#include "replay_buffer.h"

#define TD3_TARGET_UPDATE 100
#define TD3_POLICY_DELAY 2
#define TD3_TARGET_NOISE 0.2f
#define TD3_NOISE_CLIP 0.5f
#define TD3_EXPLORATION_NOISE 0.1f

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    Linear *policy;
    Linear *q1;
    Linear *q2;
    Linear *tpolicy;
    Linear *tq1;
    Linear *tq2;
    Adam *policy_opt;
    Adam *q_opt;
    tfrl_replay_buffer *replay;
    int obs_n;
    int action_n;
    int obs_dim;
    int action_dim;
    tfrl_space_type obs_type;
    tfrl_space_type action_type;
    float action_low;
    float action_high;
    float gamma;
    int batch_size;
    int train_every;
    int learning_starts;
    int grad_steps;
    float per_alpha;
    float per_beta;
    int step_count;
    int update_count;
    int seed;
    int deterministic;
    unsigned int rng_state;
} tfrl_td3_algo;

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

static unsigned int td3_lcg_next(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float td3_rand_uniform(tfrl_td3_algo *algo) {
    if (algo->deterministic) {
        return (td3_lcg_next(&algo->rng_state) & 0x00FFFFFFu) / 16777215.0f;
    }
    return (float)rand() / (float)RAND_MAX;
}

static float rand_normal(tfrl_td3_algo *algo) {
    float u1 = td3_rand_uniform(algo);
    float u2 = td3_rand_uniform(algo);
    if (u1 <= 0.0f) u1 = 1e-6f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void action_scale_bias(const tfrl_td3_algo *algo, float *scale, float *bias) {
    float low = algo->action_low;
    float high = algo->action_high;
    if (high <= low) {
        low = -1.0f;
        high = 1.0f;
    }
    *scale = (high - low) * 0.5f;
    *bias = (high + low) * 0.5f;
}

static Tensor *obs_to_tensor(const tfrl_td3_algo *algo, tfrl_obs obs) {
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

static Tensor *action_to_tensor(const tfrl_td3_algo *algo, tfrl_action action) {
    if (algo->action_type == TFRL_SPACE_BOX) {
        int shape[2] = {1, algo->action_dim};
        Tensor *t = tensor_zeros(2, shape);
        if (!t) return NULL;
        int n = action.data_len < algo->action_dim ? action.data_len : algo->action_dim;
        for (int i = 0; i < n; i++) {
            tensor_set_f32_at(t, (size_t)i, action.data[i]);
        }
        return t;
    }
    return one_hot(algo->action_n, action.index);
}

static Tensor *concat_obs_action(const tfrl_td3_algo *algo, Tensor *obs_t, Tensor *act_t) {
    int shape[2] = {1, algo->obs_dim + algo->action_dim};
    Tensor *t = tensor_zeros(2, shape);
    if (!t) return NULL;
    for (int i = 0; i < algo->obs_dim; i++) {
        tensor_set_f32_at(t, (size_t)i, tensor_get_f32_at(obs_t, (size_t)i));
    }
    for (int j = 0; j < algo->action_dim; j++) {
        size_t idx = (size_t)algo->obs_dim + (size_t)j;
        tensor_set_f32_at(t, idx, tensor_get_f32_at(act_t, (size_t)j));
    }
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

static tfrl_action td3_act(void *ctx, tfrl_obs obs) {
    tfrl_td3_algo *algo = (tfrl_td3_algo *)ctx;
    tfrl_action action = {0};
    Tensor *x = obs_to_tensor(algo, obs);
    if (!x) return action;
    Tensor *logits = linear_forward(algo->policy, x);
    if (algo->action_type == TFRL_SPACE_BOX) {
        Tensor *a = tensor_tanh(logits);
        action.data_len = algo->action_dim;
        float scale = 1.0f;
        float bias = 0.0f;
        float low = algo->action_low;
        float high = algo->action_high;
        if (high <= low) {
            low = -1.0f;
            high = 1.0f;
        }
        action_scale_bias(algo, &scale, &bias);
        for (int i = 0; i < algo->action_dim; i++) {
            float v = tensor_get_f32_at(a, (size_t)i);
            float scaled = v * scale + bias;
            float noise = rand_normal(algo) * TD3_EXPLORATION_NOISE;
            float noisy = clampf(scaled + noise, low, high);
            action.data[i] = noisy;
        }
        tensor_free(a);
    } else {
        Tensor *probs = tensor_softmax_autograd(logits);
        action.index = sample_action(probs);
        tensor_free(probs);
    }
    tensor_free(logits);
    tensor_free(x);
    return action;
}

static void td3_update(void *ctx, const tfrl_transition *transition) {
    tfrl_td3_algo *algo = (tfrl_td3_algo *)ctx;
    if (!transition) return;
    algo->step_count++;
    int step_id = algo->step_count;

    if (algo->replay) {
        tfrl_replay_push(algo->replay, transition, 1.0f);
        if (tfrl_replay_size(algo->replay) < algo->batch_size) return;
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

            int update_id = algo->update_count + 1;
            Tensor *q_loss_sum = NULL;
            Tensor *policy_loss_sum = NULL;
            for (int i = 0; i < algo->batch_size; i++) {
            const tfrl_transition *tr = tfrl_replay_get(algo->replay, idx[i]);
            if (!tr) continue;

            Tensor *x = obs_to_tensor(algo, tr->obs);
            Tensor *x_next = obs_to_tensor(algo, tr->next_obs);
            if (!x || !x_next) {
                tensor_free(x);
                tensor_free(x_next);
                continue;
            }

            Tensor *q1 = NULL;
            Tensor *q2 = NULL;
            if (algo->action_type == TFRL_SPACE_BOX) {
                Tensor *a_t = action_to_tensor(algo, tr->action);
                Tensor *qa_in = concat_obs_action(algo, x, a_t);
                tensor_free(a_t);
                q1 = linear_forward(algo->q1, qa_in);
                q2 = linear_forward(algo->q2, qa_in);
                tensor_free(qa_in);
            } else {
                q1 = linear_forward(algo->q1, x);
                q2 = linear_forward(algo->q2, x);
            }

            float target = (float)tr->reward;
            if (!tr->done) {
                if (algo->action_type == TFRL_SPACE_BOX) {
                    Tensor *next_logits = linear_forward(algo->tpolicy, x_next);
                    Tensor *next_tanh = tensor_tanh(next_logits);
                    int shape[2] = {1, algo->action_dim};
                    Tensor *next_a = tensor_new(2, shape);
                    float scale = 1.0f;
                    float bias = 0.0f;
                    float low = algo->action_low;
                    float high = algo->action_high;
                    if (high <= low) {
                        low = -1.0f;
                        high = 1.0f;
                    }
                    action_scale_bias(algo, &scale, &bias);
                    for (int j = 0; j < algo->action_dim; j++) {
                        float v = tensor_get_f32_at(next_tanh, (size_t)j);
                        float scaled = v * scale + bias;
                        float noise = clampf(rand_normal(algo) * TD3_TARGET_NOISE, -TD3_NOISE_CLIP, TD3_NOISE_CLIP);
                        float noisy = clampf(scaled + noise, low, high);
                        tensor_set_f32_at(next_a, (size_t)j, noisy);
                    }
                    Tensor *tq_in = concat_obs_action(algo, x_next, next_a);
                    Tensor *tq1 = linear_forward(algo->tq1, tq_in);
                    Tensor *tq2 = linear_forward(algo->tq2, tq_in);
                    float q1v = tensor_get_f32_at(tq1, 0);
                    float q2v = tensor_get_f32_at(tq2, 0);
                    float minq = q1v < q2v ? q1v : q2v;
                    target = (float)tr->reward + algo->gamma * minq;
                    tensor_free(tq2);
                    tensor_free(tq1);
                    tensor_free(tq_in);
                    tensor_free(next_a);
                    tensor_free(next_tanh);
                    tensor_free(next_logits);
                } else {
                    Tensor *next_logits = linear_forward(algo->tpolicy, x_next);
                    Tensor *next_probs = tensor_softmax_autograd(next_logits);
                    int next_a = argmax_tensor(next_probs);
                    Tensor *tq1 = linear_forward(algo->tq1, x_next);
                    Tensor *tq2 = linear_forward(algo->tq2, x_next);
                    float q1v = tensor_get_f32_at(tq1, (size_t)next_a);
                    float q2v = tensor_get_f32_at(tq2, (size_t)next_a);
                    float minq = q1v < q2v ? q1v : q2v;
                    target = (float)tr->reward + algo->gamma * minq;
                    tensor_free(tq2);
                    tensor_free(tq1);
                    tensor_free(next_probs);
                    tensor_free(next_logits);
                }
            }

            Tensor *q1_val = NULL;
            Tensor *q2_val = NULL;
            if (algo->action_type == TFRL_SPACE_BOX) {
                q1_val = q1;
                q2_val = q2;
            } else {
                Tensor *action_one = one_hot(algo->action_n, tr->action.index);
                Tensor *q1_sel = tensor_mul(q1, action_one);
                Tensor *q2_sel = tensor_mul(q2, action_one);
                q1_val = tensor_sum(q1_sel);
                q2_val = tensor_sum(q2_sel);
                tensor_free(q2_sel);
                tensor_free(q1_sel);
                tensor_free(action_one);
                tensor_free(q2);
                tensor_free(q1);
            }
            Tensor *target_t = tensor_new(1, (int[1]){1});
            tensor_set_f32_at(target_t, 0, target);
            Tensor *diff1 = tensor_sub(q1_val, target_t);
            Tensor *diff2 = tensor_sub(q2_val, target_t);
            Tensor *loss1 = tensor_mul(diff1, diff1);
            Tensor *loss2 = tensor_mul(diff2, diff2);
            Tensor *q_loss = tensor_add(loss1, loss2);

            if (weights[i] != 1.0f) {
                Tensor *w_t = tensor_new(1, (int[1]){1});
                tensor_set_f32_at(w_t, 0, weights[i]);
                Tensor *weighted = tensor_mul(q_loss, w_t);
                tensor_free(q_loss);
                tensor_free(w_t);
                q_loss = weighted;
            }

            if (!q_loss_sum) {
                q_loss_sum = q_loss;
            } else {
                Tensor *tmp = tensor_add(q_loss_sum, q_loss);
                tensor_free(q_loss_sum);
                tensor_free(q_loss);
                q_loss_sum = tmp;
            }

            if (update_id % TD3_POLICY_DELAY == 0) {
                Tensor *policy_loss = NULL;
                if (algo->action_type == TFRL_SPACE_BOX) {
                    Tensor *pi_logits = linear_forward(algo->policy, x);
                    Tensor *pi_action = tensor_tanh(pi_logits);
                    int shape[2] = {1, algo->action_dim};
                    Tensor *scale_t = tensor_new(2, shape);
                    Tensor *bias_t = tensor_new(2, shape);
                    float scale = 1.0f;
                    float bias = 0.0f;
                    action_scale_bias(algo, &scale, &bias);
                    for (int j = 0; j < algo->action_dim; j++) {
                        tensor_set_f32_at(scale_t, (size_t)j, scale);
                        tensor_set_f32_at(bias_t, (size_t)j, bias);
                    }
                    Tensor *pi_scaled = tensor_add(tensor_mul(pi_action, scale_t), bias_t);
                    Tensor *qa_in = concat_obs_action(algo, x, pi_scaled);
                    Tensor *q1_pi = linear_forward(algo->q1, qa_in);
                    Tensor *neg = tensor_new(1, (int[1]){1});
                    tensor_set_f32_at(neg, 0, -1.0f);
                    policy_loss = tensor_mul(q1_pi, neg);
                    tensor_free(neg);
                    tensor_free(q1_pi);
                    tensor_free(qa_in);
                    tensor_free(pi_scaled);
                    tensor_free(bias_t);
                    tensor_free(scale_t);
                    tensor_free(pi_action);
                    tensor_free(pi_logits);
                } else {
                    Tensor *pi_logits = linear_forward(algo->policy, x);
                    Tensor *pi_probs = tensor_softmax_autograd(pi_logits);
                    Tensor *q1_pi = linear_forward(algo->q1, x);
                    int shape[2] = {1, algo->action_n};
                    Tensor *q1_const = tensor_new(2, shape);
                    for (int a = 0; a < algo->action_n; a++) {
                        tensor_set_f32_at(q1_const, (size_t)a, tensor_get_f32_at(q1_pi, (size_t)a));
                    }
                    Tensor *policy_elem = tensor_mul(pi_probs, q1_const);
                    Tensor *policy_sum = tensor_sum(policy_elem);
                    Tensor *neg = tensor_new(1, (int[1]){1});
                    tensor_set_f32_at(neg, 0, -1.0f);
                    policy_loss = tensor_mul(policy_sum, neg);

                    tensor_free(neg);
                    tensor_free(policy_sum);
                    tensor_free(policy_elem);
                    tensor_free(q1_const);
                    tensor_free(q1_pi);
                    tensor_free(pi_probs);
                    tensor_free(pi_logits);
                }

                if (weights[i] != 1.0f) {
                    Tensor *w_t = tensor_new(1, (int[1]){1});
                    tensor_set_f32_at(w_t, 0, weights[i]);
                    Tensor *weighted = tensor_mul(policy_loss, w_t);
                    tensor_free(policy_loss);
                    tensor_free(w_t);
                    policy_loss = weighted;
                }

                if (!policy_loss_sum) {
                    policy_loss_sum = policy_loss;
                } else {
                    Tensor *tmp = tensor_add(policy_loss_sum, policy_loss);
                    tensor_free(policy_loss_sum);
                    tensor_free(policy_loss);
                    policy_loss_sum = tmp;
                }
            }

            tensor_free(q2_val);
            tensor_free(q1_val);
            tensor_free(loss2);
            tensor_free(loss1);
            tensor_free(diff2);
            tensor_free(diff1);
            tensor_free(target_t);
            tensor_free(x_next);
            tensor_free(x);
        }

            if (q_loss_sum) {
                adam_zero_grad(algo->q_opt);
                tensor_backward(q_loss_sum);
                adam_step(algo->q_opt, 0.0f);
                tensor_free(q_loss_sum);
            }
            if (policy_loss_sum) {
                adam_zero_grad(algo->policy_opt);
                tensor_backward(policy_loss_sum);
                adam_step(algo->policy_opt, 0.0f);
                tensor_free(policy_loss_sum);
            }

            free(idx);
            free(weights);
            algo->update_count++;
            if (update_id % TD3_TARGET_UPDATE == 0) {
                copy_linear(algo->tpolicy, algo->policy);
                copy_linear(algo->tq1, algo->q1);
                copy_linear(algo->tq2, algo->q2);
            }
        }
        return;
    }

    Tensor *x = obs_to_tensor(algo, transition->obs);
    Tensor *x_next = obs_to_tensor(algo, transition->next_obs);
    if (!x || !x_next) {
        tensor_free(x);
        tensor_free(x_next);
        return;
    }

    Tensor *q1 = NULL;
    Tensor *q2 = NULL;
    if (algo->action_type == TFRL_SPACE_BOX) {
        Tensor *a_t = action_to_tensor(algo, transition->action);
        Tensor *qa_in = concat_obs_action(algo, x, a_t);
        tensor_free(a_t);
        q1 = linear_forward(algo->q1, qa_in);
        q2 = linear_forward(algo->q2, qa_in);
        tensor_free(qa_in);
    } else {
        q1 = linear_forward(algo->q1, x);
        q2 = linear_forward(algo->q2, x);
    }

    float target = (float)transition->reward;
    if (!transition->done) {
        if (algo->action_type == TFRL_SPACE_BOX) {
            Tensor *next_logits = linear_forward(algo->tpolicy, x_next);
            Tensor *next_tanh = tensor_tanh(next_logits);
            int shape[2] = {1, algo->action_dim};
            Tensor *next_a = tensor_new(2, shape);
            float scale = 1.0f;
            float bias = 0.0f;
            float low = algo->action_low;
            float high = algo->action_high;
            if (high <= low) {
                low = -1.0f;
                high = 1.0f;
            }
            action_scale_bias(algo, &scale, &bias);
            for (int j = 0; j < algo->action_dim; j++) {
                float v = tensor_get_f32_at(next_tanh, (size_t)j);
                float scaled = v * scale + bias;
                float noise = clampf(rand_normal(algo) * TD3_TARGET_NOISE, -TD3_NOISE_CLIP, TD3_NOISE_CLIP);
                float noisy = clampf(scaled + noise, low, high);
                tensor_set_f32_at(next_a, (size_t)j, noisy);
            }
            Tensor *tq_in = concat_obs_action(algo, x_next, next_a);
            Tensor *tq1 = linear_forward(algo->tq1, tq_in);
            Tensor *tq2 = linear_forward(algo->tq2, tq_in);
            float q1v = tensor_get_f32_at(tq1, 0);
            float q2v = tensor_get_f32_at(tq2, 0);
            float minq = q1v < q2v ? q1v : q2v;
            target = (float)transition->reward + algo->gamma * minq;
            tensor_free(tq2);
            tensor_free(tq1);
            tensor_free(tq_in);
            tensor_free(next_a);
            tensor_free(next_tanh);
            tensor_free(next_logits);
        } else {
            Tensor *next_logits = linear_forward(algo->tpolicy, x_next);
            Tensor *next_probs = tensor_softmax_autograd(next_logits);
            int next_a = argmax_tensor(next_probs);
            Tensor *tq1 = linear_forward(algo->tq1, x_next);
            Tensor *tq2 = linear_forward(algo->tq2, x_next);
            float q1v = tensor_get_f32_at(tq1, (size_t)next_a);
            float q2v = tensor_get_f32_at(tq2, (size_t)next_a);
            float minq = q1v < q2v ? q1v : q2v;
            target = (float)transition->reward + algo->gamma * minq;
            tensor_free(tq2);
            tensor_free(tq1);
            tensor_free(next_probs);
            tensor_free(next_logits);
        }
    }

    Tensor *q1_val = NULL;
    Tensor *q2_val = NULL;
    if (algo->action_type == TFRL_SPACE_BOX) {
        q1_val = q1;
        q2_val = q2;
    } else {
        Tensor *action_one = one_hot(algo->action_n, transition->action.index);
        Tensor *q1_sel = tensor_mul(q1, action_one);
        Tensor *q2_sel = tensor_mul(q2, action_one);
        q1_val = tensor_sum(q1_sel);
        q2_val = tensor_sum(q2_sel);
        tensor_free(q2_sel);
        tensor_free(q1_sel);
        tensor_free(action_one);
        tensor_free(q2);
        tensor_free(q1);
    }
    Tensor *target_t = tensor_new(1, (int[1]){1});
    tensor_set_f32_at(target_t, 0, target);
    Tensor *diff1 = tensor_sub(q1_val, target_t);
    Tensor *diff2 = tensor_sub(q2_val, target_t);
    Tensor *loss1 = tensor_mul(diff1, diff1);
    Tensor *loss2 = tensor_mul(diff2, diff2);
    Tensor *q_loss = tensor_add(loss1, loss2);

    adam_zero_grad(algo->q_opt);
    tensor_backward(q_loss);
    adam_step(algo->q_opt, 0.0f);

    tensor_free(q_loss);
    tensor_free(loss2);
    tensor_free(loss1);
    tensor_free(diff2);
    tensor_free(diff1);
    tensor_free(target_t);
    tensor_free(q2_val);
    tensor_free(q1_val);

    int update_id = algo->update_count + 1;
    algo->update_count++;
    if (update_id % TD3_POLICY_DELAY == 0) {
        if (algo->action_type == TFRL_SPACE_BOX) {
            Tensor *pi_logits = linear_forward(algo->policy, x);
            Tensor *pi_action = tensor_tanh(pi_logits);
            int shape[2] = {1, algo->action_dim};
            Tensor *scale_t = tensor_new(2, shape);
            Tensor *bias_t = tensor_new(2, shape);
            float scale = 1.0f;
            float bias = 0.0f;
            action_scale_bias(algo, &scale, &bias);
            for (int j = 0; j < algo->action_dim; j++) {
                tensor_set_f32_at(scale_t, (size_t)j, scale);
                tensor_set_f32_at(bias_t, (size_t)j, bias);
            }
            Tensor *pi_scaled = tensor_add(tensor_mul(pi_action, scale_t), bias_t);
            Tensor *qa_in = concat_obs_action(algo, x, pi_scaled);
            Tensor *q1_pi = linear_forward(algo->q1, qa_in);
            Tensor *neg = tensor_new(1, (int[1]){1});
            tensor_set_f32_at(neg, 0, -1.0f);
            Tensor *policy_loss = tensor_mul(q1_pi, neg);

            adam_zero_grad(algo->policy_opt);
            tensor_backward(policy_loss);
            adam_step(algo->policy_opt, 0.0f);

            tensor_free(policy_loss);
            tensor_free(neg);
            tensor_free(q1_pi);
            tensor_free(qa_in);
            tensor_free(pi_scaled);
            tensor_free(bias_t);
            tensor_free(scale_t);
            tensor_free(pi_action);
            tensor_free(pi_logits);
        } else {
            Tensor *pi_logits = linear_forward(algo->policy, x);
            Tensor *pi_probs = tensor_softmax_autograd(pi_logits);
            Tensor *q1_pi = linear_forward(algo->q1, x);
            int shape[2] = {1, algo->action_n};
            Tensor *q1_const = tensor_new(2, shape);
            for (int a = 0; a < algo->action_n; a++) {
                tensor_set_f32_at(q1_const, (size_t)a, tensor_get_f32_at(q1_pi, (size_t)a));
            }
            Tensor *policy_elem = tensor_mul(pi_probs, q1_const);
            Tensor *policy_sum = tensor_sum(policy_elem);
            Tensor *neg = tensor_new(1, (int[1]){1});
            tensor_set_f32_at(neg, 0, -1.0f);
            Tensor *policy_loss = tensor_mul(policy_sum, neg);

            adam_zero_grad(algo->policy_opt);
            tensor_backward(policy_loss);
            adam_step(algo->policy_opt, 0.0f);

            tensor_free(policy_loss);
            tensor_free(neg);
            tensor_free(policy_sum);
            tensor_free(policy_elem);
            tensor_free(q1_const);
            tensor_free(q1_pi);
            tensor_free(pi_probs);
            tensor_free(pi_logits);
        }
    }

    tensor_free(x_next);
    tensor_free(x);

    if (update_id % TD3_TARGET_UPDATE == 0) {
        copy_linear(algo->tpolicy, algo->policy);
        copy_linear(algo->tq1, algo->q1);
        copy_linear(algo->tq2, algo->q2);
    }
}

static void td3_destroy(void *ctx) {
    tfrl_td3_algo *algo = (tfrl_td3_algo *)ctx;
    if (!algo) return;
    adam_free(algo->policy_opt);
    adam_free(algo->q_opt);
    tfrl_replay_free(algo->replay);
    tensor_free(algo->policy->weight);
    tensor_free(algo->policy->bias);
    tensor_free(algo->q1->weight);
    tensor_free(algo->q1->bias);
    tensor_free(algo->q2->weight);
    tensor_free(algo->q2->bias);
    tensor_free(algo->tpolicy->weight);
    tensor_free(algo->tpolicy->bias);
    tensor_free(algo->tq1->weight);
    tensor_free(algo->tq1->bias);
    tensor_free(algo->tq2->weight);
    tensor_free(algo->tq2->bias);
    free(algo->policy);
    free(algo->q1);
    free(algo->q2);
    free(algo->tpolicy);
    free(algo->tq1);
    free(algo->tq2);
    free(algo);
}

static void td3_save(void *ctx, const char *path) {
    tfrl_td3_algo *algo = (tfrl_td3_algo *)ctx;
    if (!algo) return;
    save_linear(algo->policy, path, "pi");
    save_linear(algo->q1, path, "q1");
    save_linear(algo->q2, path, "q2");
}

static const tfrl_algo_vtable TD3_VTABLE = {
    .act = td3_act,
    .act_batch = NULL,
    .update = td3_update,
    .save = td3_save,
    .destroy = td3_destroy,
};

tfrl_algo tfrl_algo_td3_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->obs_n <= 0 || cfg->action_n <= 0) return out;
    autograd_set_enabled(1);
    tfrl_td3_algo *algo = (tfrl_td3_algo *)calloc(1, sizeof(tfrl_td3_algo));
    if (!algo) return out;
    algo->obs_n = cfg->obs_n;
    algo->action_n = cfg->action_n;
    algo->obs_type = cfg->obs_type;
    algo->action_type = cfg->action_type;
    algo->obs_dim = cfg->obs_dims > 0 ? cfg->obs_dims : cfg->obs_n;
    algo->action_dim = cfg->action_dims > 0 ? cfg->action_dims : cfg->action_n;
    algo->action_low = (float)cfg->action_low;
    algo->action_high = (float)cfg->action_high;
    algo->gamma = cfg->gamma > 0.0f ? cfg->gamma : 0.99f;
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
    if (algo->action_type == TFRL_SPACE_BOX) {
        algo->policy = linear_create(algo->obs_dim, algo->action_dim);
        algo->q1 = linear_create(algo->obs_dim + algo->action_dim, 1);
        algo->q2 = linear_create(algo->obs_dim + algo->action_dim, 1);
        algo->tpolicy = linear_create(algo->obs_dim, algo->action_dim);
        algo->tq1 = linear_create(algo->obs_dim + algo->action_dim, 1);
        algo->tq2 = linear_create(algo->obs_dim + algo->action_dim, 1);
    } else {
        algo->policy = linear_create(algo->obs_n, algo->action_n);
        algo->q1 = linear_create(algo->obs_n, algo->action_n);
        algo->q2 = linear_create(algo->obs_n, algo->action_n);
        algo->tpolicy = linear_create(algo->obs_n, algo->action_n);
        algo->tq1 = linear_create(algo->obs_n, algo->action_n);
        algo->tq2 = linear_create(algo->obs_n, algo->action_n);
    }
    if (!algo->policy || !algo->q1 || !algo->q2 || !algo->tpolicy || !algo->tq1 || !algo->tq2) {
        td3_destroy(algo);
        return out;
    }
    init_linear(algo->policy);
    init_linear(algo->q1);
    init_linear(algo->q2);
    init_linear(algo->tpolicy);
    init_linear(algo->tq1);
    init_linear(algo->tq2);
    copy_linear(algo->tpolicy, algo->policy);
    copy_linear(algo->tq1, algo->q1);
    copy_linear(algo->tq2, algo->q2);
    if (cfg->load_path) {
        load_linear(algo->policy, cfg->load_path, "pi");
        load_linear(algo->q1, cfg->load_path, "q1");
        load_linear(algo->q2, cfg->load_path, "q2");
        copy_linear(algo->tpolicy, algo->policy);
        copy_linear(algo->tq1, algo->q1);
        copy_linear(algo->tq2, algo->q2);
    }
    Tensor *policy_params[2] = {algo->policy->weight, algo->policy->bias};
    algo->policy_opt = adam_create(policy_params, 2, cfg->lr > 0.0f ? cfg->lr : 0.0003f, 0.9f, 0.999f, 1e-8f, 0.0f);
    Tensor *q_params[4] = {algo->q1->weight, algo->q1->bias, algo->q2->weight, algo->q2->bias};
    algo->q_opt = adam_create(q_params, 4, cfg->lr > 0.0f ? cfg->lr : 0.0003f, 0.9f, 0.999f, 1e-8f, 0.0f);
    if (!algo->policy_opt || !algo->q_opt) {
        td3_destroy(algo);
        return out;
    }
    out.ctx = algo;
    out.vtable = &TD3_VTABLE;
    return out;
}
