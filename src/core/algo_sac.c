#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyfin/autograd.h"
#include "tinyfin/nn.h"
#include "tinyfin/optim.h"
#include "tinyfin/ops_activation.h"
#include "tinyfin/ops_add.h"
#include "tinyfin/ops_clamp.h"
#include "tinyfin/ops_div.h"
#include "tinyfin/ops_exp.h"
#include "tinyfin/ops_log.h"
#include "tinyfin/ops_mul.h"
#include "tinyfin/ops_reduce.h"
#include "tinyfin/ops_slice.h"
#include "tinyfin/ops_softmax.h"
#include "tinyfin/ops_sub.h"
#include "tinyfin/tensor.h"

#include "algo_api.h"
#include "replay_buffer.h"

#define SAC_TARGET_UPDATE 100
#define SAC_LOG_STD_MIN -5.0f
#define SAC_LOG_STD_MAX 2.0f
#define SAC_LOG_PROB_EPS 1e-6f

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    Linear *policy;
    Linear *q1;
    Linear *q2;
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
    float alpha;
    int batch_size;
    float per_alpha;
    float per_beta;
    int step_count;
    int seed;
    int deterministic;
    unsigned int rng_state;
} tfrl_sac_algo;

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

static unsigned int sac_lcg_next(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float sac_rand_uniform(tfrl_sac_algo *algo) {
    if (algo->deterministic) {
        return (sac_lcg_next(&algo->rng_state) & 0x00FFFFFFu) / 16777215.0f;
    }
    return (float)rand() / (float)RAND_MAX;
}

static float rand_normal(tfrl_sac_algo *algo) {
    float u1 = sac_rand_uniform(algo);
    float u2 = sac_rand_uniform(algo);
    if (u1 <= 0.0f) u1 = 1e-6f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void action_scale_bias(const tfrl_sac_algo *algo, float *scale, float *bias) {
    float low = algo->action_low;
    float high = algo->action_high;
    if (high <= low) {
        low = -1.0f;
        high = 1.0f;
    }
    *scale = (high - low) * 0.5f;
    *bias = (high + low) * 0.5f;
}

static Tensor *obs_to_tensor(const tfrl_sac_algo *algo, tfrl_obs obs) {
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

static Tensor *action_to_tensor(const tfrl_sac_algo *algo, tfrl_action action) {
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

static Tensor *concat_obs_action(const tfrl_sac_algo *algo, Tensor *obs_t, Tensor *act_t) {
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

static void sample_action_stats(const tfrl_sac_algo *algo, tfrl_obs obs, float *out_action, float *out_logp) {
    Tensor *x = obs_to_tensor(algo, obs);
    if (!x) return;
    Tensor *out = linear_forward(algo->policy, x);
    Tensor *mean_t = tensor_slice(out, 1, 0, algo->action_dim);
    Tensor *log_std_t = tensor_slice(out, 1, algo->action_dim, algo->action_dim * 2);
    if (!mean_t || !log_std_t) {
        tensor_free(log_std_t);
        tensor_free(mean_t);
        tensor_free(out);
        tensor_free(x);
        return;
    }

    float scale = 1.0f;
    float bias = 0.0f;
    action_scale_bias(algo, &scale, &bias);
    float log_scale = logf(scale);

    float logp = 0.0f;
    for (int i = 0; i < algo->action_dim; i++) {
        float mean = tensor_get_f32_at(mean_t, (size_t)i);
        float log_std = tensor_get_f32_at(log_std_t, (size_t)i);
        if (log_std < SAC_LOG_STD_MIN) log_std = SAC_LOG_STD_MIN;
        if (log_std > SAC_LOG_STD_MAX) log_std = SAC_LOG_STD_MAX;
        float std = expf(log_std);
        float eps = rand_normal(algo);
        float u = mean + std * eps;
        float a = tanhf(u);
        out_action[i] = a * scale + bias;

        float logp_i = -0.5f * (eps * eps) - log_std - 0.5f * logf(2.0f * (float)M_PI);
        float corr = logf(1.0f - a * a + SAC_LOG_PROB_EPS);
        logp += (logp_i - corr - log_scale);
    }
    if (out_logp) *out_logp = logp;

    tensor_free(log_std_t);
    tensor_free(mean_t);
    tensor_free(out);
    tensor_free(x);
}

static Tensor *policy_action_logp(tfrl_sac_algo *algo, Tensor *obs_t, Tensor **out_logp) {
    Tensor *out = linear_forward(algo->policy, obs_t);
    if (!out) return NULL;
    Tensor *mean_t = tensor_slice(out, 1, 0, algo->action_dim);
    Tensor *log_std_t = tensor_slice(out, 1, algo->action_dim, algo->action_dim * 2);
    if (!mean_t || !log_std_t) {
        tensor_free(log_std_t);
        tensor_free(mean_t);
        tensor_free(out);
        return NULL;
    }

    Tensor *log_std_c = tensor_clamp(log_std_t, SAC_LOG_STD_MIN, SAC_LOG_STD_MAX);
    Tensor *std = tensor_exp(log_std_c);
    int shape[2] = {1, algo->action_dim};
    Tensor *eps = tensor_new(2, shape);
    if (eps) {
        for (int i = 0; i < algo->action_dim; i++) {
            tensor_set_f32_at(eps, (size_t)i, rand_normal(algo));
        }
    }
    Tensor *std_eps = tensor_mul(std, eps);
    Tensor *u = tensor_add(mean_t, std_eps);
    Tensor *a = tensor_tanh(u);

    float scale = 1.0f;
    float bias = 0.0f;
    action_scale_bias(algo, &scale, &bias);
    Tensor *scale_t = tensor_new(2, shape);
    Tensor *bias_t = tensor_new(2, shape);
    if (scale_t && bias_t) {
        for (int i = 0; i < algo->action_dim; i++) {
            tensor_set_f32_at(scale_t, (size_t)i, scale);
            tensor_set_f32_at(bias_t, (size_t)i, bias);
        }
    }
    Tensor *a_scaled_unbiased = tensor_mul(a, scale_t);
    Tensor *a_scaled = tensor_add(a_scaled_unbiased, bias_t);
    tensor_free(a_scaled_unbiased);

    Tensor *diff = tensor_sub(u, mean_t);
    Tensor *diff2 = tensor_mul(diff, diff);
    Tensor *var = tensor_mul(std, std);
    Tensor *frac = tensor_div(diff2, var);
    Tensor *neg_half = tensor_new(1, (int[1]){1});
    tensor_set_f32_at(neg_half, 0, -0.5f);
    Tensor *term1 = tensor_mul(frac, neg_half);
    Tensor *neg_one = tensor_new(1, (int[1]){1});
    tensor_set_f32_at(neg_one, 0, -1.0f);
    Tensor *term2 = tensor_mul(log_std_c, neg_one);

    Tensor *const_t = tensor_new(2, shape);
    float const_v = -0.5f * logf(2.0f * (float)M_PI);
    for (int i = 0; i < algo->action_dim; i++) {
        tensor_set_f32_at(const_t, (size_t)i, const_v);
    }
    Tensor *logp_dim = tensor_add(term1, term2);
    Tensor *logp_dim_const = tensor_add(logp_dim, const_t);
    Tensor *logp = tensor_sum(logp_dim_const);

    Tensor *a2 = tensor_mul(a, a);
    Tensor *one = tensor_new(2, shape);
    Tensor *eps_t = tensor_new(2, shape);
    for (int i = 0; i < algo->action_dim; i++) {
        tensor_set_f32_at(one, (size_t)i, 1.0f);
        tensor_set_f32_at(eps_t, (size_t)i, SAC_LOG_PROB_EPS);
    }
    Tensor *one_minus = tensor_sub(one, a2);
    Tensor *one_minus_eps = tensor_add(one_minus, eps_t);
    Tensor *log_det = tensor_log(one_minus_eps);
    Tensor *corr = tensor_sum(log_det);
    Tensor *logp_corr = tensor_sub(logp, corr);

    float scale_log = logf(scale);
    Tensor *scale_log_t = tensor_new(1, (int[1]){1});
    tensor_set_f32_at(scale_log_t, 0, scale_log * (float)algo->action_dim);
    Tensor *logp_final = tensor_sub(logp_corr, scale_log_t);

    if (out_logp) *out_logp = logp_final;

    tensor_free(scale_log_t);
    tensor_free(logp_corr);
    tensor_free(corr);
    tensor_free(log_det);
    tensor_free(one_minus_eps);
    tensor_free(one_minus);
    tensor_free(eps_t);
    tensor_free(one);
    tensor_free(a2);
    tensor_free(logp_dim_const);
    tensor_free(logp_dim);
    tensor_free(const_t);
    tensor_free(term2);
    tensor_free(term1);
    tensor_free(neg_one);
    tensor_free(neg_half);
    tensor_free(frac);
    tensor_free(var);
    tensor_free(diff2);
    tensor_free(diff);
    tensor_free(bias_t);
    tensor_free(scale_t);
    tensor_free(a);
    tensor_free(u);
    tensor_free(std_eps);
    tensor_free(eps);
    tensor_free(std);
    tensor_free(log_std_c);
    tensor_free(log_std_t);
    tensor_free(mean_t);
    tensor_free(out);
    return a_scaled;
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

static tfrl_action sac_act(void *ctx, tfrl_obs obs) {
    tfrl_sac_algo *algo = (tfrl_sac_algo *)ctx;
    tfrl_action action = {0};
    if (algo->action_type == TFRL_SPACE_BOX) {
        action.data_len = algo->action_dim;
        sample_action_stats(algo, obs, action.data, NULL);
        return action;
    } else {
        Tensor *x = obs_to_tensor(algo, obs);
        if (!x) return action;
        Tensor *logits = linear_forward(algo->policy, x);
        Tensor *probs = tensor_softmax_autograd(logits);
        action.index = sample_action(probs);
        tensor_free(probs);
        tensor_free(logits);
        tensor_free(x);
    }
    return action;
}

static void sac_update(void *ctx, const tfrl_transition *transition) {
    tfrl_sac_algo *algo = (tfrl_sac_algo *)ctx;
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
                    float next_buf[4] = {0};
                    float logp = 0.0f;
                    sample_action_stats(algo, tr->next_obs, next_buf, &logp);
                    tfrl_action next_action = {.data_len = algo->action_dim};
                    for (int j = 0; j < algo->action_dim; j++) next_action.data[j] = next_buf[j];
                    Tensor *next_a = action_to_tensor(algo, next_action);
                    Tensor *tq_in = concat_obs_action(algo, x_next, next_a);
                    Tensor *tq1 = linear_forward(algo->tq1, tq_in);
                    Tensor *tq2 = linear_forward(algo->tq2, tq_in);
                    float q1v = tensor_get_f32_at(tq1, 0);
                    float q2v = tensor_get_f32_at(tq2, 0);
                    float minq = q1v < q2v ? q1v : q2v;
                    target = (float)tr->reward + algo->gamma * (minq - algo->alpha * logp);
                    tensor_free(tq2);
                    tensor_free(tq1);
                    tensor_free(tq_in);
                    tensor_free(next_a);
                } else {
                    Tensor *next_logits = linear_forward(algo->policy, x_next);
                    Tensor *next_probs = tensor_softmax_autograd(next_logits);
                    Tensor *next_logp = tensor_log(next_probs);
                    Tensor *tq1 = linear_forward(algo->tq1, x_next);
                    Tensor *tq2 = linear_forward(algo->tq2, x_next);
                    float soft_value = 0.0f;
                    for (int a = 0; a < algo->action_n; a++) {
                        float p = tensor_get_f32_at(next_probs, (size_t)a);
                        float logp = tensor_get_f32_at(next_logp, (size_t)a);
                        float q1v = tensor_get_f32_at(tq1, (size_t)a);
                        float q2v = tensor_get_f32_at(tq2, (size_t)a);
                        float minq = q1v < q2v ? q1v : q2v;
                        soft_value += p * (minq - algo->alpha * logp);
                    }
                    target = (float)tr->reward + algo->gamma * soft_value;
                    tensor_free(tq2);
                    tensor_free(tq1);
                    tensor_free(next_logp);
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

            Tensor *policy_loss = NULL;
            if (algo->action_type == TFRL_SPACE_BOX) {
                Tensor *logp = NULL;
                Tensor *pi_action = policy_action_logp(algo, x, &logp);
                Tensor *qa_in = concat_obs_action(algo, x, pi_action);
                Tensor *q1_pi = linear_forward(algo->q1, qa_in);
                Tensor *alpha_t = tensor_new(1, (int[1]){1});
                tensor_set_f32_at(alpha_t, 0, algo->alpha);
                Tensor *alpha_logp = tensor_mul(logp, alpha_t);
                policy_loss = tensor_sub(alpha_logp, q1_pi);
                tensor_free(alpha_logp);
                tensor_free(alpha_t);
                tensor_free(q1_pi);
                tensor_free(qa_in);
                tensor_free(pi_action);
                tensor_free(logp);
            } else {
                Tensor *pi_logits = linear_forward(algo->policy, x);
                Tensor *pi_probs = tensor_softmax_autograd(pi_logits);
                Tensor *pi_logp = tensor_log(pi_probs);
                Tensor *q1_pi = linear_forward(algo->q1, x);
                Tensor *q2_pi = linear_forward(algo->q2, x);

                int shape[2] = {1, algo->action_n};
                Tensor *min_q = tensor_new(2, shape);
                for (int a = 0; a < algo->action_n; a++) {
                    float q1v = tensor_get_f32_at(q1_pi, (size_t)a);
                    float q2v = tensor_get_f32_at(q2_pi, (size_t)a);
                    float minv = q1v < q2v ? q1v : q2v;
                    tensor_set_f32_at(min_q, (size_t)a, minv);
                }
                Tensor *alpha_t = tensor_new(1, (int[1]){1});
                tensor_set_f32_at(alpha_t, 0, algo->alpha);
                Tensor *alpha_logp = tensor_mul(pi_logp, alpha_t);
                Tensor *alpha_logp_sub = tensor_sub(alpha_logp, min_q);
                Tensor *policy_elem = tensor_mul(pi_probs, alpha_logp_sub);
                policy_loss = tensor_sum(policy_elem);

                tensor_free(policy_elem);
                tensor_free(alpha_logp_sub);
                tensor_free(alpha_logp);
                tensor_free(alpha_t);
                tensor_free(min_q);
                tensor_free(q2_pi);
                tensor_free(q1_pi);
                tensor_free(pi_logp);
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
        algo->step_count++;
        if (algo->step_count % SAC_TARGET_UPDATE == 0) {
            copy_linear(algo->tq1, algo->q1);
            copy_linear(algo->tq2, algo->q2);
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
            float action_buf[4] = {0};
            float logp = 0.0f;
            sample_action_stats(algo, transition->next_obs, action_buf, &logp);
            tfrl_action next_action = {.data_len = algo->action_dim};
            for (int i = 0; i < algo->action_dim; i++) next_action.data[i] = action_buf[i];
            Tensor *a_t = action_to_tensor(algo, next_action);
            Tensor *tq_in = concat_obs_action(algo, x_next, a_t);
            Tensor *tq1 = linear_forward(algo->tq1, tq_in);
            Tensor *tq2 = linear_forward(algo->tq2, tq_in);
            float q1v = tensor_get_f32_at(tq1, 0);
            float q2v = tensor_get_f32_at(tq2, 0);
            float minq = q1v < q2v ? q1v : q2v;
            target = (float)transition->reward + algo->gamma * (minq - algo->alpha * logp);
            tensor_free(tq2);
            tensor_free(tq1);
            tensor_free(tq_in);
            tensor_free(a_t);
        } else {
            Tensor *next_logits = linear_forward(algo->policy, x_next);
            Tensor *next_probs = tensor_softmax_autograd(next_logits);
            Tensor *next_logp = tensor_log(next_probs);
            Tensor *tq1 = linear_forward(algo->tq1, x_next);
            Tensor *tq2 = linear_forward(algo->tq2, x_next);
            float soft_value = 0.0f;
            for (int a = 0; a < algo->action_n; a++) {
                float p = tensor_get_f32_at(next_probs, (size_t)a);
                float logp = tensor_get_f32_at(next_logp, (size_t)a);
                float q1v = tensor_get_f32_at(tq1, (size_t)a);
                float q2v = tensor_get_f32_at(tq2, (size_t)a);
                float minq = q1v < q2v ? q1v : q2v;
                soft_value += p * (minq - algo->alpha * logp);
            }
            target = (float)transition->reward + algo->gamma * soft_value;
            tensor_free(tq2);
            tensor_free(tq1);
            tensor_free(next_logp);
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

    Tensor *policy_loss = NULL;
    if (algo->action_type == TFRL_SPACE_BOX) {
        Tensor *logp = NULL;
        Tensor *pi_action = policy_action_logp(algo, x, &logp);
        Tensor *qa_in = concat_obs_action(algo, x, pi_action);
        Tensor *q1_pi = linear_forward(algo->q1, qa_in);
        Tensor *alpha_t = tensor_new(1, (int[1]){1});
        tensor_set_f32_at(alpha_t, 0, algo->alpha);
        Tensor *alpha_logp = tensor_mul(logp, alpha_t);
        policy_loss = tensor_sub(alpha_logp, q1_pi);
        tensor_free(alpha_logp);
        tensor_free(alpha_t);
        tensor_free(q1_pi);
        tensor_free(qa_in);
        tensor_free(pi_action);
        tensor_free(logp);
    } else {
        Tensor *pi_logits = linear_forward(algo->policy, x);
        Tensor *pi_probs = tensor_softmax_autograd(pi_logits);
        Tensor *pi_logp = tensor_log(pi_probs);
        Tensor *q1_pi = linear_forward(algo->q1, x);
        Tensor *q2_pi = linear_forward(algo->q2, x);

        int shape[2] = {1, algo->action_n};
        Tensor *min_q = tensor_new(2, shape);
        for (int a = 0; a < algo->action_n; a++) {
            float q1v = tensor_get_f32_at(q1_pi, (size_t)a);
            float q2v = tensor_get_f32_at(q2_pi, (size_t)a);
            float minv = q1v < q2v ? q1v : q2v;
            tensor_set_f32_at(min_q, (size_t)a, minv);
        }
        Tensor *alpha_t = tensor_new(1, (int[1]){1});
        tensor_set_f32_at(alpha_t, 0, algo->alpha);
        Tensor *alpha_logp = tensor_mul(pi_logp, alpha_t);
        Tensor *alpha_logp_sub = tensor_sub(alpha_logp, min_q);
        Tensor *policy_elem = tensor_mul(pi_probs, alpha_logp_sub);
        policy_loss = tensor_sum(policy_elem);

        tensor_free(policy_elem);
        tensor_free(alpha_logp_sub);
        tensor_free(alpha_logp);
        tensor_free(alpha_t);
        tensor_free(min_q);
        tensor_free(q2_pi);
        tensor_free(q1_pi);
        tensor_free(pi_logp);
        tensor_free(pi_probs);
        tensor_free(pi_logits);
    }

    adam_zero_grad(algo->policy_opt);
    tensor_backward(policy_loss);
    adam_step(algo->policy_opt, 0.0f);

    tensor_free(policy_loss);

    tensor_free(x_next);
    tensor_free(x);

    algo->step_count++;
    if (algo->step_count % SAC_TARGET_UPDATE == 0) {
        copy_linear(algo->tq1, algo->q1);
        copy_linear(algo->tq2, algo->q2);
    }
}

static void sac_destroy(void *ctx) {
    tfrl_sac_algo *algo = (tfrl_sac_algo *)ctx;
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
    tensor_free(algo->tq1->weight);
    tensor_free(algo->tq1->bias);
    tensor_free(algo->tq2->weight);
    tensor_free(algo->tq2->bias);
    free(algo->policy);
    free(algo->q1);
    free(algo->q2);
    free(algo->tq1);
    free(algo->tq2);
    free(algo);
}

static void sac_save(void *ctx, const char *path) {
    tfrl_sac_algo *algo = (tfrl_sac_algo *)ctx;
    if (!algo) return;
    save_linear(algo->policy, path, "pi");
    save_linear(algo->q1, path, "q1");
    save_linear(algo->q2, path, "q2");
}

static const tfrl_algo_vtable SAC_VTABLE = {
    .act = sac_act,
    .act_batch = NULL,
    .update = sac_update,
    .save = sac_save,
    .destroy = sac_destroy,
};

tfrl_algo tfrl_algo_sac_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->obs_n <= 0 || cfg->action_n <= 0) return out;
    autograd_set_enabled(1);
    tfrl_sac_algo *algo = (tfrl_sac_algo *)calloc(1, sizeof(tfrl_sac_algo));
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
    algo->alpha = cfg->entropy_coef > 0.0f ? cfg->entropy_coef : 0.2f;
    algo->batch_size = cfg->batch_size > 0 ? cfg->batch_size : 32;
    algo->per_alpha = cfg->per_alpha;
    algo->per_beta = cfg->per_beta;
    algo->seed = cfg->seed;
    algo->deterministic = cfg->deterministic;
    algo->rng_state = (unsigned int)(cfg->seed > 0 ? cfg->seed : 1);
    if (cfg->replay_size > 0) {
        algo->replay = tfrl_replay_create(cfg->replay_size, algo->per_alpha);
    }
    if (algo->action_type == TFRL_SPACE_BOX) {
        algo->policy = linear_create(algo->obs_dim, algo->action_dim * 2);
        algo->q1 = linear_create(algo->obs_dim + algo->action_dim, 1);
        algo->q2 = linear_create(algo->obs_dim + algo->action_dim, 1);
        algo->tq1 = linear_create(algo->obs_dim + algo->action_dim, 1);
        algo->tq2 = linear_create(algo->obs_dim + algo->action_dim, 1);
    } else {
        algo->policy = linear_create(algo->obs_n, algo->action_n);
        algo->q1 = linear_create(algo->obs_n, algo->action_n);
        algo->q2 = linear_create(algo->obs_n, algo->action_n);
        algo->tq1 = linear_create(algo->obs_n, algo->action_n);
        algo->tq2 = linear_create(algo->obs_n, algo->action_n);
    }
    if (!algo->policy || !algo->q1 || !algo->q2 || !algo->tq1 || !algo->tq2) {
        sac_destroy(algo);
        return out;
    }
    init_linear(algo->policy);
    init_linear(algo->q1);
    init_linear(algo->q2);
    init_linear(algo->tq1);
    init_linear(algo->tq2);
    copy_linear(algo->tq1, algo->q1);
    copy_linear(algo->tq2, algo->q2);
    if (cfg->load_path) {
        load_linear(algo->policy, cfg->load_path, "pi");
        load_linear(algo->q1, cfg->load_path, "q1");
        load_linear(algo->q2, cfg->load_path, "q2");
        copy_linear(algo->tq1, algo->q1);
        copy_linear(algo->tq2, algo->q2);
    }
    Tensor *policy_params[2] = {algo->policy->weight, algo->policy->bias};
    algo->policy_opt = adam_create(policy_params, 2, cfg->lr > 0.0f ? cfg->lr : 0.0003f, 0.9f, 0.999f, 1e-8f, 0.0f);
    Tensor *q_params[4] = {algo->q1->weight, algo->q1->bias, algo->q2->weight, algo->q2->bias};
    algo->q_opt = adam_create(q_params, 4, cfg->lr > 0.0f ? cfg->lr : 0.0003f, 0.9f, 0.999f, 1e-8f, 0.0f);
    if (!algo->policy_opt || !algo->q_opt) {
        sac_destroy(algo);
        return out;
    }
    out.ctx = algo;
    out.vtable = &SAC_VTABLE;
    return out;
}
