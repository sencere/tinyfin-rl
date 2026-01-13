#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "tinyfin/tensor.h"
#include "tinyfin/autograd.h"
#include "tinyfin/nn.h"
#include "tinyfin/optim.h"
#include "tinyfin/ops_add.h"
#include "tinyfin/ops_sub.h"
#include "tinyfin/ops_mul.h"
#include "tinyfin/ops_reduce.h"
#include "tinyfin/ops_log.h"
#include "tinyfin/ops_exp.h"
#include "tinyfin/ops_clamp.h"
#include "tinyfin/ops_softmax.h"

#include "rl_train.h"

static Tensor *one_hot(int n, int idx) {
    int shape[2] = {1, n};
    Tensor *t = tensor_zeros(2, shape);
    if (!t) return NULL;
    if (idx >= 0 && idx < n) {
        tensor_set_f32_at(t, (size_t)idx, 1.0f);
    }
    return t;
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

static float tensor_item(Tensor *t) {
    return t->size > 0 ? tensor_get_f32_at(t, 0) : 0.0f;
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
int tfrl_train_dqn(tfrl_env *env, const tfrl_dqn_config *cfg, tfrl_train_metrics *out_metrics) {
    if (!env || !cfg) return TFRL_ERR_INVALID;
    if (cfg->obs_n <= 0 || cfg->action_n <= 0) return TFRL_ERR_INVALID;
    autograd_set_enabled(1);

    Linear *q = linear_create(cfg->obs_n, cfg->action_n);
    if (!q) return TFRL_ERR;
    init_linear(q);
    Tensor *params[2] = {q->weight, q->bias};
    SGD *opt = sgd_create(params, 2, cfg->lr, 0.0f, 0.0f);
    if (!opt) return TFRL_ERR;

    if (cfg->load_path) {
        load_linear(q, cfg->load_path, "q");
    }

    tfrl_value obs = {0};
    tfrl_step step = {0};
    if (tfrl_env_reset(env, &obs) != TFRL_OK) return TFRL_ERR;
    int total_steps = cfg->steps > 0 ? cfg->steps : 1000;
    float return_sum = 0.0f;
    float loss_sum = 0.0f;
    int episode_steps = 0;

    for (int step_idx = 0; step_idx < total_steps; step_idx++) {
        Tensor *x = one_hot(cfg->obs_n, (int)obs.as.i64);
        Tensor *q_values = linear_forward(q, x);
        int action = 0;
        float r = (float)rand() / (float)RAND_MAX;
        if (r < cfg->epsilon) {
            action = rand() % cfg->action_n;
        } else {
            action = argmax_tensor(q_values);
        }

        tfrl_value act = {.type = TFRL_VALUE_I64, .as.i64 = action};
        if (tfrl_env_step(env, &act, &step) != TFRL_OK) break;

        float reward = (float)step.reward;
        return_sum += reward;
        episode_steps += 1;

        float target = reward;
        if (!tfrl_step_done(&step)) {
            Tensor *x_next = one_hot(cfg->obs_n, (int)step.observation.as.i64);
            Tensor *q_next = linear_forward(q, x_next);
            int next_a = argmax_tensor(q_next);
            float max_q = tensor_get_f32_at(q_next, (size_t)next_a);
            target = reward + cfg->gamma * max_q;
            tensor_free(x_next);
            tensor_free(q_next);
        }

        Tensor *action_one = one_hot(cfg->action_n, action);
        Tensor *q_sel = tensor_mul(q_values, action_one);
        Tensor *q_val = tensor_sum(q_sel);
        int target_shape[1] = {1};
        Tensor *target_t = tensor_new(1, target_shape);
        tensor_set_f32_at(target_t, 0, target);
        Tensor *diff = tensor_sub(q_val, target_t);
        Tensor *loss = tensor_mul(diff, diff);

        sgd_zero_grad(opt);
        tensor_backward(loss);
        sgd_step(opt, 0.0f);
        loss_sum += tensor_item(loss);

        tensor_free(action_one);
        tensor_free(q_sel);
        tensor_free(q_val);
        tensor_free(target_t);
        tensor_free(diff);
        tensor_free(loss);
        tensor_free(q_values);
        tensor_free(x);

        if (tfrl_step_done(&step)) {
            tfrl_env_reset(env, &obs);
            episode_steps = 0;
        } else {
            obs = step.observation;
        }
    }

    if (cfg->eval_steps > 0) {
        tfrl_env_reset(env, &obs);
        for (int i = 0; i < cfg->eval_steps; i++) {
            Tensor *x = one_hot(cfg->obs_n, (int)obs.as.i64);
            Tensor *q_values = linear_forward(q, x);
            int action = argmax_tensor(q_values);
            tfrl_value act = {.type = TFRL_VALUE_I64, .as.i64 = action};
            if (tfrl_env_step(env, &act, &step) != TFRL_OK) break;
            tensor_free(q_values);
            tensor_free(x);
            if (tfrl_step_done(&step)) {
                tfrl_env_reset(env, &obs);
            } else {
                obs = step.observation;
            }
        }
    }

    if (cfg->save_path) {
        save_linear(q, cfg->save_path, "q");
    }

    if (out_metrics) {
        out_metrics->return_mean = return_sum / (float)total_steps;
        out_metrics->loss = loss_sum / (float)total_steps;
    }
    sgd_free(opt);
    tensor_free(q->weight);
    tensor_free(q->bias);
    free(q);
    return TFRL_OK;
}

static void compute_returns(float *returns, const float *rewards, const int *dones, int n, float gamma) {
    float g = 0.0f;
    for (int i = n - 1; i >= 0; i--) {
        if (dones[i]) g = 0.0f;
        g = rewards[i] + gamma * g;
        returns[i] = g;
    }
}

int tfrl_train_ppo(tfrl_env *env, const tfrl_ppo_config *cfg, tfrl_train_metrics *out_metrics) {
    if (!env || !cfg) return TFRL_ERR_INVALID;
    if (cfg->obs_n <= 0 || cfg->action_n <= 0) return TFRL_ERR_INVALID;
    autograd_set_enabled(1);

    Linear *policy = linear_create(cfg->obs_n, cfg->action_n);
    Linear *value = linear_create(cfg->obs_n, 1);
    if (!policy || !value) return TFRL_ERR;
    init_linear(policy);
    init_linear(value);

    Tensor *params[4] = {policy->weight, policy->bias, value->weight, value->bias};
    Adam *opt = adam_create(params, 4, cfg->lr, 0.9f, 0.999f, 1e-8f, 0.0f);
    if (!opt) return TFRL_ERR;

    int batch = cfg->steps_per_batch > 0 ? cfg->steps_per_batch : 64;
    float *rewards = calloc((size_t)batch, sizeof(float));
    float *returns = calloc((size_t)batch, sizeof(float));
    int *actions = calloc((size_t)batch, sizeof(int));
    int *obs_idx = calloc((size_t)batch, sizeof(int));
    int *dones = calloc((size_t)batch, sizeof(int));
    float *old_logp = calloc((size_t)batch, sizeof(float));

    tfrl_value obs = {0};
    tfrl_step step = {0};
    if (cfg->load_path) {
        load_linear(policy, cfg->load_path, "pi");
        load_linear(value, cfg->load_path, "v");
    }

    tfrl_env_reset(env, &obs);
    float return_sum = 0.0f;
    float loss_sum = 0.0f;

    int updates = cfg->updates > 0 ? cfg->updates : 10;
    for (int update = 0; update < updates; update++) {
        for (int i = 0; i < batch; i++) {
            obs_idx[i] = (int)obs.as.i64;
            Tensor *x = one_hot(cfg->obs_n, obs_idx[i]);
            Tensor *logits = linear_forward(policy, x);
            Tensor *probs = tensor_softmax_autograd(logits);
            Tensor *logp = tensor_log(probs);
            int action = argmax_tensor(logits);
            actions[i] = action;
            old_logp[i] = tensor_get_f32_at(logp, (size_t)action);
            tfrl_value act = {.type = TFRL_VALUE_I64, .as.i64 = action};
            tfrl_env_step(env, &act, &step);
            rewards[i] = (float)step.reward;
            dones[i] = tfrl_step_done(&step);
            if (dones[i]) {
                tfrl_env_reset(env, &obs);
            } else {
                obs = step.observation;
            }
            tensor_free(logp);
            tensor_free(probs);
            tensor_free(logits);
            tensor_free(x);
        }

        compute_returns(returns, rewards, dones, batch, cfg->gamma);

        int epochs = cfg->epochs > 0 ? cfg->epochs : 2;
        for (int e = 0; e < epochs; e++) {
            for (int i = 0; i < batch; i++) {
                Tensor *x = one_hot(cfg->obs_n, obs_idx[i]);
                Tensor *logits = linear_forward(policy, x);
                Tensor *probs = tensor_softmax_autograd(logits);
                Tensor *logp = tensor_log(probs);
                Tensor *action_one = one_hot(cfg->action_n, actions[i]);
                Tensor *logp_sel = tensor_mul(logp, action_one);
                Tensor *logp_i = tensor_sum(logp_sel);
                Tensor *old_logp_t = tensor_new(1, (int[1]){1});
                tensor_set_f32_at(old_logp_t, 0, old_logp[i]);
                Tensor *ratio_t = tensor_exp(tensor_sub(logp_i, old_logp_t));
                Tensor *clipped = tensor_clamp(ratio_t, 1.0f - cfg->clip_eps, 1.0f + cfg->clip_eps);
                Tensor *adv_t = tensor_new(1, (int[1]){1});
                tensor_set_f32_at(adv_t, 0, returns[i]);
                Tensor *policy_loss = tensor_mul(clipped, adv_t);
                Tensor *neg_t = tensor_new(1, (int[1]){1});
                tensor_set_f32_at(neg_t, 0, -1.0f);
                Tensor *policy_loss_neg = tensor_mul(policy_loss, neg_t);

                Tensor *v = linear_forward(value, x);
                Tensor *target = tensor_new(1, (int[1]){1});
                tensor_set_f32_at(target, 0, returns[i]);
                Tensor *vdiff = tensor_sub(v, target);
                Tensor *value_loss = tensor_mul(vdiff, vdiff);

                Tensor *total = tensor_add(policy_loss_neg, value_loss);

                adam_zero_grad(opt);
                tensor_backward(total);
                adam_step(opt, 0.0f);
                loss_sum += tensor_item(total);

                tensor_free(total);
                tensor_free(value_loss);
                tensor_free(vdiff);
                tensor_free(target);
                tensor_free(v);
                tensor_free(policy_loss_neg);
                tensor_free(neg_t);
                tensor_free(policy_loss);
                tensor_free(adv_t);
                tensor_free(clipped);
                tensor_free(ratio_t);
                tensor_free(old_logp_t);
                tensor_free(logp_i);
                tensor_free(logp_sel);
                tensor_free(action_one);
                tensor_free(logp);
                tensor_free(probs);
                tensor_free(logits);
                tensor_free(x);
            }
        }
        for (int i = 0; i < batch; i++) {
            return_sum += rewards[i];
        }
    }

    if (out_metrics) {
        int total = updates * batch;
        out_metrics->return_mean = return_sum / (float)total;
        out_metrics->loss = loss_sum / (float)total;
    }

    if (cfg->eval_steps > 0) {
        tfrl_env_reset(env, &obs);
        for (int i = 0; i < cfg->eval_steps; i++) {
            Tensor *x = one_hot(cfg->obs_n, (int)obs.as.i64);
            Tensor *logits = linear_forward(policy, x);
            int action = argmax_tensor(logits);
            tfrl_value act = {.type = TFRL_VALUE_I64, .as.i64 = action};
            if (tfrl_env_step(env, &act, &step) != TFRL_OK) break;
            tensor_free(logits);
            tensor_free(x);
            if (tfrl_step_done(&step)) {
                tfrl_env_reset(env, &obs);
            } else {
                obs = step.observation;
            }
        }
    }

    if (cfg->save_path) {
        save_linear(policy, cfg->save_path, "pi");
        save_linear(value, cfg->save_path, "v");
    }
    free(rewards);
    free(returns);
    free(actions);
    free(obs_idx);
    free(dones);
    free(old_logp);
    adam_free(opt);
    tensor_free(policy->weight);
    tensor_free(policy->bias);
    tensor_free(value->weight);
    tensor_free(value->bias);
    free(policy);
    free(value);
    return TFRL_OK;
}
