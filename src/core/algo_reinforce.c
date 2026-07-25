#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "algo_api.h"

#include "tinyfin/autograd.h"
#include "tinyfin/nn.h"
#include "tinyfin/ops_activation.h"
#include "tinyfin/ops_add.h"
#include "tinyfin/ops_log.h"
#include "tinyfin/ops_loss.h"
#include "tinyfin/ops_mul.h"
#include "tinyfin/ops_reduce.h"
#include "tinyfin/ops_softmax.h"
#include "tinyfin/optim.h"
#include "tinyfin/tensor.h"

typedef struct {
    int obs_type;
    int obs_dim;
    int obs_n;
    int action_n;
    int deterministic;
    float gamma;
    float lr;
    float epsilon;
    float entropy_coef;
    float clip_eps;
    float gae_lambda;
    float value_coef;
    float action_bias_right;
    int action_bias_index;
    int hidden_size;
    int rollout_batch_size;

    Linear *fc1;
    Linear *fc2;
    Linear *out;
    Linear *value;
    Adam *opt;

    int traj_count;
    struct reinforce_traj {
        int capacity;
        int length;
        float *obs_buf;
        int *action_buf;
        float *reward_buf;
        float *old_logp_buf;
    } *trajs;

    float last_return;
    int last_steps;
    float last_logprob;
} tfrl_reinforce_algo;

typedef struct {
    Tensor *input;
    Tensor *h1;
    Tensor *h1a;
    Tensor *h2;
    Tensor *h2a;
    Tensor *logits;
    Tensor *value;
} tfrl_policy_cache;

static float frand_unit(void) {
    return (float)rand() / (float)RAND_MAX;
}

static void init_linear_params(Linear *layer, int in_features, int out_features) {
    if (!layer || !layer->weight || !layer->bias) return;
    tensor_set_requires_grad(layer->weight, 1);
    tensor_set_requires_grad(layer->bias, 1);
    float scale = sqrtf(2.0f / (float)(in_features + out_features));
    for (size_t i = 0; i < layer->weight->size; i++) {
        float r = frand_unit() * 2.0f - 1.0f;
        layer->weight->data[i] = r * scale;
    }
    tensor_fill(layer->bias, 0.0f);
}

static int reinforce_input_dim(const tfrl_algo_config *cfg) {
    if (!cfg) return 0;
    if (cfg->obs_type == TFRL_SPACE_DISCRETE) {
        if (cfg->obs_n > 0 && cfg->obs_n <= 128) return cfg->obs_n;
        return 1;
    }
    if (cfg->obs_dims > 0) return cfg->obs_dims;
    return 0;
}

static void reinforce_obs_to_vec(const tfrl_reinforce_algo *algo, tfrl_obs obs, float *out) {
    if (!algo || !out || algo->obs_dim <= 0) return;
    if (algo->obs_type == TFRL_SPACE_DISCRETE) {
        if (algo->obs_dim == 1) {
            float denom = (algo->obs_n > 1) ? (float)(algo->obs_n - 1) : 1.0f;
            out[0] = denom > 0.0f ? (float)obs.index / denom : 0.0f;
            return;
        }
        for (int i = 0; i < algo->obs_dim; i++) out[i] = 0.0f;
        int idx = obs.index;
        if (idx < 0) idx = 0;
        if (idx >= algo->obs_dim) idx = algo->obs_dim - 1;
        out[idx] = 1.0f;
        return;
    }

    int len = obs.data_len;
    if (len < 0) len = 0;
    if (len > algo->obs_dim) len = algo->obs_dim;
    for (int i = 0; i < len; i++) out[i] = obs.data[i];
    for (int i = len; i < algo->obs_dim; i++) out[i] = 0.0f;
}

static Tensor *reinforce_make_input_tensor(const tfrl_reinforce_algo *algo, const float *vec) {
    if (!algo || algo->obs_dim <= 0 || !vec) return NULL;
    int shape[2] = {1, algo->obs_dim};
    Tensor *t = tensor_new(2, shape);
    if (!t) return NULL;
    for (int i = 0; i < algo->obs_dim; i++) {
        t->data[i] = vec[i];
    }
    return t;
}

static void reinforce_forward(tfrl_reinforce_algo *algo, tfrl_policy_cache *cache) {
    cache->h1 = linear_forward(algo->fc1, cache->input);
    cache->h1a = tensor_tanh(cache->h1);
    cache->h2 = linear_forward(algo->fc2, cache->h1a);
    cache->h2a = tensor_tanh(cache->h2);
    cache->logits = linear_forward(algo->out, cache->h2a);
    cache->value = linear_forward(algo->value, cache->h2a);
}

static void reinforce_free_cache(tfrl_policy_cache *cache) {
    if (!cache) return;
    if (cache->value) tensor_free(cache->value);
    if (cache->logits) tensor_free(cache->logits);
    if (cache->h2a) tensor_free(cache->h2a);
    if (cache->h2) tensor_free(cache->h2);
    if (cache->h1a) tensor_free(cache->h1a);
    if (cache->h1) tensor_free(cache->h1);
    if (cache->input) tensor_free(cache->input);
    memset(cache, 0, sizeof(*cache));
}

static double reinforce_sum_abs_grad(const Tensor *t) {
    if (!t || !t->grad || !t->grad->data) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < t->grad->size; i++) {
        sum += fabs((double)t->grad->data[i]);
    }
    return sum;
}

static int reinforce_sample_action(tfrl_reinforce_algo *algo, Tensor *probs) {
    if (!algo || !probs || probs->size == 0) return 0;
    if (algo->deterministic) {
        int best = 0;
        float best_v = probs->data[0];
        for (int i = 1; i < algo->action_n; i++) {
            float v = probs->data[i];
            if (v > best_v) {
                best_v = v;
                best = i;
            }
        }
        return best;
    }

    if (algo->epsilon > 0.0f && frand_unit() < algo->epsilon) {
        return rand() % algo->action_n;
    }

    float r = frand_unit();
    float acc = 0.0f;
    for (int i = 0; i < algo->action_n; i++) {
        acc += probs->data[i];
        if (r <= acc) return i;
    }
    return algo->action_n - 1;
}

static struct reinforce_traj *reinforce_get_traj(tfrl_reinforce_algo *algo, int env_id) {
    if (!algo) return NULL;
    if (env_id < 0) env_id = 0;
    if (env_id >= algo->traj_count) {
        int new_count = algo->traj_count > 0 ? algo->traj_count : 1;
        while (new_count <= env_id) new_count *= 2;
        struct reinforce_traj *new_trajs =
            (struct reinforce_traj *)realloc(algo->trajs, (size_t)new_count * sizeof(*new_trajs));
        if (!new_trajs) return NULL;
        for (int i = algo->traj_count; i < new_count; i++) {
            new_trajs[i].capacity = 0;
            new_trajs[i].length = 0;
            new_trajs[i].obs_buf = NULL;
            new_trajs[i].action_buf = NULL;
            new_trajs[i].reward_buf = NULL;
            new_trajs[i].old_logp_buf = NULL;
        }
        algo->trajs = new_trajs;
        algo->traj_count = new_count;
    }
    return &algo->trajs[env_id];
}

static void reinforce_traj_push(tfrl_reinforce_algo *algo, struct reinforce_traj *traj,
                                const tfrl_transition *tr) {
    if (!algo || !traj || !tr || algo->obs_dim <= 0) return;
    if (traj->length >= traj->capacity) {
        int new_cap = traj->capacity > 0 ? traj->capacity * 2 : 512;
        size_t obs_bytes = (size_t)new_cap * (size_t)algo->obs_dim * sizeof(float);
        float *new_obs = (float *)realloc(traj->obs_buf, obs_bytes);
        int *new_actions = (int *)realloc(traj->action_buf, (size_t)new_cap * sizeof(int));
        float *new_rewards = (float *)realloc(traj->reward_buf, (size_t)new_cap * sizeof(float));
        float *new_old_logp = (float *)realloc(traj->old_logp_buf, (size_t)new_cap * sizeof(float));
        if (!new_obs || !new_actions || !new_rewards || !new_old_logp) return;
        traj->obs_buf = new_obs;
        traj->action_buf = new_actions;
        traj->reward_buf = new_rewards;
        traj->old_logp_buf = new_old_logp;
        traj->capacity = new_cap;
    }

    float *dst = traj->obs_buf + (size_t)traj->length * (size_t)algo->obs_dim;
    reinforce_obs_to_vec(algo, tr->obs, dst);
    int act = tr->action.index;
    if (act < 0) act = 0;
    if (act >= algo->action_n) act = algo->action_n - 1;
    traj->action_buf[traj->length] = act;
    traj->reward_buf[traj->length] = (float)tr->reward;
    traj->old_logp_buf[traj->length] = algo->last_logprob;
    traj->length += 1;
}

static void reinforce_train_episode(tfrl_reinforce_algo *algo, struct reinforce_traj *traj) {
    if (!algo || !traj || traj->length <= 0) return;
    int T = traj->length;

    float *returns = (float *)malloc((size_t)T * sizeof(float));
    float *advantages = (float *)malloc((size_t)T * sizeof(float));
    if (!returns || !advantages) {
        free(returns);
        free(advantages);
        traj->length = 0;
        return;
    }

    float G = 0.0f;
    float gae = 0.0f;
    for (int t = T - 1; t >= 0; t--) {
        G = traj->reward_buf[t] + algo->gamma * G;
        returns[t] = G;
        /* With the current one-step value head, use zero bootstrap at the
         * trajectory boundary and GAE(lambda) over observed rewards. */
        gae = traj->reward_buf[t] + algo->gamma * algo->gae_lambda * gae;
        advantages[t] = gae;
    }

    float mean = 0.0f;
    for (int t = 0; t < T; t++) mean += advantages[t];
    mean /= (float)T;
    float var = 0.0f;
    for (int t = 0; t < T; t++) {
        float d = advantages[t] - mean;
        var += d * d;
    }
    var /= (float)T;
    float std = sqrtf(var);
    if (std < 1e-6f) std = 1.0f;

    adam_zero_grad(algo->opt);

    for (int t = 0; t < T; t++) {
        float ret_norm = (returns[t] - mean) / std;
        float adv_norm = (advantages[t] - mean) / std;
        float *vec = traj->obs_buf + (size_t)t * (size_t)algo->obs_dim;

        tfrl_policy_cache cache = {0};
        cache.input = reinforce_make_input_tensor(algo, vec);
        if (!cache.input) continue;
        reinforce_forward(algo, &cache);

        if (algo->action_bias_right != 0.0f && algo->action_bias_index >= 0 &&
            algo->action_bias_index < algo->action_n) {
            cache.logits->data[algo->action_bias_index] += algo->action_bias_right;
        }

        Tensor *probs = tensor_softmax(cache.logits);
        int shape[2] = {1, algo->action_n};
        Tensor *target = tensor_zeros(2, shape);
        if (target) {
            int act = traj->action_buf[t];
            target->data[act] = 1.0f;
        }

        Tensor *ce = tensor_cross_entropy(probs, target);
        float value_pred = cache.value && cache.value->data ? cache.value->data[0] : 0.0f;
        float adv = adv_norm - value_pred;
        if (probs && algo->clip_eps > 0.0f) {
            int act = traj->action_buf[t];
            float current = logf(probs->data[act] > 1e-8f ? probs->data[act] : 1e-8f);
            float ratio = expf(current - traj->old_logp_buf[t]);
            float lo = 1.0f - algo->clip_eps;
            float hi = 1.0f + algo->clip_eps;
            if ((adv > 0.0f && ratio > hi) || (adv < 0.0f && ratio < lo)) {
                float clipped = ratio < lo ? lo : (ratio > hi ? hi : ratio);
                adv *= clipped / (fabsf(ratio) > 1e-8f ? ratio : 1.0f);
            }
        }
        int shape1[1] = {1};
        Tensor *ret = tensor_new(1, shape1);
        if (ret) ret->data[0] = adv;
        Tensor *policy_loss = tensor_mul(ce, ret);

        Tensor *value_target = tensor_new(2, (int[2]){1, 1});
        if (value_target) value_target->data[0] = ret_norm;
        Tensor *value_loss = NULL;
        if (cache.value && value_target) {
            value_loss = tensor_mse_loss(cache.value, value_target);
        }

        Tensor *entropy = NULL;
        Tensor *entropy_scaled = NULL;
        Tensor *entropy_log_probs = NULL;
        Tensor *entropy_p_log_p = NULL;
        Tensor *entropy_sum = NULL;
        Tensor *entropy_coef = NULL;
        if (algo->entropy_coef > 0.0f && probs) {
            entropy_log_probs = tensor_log(probs);
            entropy_p_log_p = tensor_mul(probs, entropy_log_probs);
            entropy_sum = tensor_sum(entropy_p_log_p);
            entropy_coef = tensor_new(1, shape1);
            if (entropy_coef) entropy_coef->data[0] = -algo->entropy_coef;
            if (entropy_sum && entropy_coef) entropy_scaled = tensor_mul(entropy_sum, entropy_coef);
            entropy = entropy_scaled;
        }

        Tensor *total = policy_loss;
        if (value_loss) {
            Tensor *coef = tensor_new(1, shape1);
            if (coef) coef->data[0] = algo->value_coef;
            Tensor *scaled = coef ? tensor_mul(value_loss, coef) : value_loss;
            if (coef) tensor_free(coef);
            if (scaled) {
                Tensor *next = tensor_add(total, scaled);
                if (total && total != policy_loss) tensor_free(total);
                if (scaled && scaled != value_loss) tensor_free(scaled);
                total = next;
            }
        }
        if (entropy_scaled) {
            Tensor *next = tensor_add(total, entropy_scaled);
            if (total && total != policy_loss) tensor_free(total);
            total = next;
        }

        if (total) tensor_backward_with_retain(total, 0);
        if (getenv("TFRL_DEBUG_GRAD")) {
            double gsum = 0.0;
            gsum += reinforce_sum_abs_grad(algo->fc1->weight);
            gsum += reinforce_sum_abs_grad(algo->fc1->bias);
            gsum += reinforce_sum_abs_grad(algo->fc2->weight);
            gsum += reinforce_sum_abs_grad(algo->fc2->bias);
            gsum += reinforce_sum_abs_grad(algo->out->weight);
            gsum += reinforce_sum_abs_grad(algo->out->bias);
            gsum += reinforce_sum_abs_grad(algo->value->weight);
            gsum += reinforce_sum_abs_grad(algo->value->bias);
            if (gsum == 0.0) {
                fprintf(stderr, "[diag] grad sum is zero at t=%d\n", t);
            } else if (t == 0) {
                fprintf(stderr, "[diag] grad sum at t=%d: %.6f\n", t, gsum);
            }
        }

        if (total) tensor_free(total);
        if (entropy && entropy != entropy_scaled) tensor_free(entropy);
        if (entropy_scaled) tensor_free(entropy_scaled);
        if (entropy_sum) tensor_free(entropy_sum);
        if (entropy_p_log_p) tensor_free(entropy_p_log_p);
        if (entropy_log_probs) tensor_free(entropy_log_probs);
        if (entropy_coef) tensor_free(entropy_coef);
        if (value_loss) tensor_free(value_loss);
        if (value_target) tensor_free(value_target);
        if (ret) tensor_free(ret);
        if (ce) tensor_free(ce);
        if (target) tensor_free(target);
        if (probs) tensor_free(probs);
        reinforce_free_cache(&cache);
    }

    adam_step(algo->opt, 1.0f);

    algo->last_return = returns[0];
    algo->last_steps = T;
    free(returns);
    free(advantages);
    traj->length = 0;
}

static tfrl_action reinforce_act(void *ctx, tfrl_obs obs) {
    tfrl_reinforce_algo *algo = (tfrl_reinforce_algo *)ctx;
    tfrl_action action = {0};
    if (!algo || algo->obs_dim <= 0) return action;

    float *vec = (float *)calloc((size_t)algo->obs_dim, sizeof(float));
    if (!vec) return action;
    reinforce_obs_to_vec(algo, obs, vec);

    int prev_autograd = autograd_get_enabled();
    autograd_set_enabled(0);

    tfrl_policy_cache cache = {0};
    cache.input = reinforce_make_input_tensor(algo, vec);
    if (!cache.input) {
        autograd_set_enabled(prev_autograd);
        free(vec);
        return action;
    }
    reinforce_forward(algo, &cache);

    if (algo->action_bias_right != 0.0f && algo->action_bias_index >= 0 &&
        algo->action_bias_index < algo->action_n) {
        cache.logits->data[algo->action_bias_index] += algo->action_bias_right;
    }

    Tensor *probs = tensor_softmax(cache.logits);
    int act = reinforce_sample_action(algo, probs);
    action.index = act;
    algo->last_logprob = probs && probs->data[act] > 1e-8f
        ? logf(probs->data[act]) : logf(1e-8f);

    const char *dbg = getenv("TFRL_DEBUG_ACTIONS");
    if (dbg && *dbg) {
        static int printed = 0;
        if (printed < 20) {
            fprintf(stderr, "[reinforce] action=%d probs=", act);
            if (probs) {
                for (int i = 0; i < algo->action_n; i++) {
                    fprintf(stderr, "%0.3f", probs->data[i]);
                    if (i + 1 < algo->action_n) fputc(',', stderr);
                }
            }
            fprintf(stderr, "\n");
            printed++;
        }
    }

    if (probs) tensor_free(probs);
    reinforce_free_cache(&cache);
    autograd_set_enabled(prev_autograd);
    free(vec);
    return action;
}

static void reinforce_act_batch(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out_actions) {
    if (!ctx || !out_actions || count <= 0) return;
    for (int i = 0; i < count; i++) {
        out_actions[i] = reinforce_act(ctx, obs ? obs[i] : (tfrl_obs){0});
    }
}

static void reinforce_update(void *ctx, const tfrl_transition *transition) {
    tfrl_reinforce_algo *algo = (tfrl_reinforce_algo *)ctx;
    if (!algo || !transition) return;
    autograd_set_enabled(1);
    struct reinforce_traj *traj = reinforce_get_traj(algo, transition->policy_version);
    if (!traj) return;
    reinforce_traj_push(algo, traj, transition);
    if (transition->done || traj->length >= algo->rollout_batch_size) {
        reinforce_train_episode(algo, traj);
    }
}

static void reinforce_destroy(void *ctx) {
    tfrl_reinforce_algo *algo = (tfrl_reinforce_algo *)ctx;
    if (!algo) return;
    if (algo->opt) adam_free(algo->opt);
    if (algo->fc1) {
        tensor_free(algo->fc1->weight);
        tensor_free(algo->fc1->bias);
        free(algo->fc1);
    }
    if (algo->fc2) {
        tensor_free(algo->fc2->weight);
        tensor_free(algo->fc2->bias);
        free(algo->fc2);
    }
    if (algo->out) {
        tensor_free(algo->out->weight);
        tensor_free(algo->out->bias);
        free(algo->out);
    }
    if (algo->value) {
        tensor_free(algo->value->weight);
        tensor_free(algo->value->bias);
        free(algo->value);
    }
    if (algo->trajs) {
        for (int i = 0; i < algo->traj_count; i++) {
            free(algo->trajs[i].obs_buf);
            free(algo->trajs[i].action_buf);
            free(algo->trajs[i].reward_buf);
            free(algo->trajs[i].old_logp_buf);
        }
        free(algo->trajs);
    }
    free(algo);
}

static int reinforce_diagnostics(void *ctx, char *buffer, size_t buffer_len) {
    tfrl_reinforce_algo *algo = (tfrl_reinforce_algo *)ctx;
    if (!buffer || buffer_len == 0 || !algo) return 0;
    snprintf(buffer, buffer_len,
             "ppo_baseline ep_return=%.3f ep_len=%d lr=%.5f gamma=%.3f clip=%.3f gae=%.3f entropy=%.3f",
             algo->last_return, algo->last_steps, algo->lr, algo->gamma,
             algo->clip_eps, algo->gae_lambda, algo->entropy_coef);
    return 1;
}

static const tfrl_algo_vtable REINFORCE_VTABLE = {
    .act = reinforce_act,
    .act_env = NULL,
    .act_batch = reinforce_act_batch,
    .update = reinforce_update,
    .save = NULL,
    .destroy = reinforce_destroy,
    .diagnostics = reinforce_diagnostics,
};

tfrl_algo tfrl_algo_reinforce_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->action_n <= 0) return out;

    int obs_dim = reinforce_input_dim(cfg);
    if (obs_dim <= 0) return out;

    const char *dbg = getenv("TFRL_DEBUG_ALGO");
    if (dbg && *dbg) {
        fprintf(stderr, "[reinforce] obs_dim=%d action_n=%d obs_type=%d obs_n=%d\n",
                obs_dim, cfg->action_n, cfg->obs_type, cfg->obs_n);
    }

    tfrl_reinforce_algo *algo = (tfrl_reinforce_algo *)calloc(1, sizeof(tfrl_reinforce_algo));
    if (!algo) return out;

    algo->obs_type = cfg->obs_type;
    algo->obs_dim = obs_dim;
    algo->obs_n = cfg->obs_n;
    algo->action_n = cfg->action_n;
    algo->deterministic = cfg->deterministic;
    algo->gamma = (cfg->gamma > 0.0f) ? cfg->gamma : 0.99f;
    algo->lr = (cfg->lr > 0.0f) ? cfg->lr : 0.0005f;
    algo->epsilon = (cfg->epsilon >= 0.0f) ? cfg->epsilon : 0.1f;
    algo->entropy_coef = (cfg->entropy_coef > 0.0f) ? cfg->entropy_coef : 0.01f;
    algo->clip_eps = cfg->clip_eps > 0.0f ? cfg->clip_eps : 0.2f;
    algo->gae_lambda = cfg->gae_lambda > 0.0f ? cfg->gae_lambda : 0.95f;
    algo->value_coef = 0.5f;
    algo->action_bias_right = cfg->action_bias_right;
    algo->action_bias_index = (cfg->action_n > 2) ? 2 : -1;
    algo->hidden_size = 64;
    algo->rollout_batch_size = cfg->steps_per_batch > 0 ? cfg->steps_per_batch : 64;
    if (algo->rollout_batch_size > 2048) algo->rollout_batch_size = 2048;

    algo->fc1 = linear_create(algo->obs_dim, algo->hidden_size);
    algo->fc2 = linear_create(algo->hidden_size, algo->hidden_size);
    algo->out = linear_create(algo->hidden_size, algo->action_n);
    algo->value = linear_create(algo->hidden_size, 1);
    if (!algo->fc1 || !algo->fc2 || !algo->out || !algo->value) {
        reinforce_destroy(algo);
        return out;
    }

    init_linear_params(algo->fc1, algo->obs_dim, algo->hidden_size);
    init_linear_params(algo->fc2, algo->hidden_size, algo->hidden_size);
    init_linear_params(algo->out, algo->hidden_size, algo->action_n);
    init_linear_params(algo->value, algo->hidden_size, 1);

    Tensor *params[8] = {
        algo->fc1->weight, algo->fc1->bias,
        algo->fc2->weight, algo->fc2->bias,
        algo->out->weight, algo->out->bias,
        algo->value->weight, algo->value->bias
    };
    algo->opt = adam_create(params, 8, algo->lr, 0.9f, 0.999f, 1e-8f, 0.0f);
    if (!algo->opt) {
        reinforce_destroy(algo);
        return out;
    }

    algo->traj_count = 0;
    algo->trajs = NULL;

    out.ctx = algo;
    out.vtable = &REINFORCE_VTABLE;
    return out;
}
