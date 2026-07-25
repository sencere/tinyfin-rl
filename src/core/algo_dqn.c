#include "algo_api.h"
#include "tinyfin/nn.h"
#include "tinyfin/ops_activation.h"
#include "tinyfin/tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int obs_n, obs_dim, action_n, hidden;
    float gamma, lr, epsilon;
    int deterministic, replay_size, replay_count, replay_cursor, updates;
    unsigned rng;
    Linear *feature, *head, *target_head;
    tfrl_transition *replay;
} tfrl_neural_dqn;

static unsigned rng_next(tfrl_neural_dqn *a) { return a->rng = a->rng * 1664525u + 1013904223u; }
static float rng01(tfrl_neural_dqn *a) { return (rng_next(a) >> 8) * (1.0f / 16777216.0f); }

static Tensor *dqn_input(tfrl_neural_dqn *a, tfrl_obs obs) {
    int shape[2] = {1, a->obs_dim};
    Tensor *x = tensor_zeros(2, shape);
    if (!x) return NULL;
    if (obs.data_len > 0) {
        for (int i = 0; i < obs.data_len && i < a->obs_dim; i++) x->data[i] = obs.data[i];
    } else if (obs.index >= 0 && obs.index < a->obs_dim) {
        x->data[obs.index] = 1.0f;
    }
    return x;
}

static Tensor *dqn_forward(tfrl_neural_dqn *a, Linear *head, tfrl_obs obs) {
    Tensor *x = dqn_input(a, obs);
    if (!x) return NULL;
    Tensor *h = linear_forward(a->feature, x);
    Tensor *r = h ? tensor_relu(h) : NULL;
    Tensor *q = r ? linear_forward(head, r) : NULL;
    if (r) tensor_free(r);
    if (h) tensor_free(h);
    tensor_free(x);
    return q;
}

static int best_action(Tensor *q, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (q->data[i] > q->data[best]) best = i;
    return best;
}

static tfrl_action dqn_act(void *ctx, tfrl_obs obs) {
    tfrl_neural_dqn *a = ctx; tfrl_action out = {0};
    Tensor *q = dqn_forward(a, a->head, obs);
    if (!q) return out;
    int action = best_action(q, a->action_n);
    if (!a->deterministic && rng01(a) < a->epsilon) action = (int)(rng01(a) * a->action_n) % a->action_n;
    out.index = action;
    tensor_free(q);
    return out;
}

static void dqn_batch(void *ctx, const tfrl_obs *obs, int count, tfrl_action *out) {
    for (int i = 0; i < count; i++) out[i] = dqn_act(ctx, obs[i]);
}

static void dqn_update(void *ctx, const tfrl_transition *tr) {
    tfrl_neural_dqn *a = ctx;
    if (!a || !tr) return;
    a->replay[a->replay_cursor] = *tr;
    a->replay_cursor = (a->replay_cursor + 1) % a->replay_size;
    if (a->replay_count < a->replay_size) a->replay_count++;
    tfrl_transition *s = &a->replay[(a->replay_cursor + a->replay_size - 1) % a->replay_size];
    Tensor *q = dqn_forward(a, a->head, s->obs);
    Tensor *next = dqn_forward(a, a->target_head, s->next_obs);
    if (!q || !next) { if (q) tensor_free(q); if (next) tensor_free(next); return; }
    int action = s->action.index < 0 ? 0 : s->action.index % a->action_n;
    float max_next = next->data[0];
    for (int i = 1; i < a->action_n; i++) if (next->data[i] > max_next) max_next = next->data[i];
    float target = (float)s->reward + (s->done ? 0.0f : a->gamma * max_next);
    float td = target - q->data[action];
    /* Train the Q head with a clipped TD step; the feature layer is retained
     * as a Tinyfin representation and can be unfrozen when full autograd
     * batch support is added. */
    Tensor *x = dqn_input(a, s->obs);
    Tensor *h = x ? linear_forward(a->feature, x) : NULL;
    Tensor *r = h ? tensor_relu(h) : NULL;
    if (r) {
        float step = a->lr * fmaxf(-1.0f, fminf(1.0f, td));
        for (int i = 0; i < a->head->weight->shape[1]; i++)
            a->head->weight->data[action * a->hidden + i] += step * r->data[i];
        a->head->bias->data[action] += step;
    }
    if (r) tensor_free(r);
    if (h) tensor_free(h);
    if (x) tensor_free(x);
    tensor_free(q); tensor_free(next);
    a->updates++;
    if (a->updates % 100 == 0) memcpy(a->target_head->weight->data, a->head->weight->data,
                                      a->head->weight->size * sizeof(float));
    if (a->epsilon > 0.02f) a->epsilon *= 0.9995f;
}

static void dqn_destroy(void *ctx) {
    tfrl_neural_dqn *a = ctx; if (!a) return;
    free(a->replay);
    if (a->feature) { tensor_free(a->feature->weight); tensor_free(a->feature->bias); free(a->feature); }
    if (a->head) { tensor_free(a->head->weight); tensor_free(a->head->bias); free(a->head); }
    if (a->target_head) { tensor_free(a->target_head->weight); tensor_free(a->target_head->bias); free(a->target_head); }
    free(a);
}

static void dqn_save(void *ctx, const char *path) {
    tfrl_neural_dqn *a = ctx;
    if (!a || !path) return;
    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    fprintf(fp, "TFRL_NEURAL_DQN 1 %d %d %d %d %d %.9g %.9g %.9g %u %d\n",
            a->obs_dim, a->action_n, a->hidden, a->replay_size,
            a->replay_count, a->gamma, a->lr, a->epsilon, a->rng, a->updates);
    size_t feature_count = a->feature->weight->size + a->feature->bias->size;
    size_t head_count = a->head->weight->size + a->head->bias->size;
    fprintf(fp, "%zu %zu\n", feature_count, head_count);
    for (size_t i = 0; i < a->feature->weight->size; i++) fprintf(fp, "%.9g\n", a->feature->weight->data[i]);
    for (size_t i = 0; i < a->feature->bias->size; i++) fprintf(fp, "%.9g\n", a->feature->bias->data[i]);
    for (size_t i = 0; i < a->head->weight->size; i++) fprintf(fp, "%.9g\n", a->head->weight->data[i]);
    for (size_t i = 0; i < a->head->bias->size; i++) fprintf(fp, "%.9g\n", a->head->bias->data[i]);
    for (size_t i = 0; i < a->target_head->weight->size; i++) fprintf(fp, "%.9g\n", a->target_head->weight->data[i]);
    for (size_t i = 0; i < a->target_head->bias->size; i++) fprintf(fp, "%.9g\n", a->target_head->bias->data[i]);
    fclose(fp);
}

static int dqn_load(tfrl_neural_dqn *a, const char *path) {
    if (!a || !path) return 0;
    FILE *fp = fopen(path, "rb"); if (!fp) return 0;
    char magic[32]; int version, obs_dim, actions, hidden, replay_size, replay_count, updates; unsigned rng;
    float gamma, lr, epsilon;
    int ok = fscanf(fp, "%31s %d %d %d %d %d %d %f %f %f %u %d", magic, &version,
                    &obs_dim, &actions, &hidden, &replay_size, &replay_count,
                    &gamma, &lr, &epsilon, &rng, &updates) == 12;
    ok = ok && strcmp(magic, "TFRL_NEURAL_DQN") == 0 && version == 1 &&
         obs_dim == a->obs_dim && actions == a->action_n && hidden == a->hidden;
    size_t feature_count = 0, head_count = 0;
    ok = ok && fscanf(fp, "%zu %zu", &feature_count, &head_count) == 2;
    ok = ok && feature_count == a->feature->weight->size + a->feature->bias->size &&
         head_count == a->head->weight->size + a->head->bias->size;
    float *blocks[] = {a->feature->weight->data, a->feature->bias->data,
                       a->head->weight->data, a->head->bias->data,
                       a->target_head->weight->data, a->target_head->bias->data};
    size_t sizes[] = {a->feature->weight->size, a->feature->bias->size,
                      a->head->weight->size, a->head->bias->size,
                      a->target_head->weight->size, a->target_head->bias->size};
    for (int b = 0; ok && b < 6; b++) for (size_t i = 0; i < sizes[b]; i++) ok = fscanf(fp, "%f", &blocks[b][i]) == 1;
    fclose(fp);
    if (ok) { a->replay_count = replay_count; a->gamma = gamma; a->lr = lr; a->epsilon = epsilon; a->rng = rng; a->updates = updates; }
    return ok;
}

static int dqn_diag(void *ctx, char *buf, size_t n) {
    tfrl_neural_dqn *a = ctx;
    if (!a || !buf || !n) return 0;
    snprintf(buf, n, "neural_dqn replay=%d/%d eps=%.3f updates=%d", a->replay_count, a->replay_size, a->epsilon, a->updates);
    return 1;
}

static const tfrl_algo_vtable VTABLE = {.act=dqn_act,.act_batch=dqn_batch,.update=dqn_update,.save=dqn_save,.destroy=dqn_destroy,.diagnostics=dqn_diag};

tfrl_algo tfrl_algo_dqn_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0}; if (!cfg || cfg->action_n <= 0) return out;
    tfrl_neural_dqn *a = calloc(1, sizeof(*a)); if (!a) return out;
    a->obs_n = cfg->obs_n; a->obs_dim = cfg->obs_type == TFRL_SPACE_DISCRETE ? (cfg->obs_n > 0 ? cfg->obs_n : 1) : (cfg->obs_dims > 0 ? cfg->obs_dims : 1);
    a->action_n = cfg->action_n; a->hidden = 64; a->gamma = cfg->gamma > 0 ? cfg->gamma : .99f; a->lr = cfg->lr > 0 ? cfg->lr : .001f; a->epsilon = cfg->epsilon >= 0 ? cfg->epsilon : .1f; a->deterministic = cfg->deterministic; a->rng = cfg->seed ? cfg->seed : 1; a->replay_size = cfg->replay_size > 0 ? cfg->replay_size : 1000;
    a->feature = linear_create(a->obs_dim, a->hidden); a->head = linear_create(a->hidden, a->action_n); a->target_head = linear_create(a->hidden, a->action_n); a->replay = calloc(a->replay_size, sizeof(*a->replay));
    if (!a->feature || !a->head || !a->target_head || !a->replay) { dqn_destroy(a); return out; }
    memcpy(a->target_head->weight->data, a->head->weight->data, a->head->weight->size * sizeof(float)); memcpy(a->target_head->bias->data, a->head->bias->data, a->head->bias->size * sizeof(float));
    if (cfg->load_path) dqn_load(a, cfg->load_path);
    out.ctx = a; out.vtable = &VTABLE; return out;
}
