#include <math.h>
#include <stdlib.h>

#include "replay_buffer.h"

struct tfrl_replay_buffer {
    int capacity;
    int size;
    int pos;
    float alpha;
    tfrl_transition *data;
    float *priorities;
};

static unsigned int lcg_next(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float lcg_uniform(unsigned int *state) {
    return (lcg_next(state) & 0x00FFFFFFu) / 16777215.0f;
}

static float replay_priority(const tfrl_replay_buffer *buf, int idx) {
    if (!buf || !buf->priorities) return 1.0f;
    float p = buf->priorities[idx];
    return p > 1e-6f ? p : 1e-6f;
}

tfrl_replay_buffer *tfrl_replay_create(int capacity, float alpha) {
    if (capacity <= 0) return NULL;
    tfrl_replay_buffer *buf = (tfrl_replay_buffer *)calloc(1, sizeof(tfrl_replay_buffer));
    if (!buf) return NULL;
    buf->capacity = capacity;
    buf->alpha = alpha;
    buf->data = (tfrl_transition *)calloc((size_t)capacity, sizeof(tfrl_transition));
    if (!buf->data) {
        free(buf);
        return NULL;
    }
    if (alpha > 0.0f) {
        buf->priorities = (float *)calloc((size_t)capacity, sizeof(float));
        if (!buf->priorities) {
            free(buf->data);
            free(buf);
            return NULL;
        }
        for (int i = 0; i < capacity; i++) {
            buf->priorities[i] = 1.0f;
        }
    }
    return buf;
}

void tfrl_replay_free(tfrl_replay_buffer *buf) {
    if (!buf) return;
    free(buf->data);
    free(buf->priorities);
    free(buf);
}

void tfrl_replay_push(tfrl_replay_buffer *buf, const tfrl_transition *transition, float priority) {
    if (!buf || !transition) return;
    buf->data[buf->pos] = *transition;
    if (buf->priorities) {
        buf->priorities[buf->pos] = priority > 0.0f ? priority : 1.0f;
    }
    buf->pos = (buf->pos + 1) % buf->capacity;
    if (buf->size < buf->capacity) {
        buf->size++;
    }
}

int tfrl_replay_sample(tfrl_replay_buffer *buf, int batch, int *out_idx, float *out_weights, float beta) {
    if (!buf || batch <= 0 || !out_idx) return 0;
    if (buf->size < batch) return 0;
    if (!buf->priorities || buf->alpha <= 0.0f) {
        for (int i = 0; i < batch; i++) {
            out_idx[i] = rand() % buf->size;
            if (out_weights) out_weights[i] = 1.0f;
        }
        return batch;
    }

    float sum = 0.0f;
    for (int i = 0; i < buf->size; i++) {
        float p = replay_priority(buf, i);
        sum += powf(p, buf->alpha);
    }
    if (sum <= 0.0f) return 0;

    float max_weight = 1e-6f;
    for (int i = 0; i < batch; i++) {
        float r = ((float)rand() / (float)RAND_MAX) * sum;
        float acc = 0.0f;
        int idx = 0;
        for (int j = 0; j < buf->size; j++) {
            float p = powf(replay_priority(buf, j), buf->alpha);
            acc += p;
            if (r <= acc) {
                idx = j;
                break;
            }
        }
        out_idx[i] = idx;
        if (out_weights) {
            float p = powf(replay_priority(buf, idx), buf->alpha) / sum;
            float w = powf((float)buf->size * p, -beta);
            out_weights[i] = w;
            if (w > max_weight) max_weight = w;
        }
    }

    if (out_weights && max_weight > 0.0f) {
        for (int i = 0; i < batch; i++) {
            out_weights[i] /= max_weight;
        }
    }
    return batch;
}

int tfrl_replay_sample_deterministic(tfrl_replay_buffer *buf, int batch, int *out_idx, float *out_weights, float beta, unsigned int seed) {
    if (!buf || batch <= 0 || !out_idx) return 0;
    if (buf->size < batch) return 0;
    unsigned int state = seed;
    if (!buf->priorities || buf->alpha <= 0.0f) {
        for (int i = 0; i < batch; i++) {
            out_idx[i] = (int)(lcg_next(&state) % (unsigned int)buf->size);
            if (out_weights) out_weights[i] = 1.0f;
        }
        return batch;
    }

    float sum = 0.0f;
    for (int i = 0; i < buf->size; i++) {
        float p = replay_priority(buf, i);
        sum += powf(p, buf->alpha);
    }
    if (sum <= 0.0f) return 0;

    float max_weight = 1e-6f;
    for (int i = 0; i < batch; i++) {
        float r = lcg_uniform(&state) * sum;
        float acc = 0.0f;
        int idx = 0;
        for (int j = 0; j < buf->size; j++) {
            float p = powf(replay_priority(buf, j), buf->alpha);
            acc += p;
            if (r <= acc) {
                idx = j;
                break;
            }
        }
        out_idx[i] = idx;
        if (out_weights) {
            float p = powf(replay_priority(buf, idx), buf->alpha) / sum;
            float w = powf((float)buf->size * p, -beta);
            out_weights[i] = w;
            if (w > max_weight) max_weight = w;
        }
    }

    if (out_weights && max_weight > 0.0f) {
        for (int i = 0; i < batch; i++) {
            out_weights[i] /= max_weight;
        }
    }
    return batch;
}

void tfrl_replay_update_priority(tfrl_replay_buffer *buf, int idx, float priority) {
    if (!buf || !buf->priorities) return;
    if (idx < 0 || idx >= buf->size) return;
    buf->priorities[idx] = priority > 0.0f ? priority : 1.0f;
}

const tfrl_transition *tfrl_replay_get(const tfrl_replay_buffer *buf, int idx) {
    if (!buf || idx < 0 || idx >= buf->size) return NULL;
    return &buf->data[idx];
}

int tfrl_replay_size(const tfrl_replay_buffer *buf) {
    return buf ? buf->size : 0;
}
