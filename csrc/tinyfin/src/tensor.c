#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/********************
 * Constructors
 ********************/
Tensor *tensor_new(int ndim, const int *shape) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    t->shape = (int *)malloc(sizeof(int) * ndim);
    if (!t->shape) { free(t); return NULL; }
    memcpy(t->shape, shape, sizeof(int) * ndim);

    t->size = 1;
    for (int i = 0; i < ndim; i++) t->size *= (size_t)shape[i];

    t->strides = (int *)malloc(sizeof(int) * ndim);
    if (!t->strides) { free(t->shape); free(t); return NULL; }
    tensor_contiguous_strides(t->shape, ndim, t->strides);

    t->raw_data = calloc(t->size, sizeof(float));
    if (!t->raw_data) { free(t->strides); free(t->shape); free(t); return NULL; }
    t->data = (float *)t->raw_data;

    t->grad = NULL;
    t->requires_grad = 0;
    t->grad_fn = NULL;
    t->device = 0; /* DEVICE_CPU */
    t->dtype = DTYPE_FLOAT32;

    return t;
}

Tensor *tensor_new_like(Tensor *t, int requires_grad) {
    if (!t) return NULL;
    Tensor *out = tensor_new(t->ndim, t->shape);
    if (!out) return NULL;

    out->requires_grad = requires_grad;
    if (requires_grad) {
        out->grad = tensor_zeros(out->ndim, out->shape);
        if (out->grad) out->grad->requires_grad = 0;
    }
    out->device = t->device;
    out->dtype = t->dtype;
    tensor_set_dtype(out, t->dtype);
    return out;
}

Tensor *tensor_zeros(int ndim, const int *shape) {
    return tensor_new(ndim, shape);
}

void tensor_set_device(Tensor *t, int device) {
    if (!t) return;
    t->device = device;
    if (t->grad) t->grad->device = device;
}

int tensor_get_device(const Tensor *t) {
    if (!t) return -1;
    return t->device;
}

/********************
 * Stride utilities
 ********************/
void tensor_contiguous_strides(const int *shape, int ndim, int *out_strides) {
    int acc = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        out_strides[i] = acc;
        acc *= shape[i];
    }
}

/* ---------- dtype / raw access helpers ---------- */
int tensor_set_dtype(Tensor *t, int dtype) {
    if (!t) return 0;
    if ((int)t->dtype == dtype) return 1;
    /* Promote or reallocate buffer: only support promoting float32->float64 here. */
    if (dtype == DTYPE_FLOAT64 && t->dtype == DTYPE_FLOAT32) {
        double *buf = (double *)malloc(sizeof(double) * t->size);
        if (!buf) return 0;
        /* copy existing float32 -> double */
        for (size_t i = 0; i < t->size; i++) buf[i] = (double)((float *)t->raw_data)[i];
        free(t->raw_data);
        t->raw_data = buf;
        t->data = (float *)t->raw_data; /* keep float view but invalid for f64; callers should use f64 accessors */
        t->dtype = DTYPE_FLOAT64;
        return 1;
    }
    /* Other transitions not supported yet */
    return 0;
}

float tensor_get_f32_at(const Tensor *t, size_t off) {
    if (!t || !t->raw_data) return 0.0f;
    if (t->dtype == DTYPE_FLOAT32) return ((float *)t->raw_data)[off];
    /* if stored as float64, downcast */
    return (float)((double *)t->raw_data)[off];
}

double tensor_get_f64_at(const Tensor *t, size_t off) {
    if (!t || !t->raw_data) return 0.0;
    if (t->dtype == DTYPE_FLOAT64) return ((double *)t->raw_data)[off];
    /* if stored as float32, upcast */
    return (double)((float *)t->raw_data)[off];
}

void tensor_set_f32_at(Tensor *t, size_t off, float v) {
    if (!t || !t->raw_data) return;
    if (t->dtype == DTYPE_FLOAT32) ((float *)t->raw_data)[off] = v;
    else ((double *)t->raw_data)[off] = (double)v;
}

void tensor_set_f64_at(Tensor *t, size_t off, double v) {
    if (!t || !t->raw_data) return;
    if (t->dtype == DTYPE_FLOAT64) ((double *)t->raw_data)[off] = v;
    else ((float *)t->raw_data)[off] = (float)v;
}

/********************
 * Broadcasting helpers
 ********************/
int tensor_broadcast_shape(const Tensor *a, const Tensor *b, int **out_shape, int *out_ndim) {
    int ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    int *s = (int *)malloc(sizeof(int) * ndim);
    if (!s) return 0;

    for (int i = 0; i < ndim; i++) {
        int ai = i - (ndim - a->ndim);
        int bi = i - (ndim - b->ndim);
        int as = (ai < 0) ? 1 : a->shape[ai];
        int bs = (bi < 0) ? 1 : b->shape[bi];
        if (as == bs) s[i] = as;
        else if (as == 1) s[i] = bs;
        else if (bs == 1) s[i] = as;
        else { free(s); return 0; }
    }

    *out_shape = s;
    *out_ndim = ndim;
    return 1;
}

Tensor tensor_expand(const Tensor *t, const int *out_shape, int out_ndim) {
    Tensor r;
    r.ndim = out_ndim;
    r.shape = (int *)malloc(sizeof(int) * out_ndim);
    r.strides = (int *)malloc(sizeof(int) * out_ndim);
    r.data = t->data;
    /* propagate raw storage and dtype so accessors work */
    r.raw_data = t->raw_data;
    r.dtype = t->dtype;
    /* compute expanded size */
    r.size = 1;
    for (int i = 0; i < out_ndim; i++) r.size *= out_shape[i];
    r.grad = NULL;
    r.requires_grad = t->requires_grad;
    r.grad_fn = NULL;

    for (int i = 0; i < out_ndim; i++) {
        int src_i = i - (out_ndim - t->ndim);
        if (src_i < 0) { r.shape[i] = out_shape[i]; r.strides[i] = 0; continue; }
        r.shape[i] = out_shape[i];
        r.strides[i] = (t->shape[src_i] == out_shape[i]) ? t->strides[src_i] : 0;
    }
    return r;
}

/********************
 * Gradient Utilities
 * (attach handled in autograd.c)
 ********************/

void tensor_reduce_sum_broadcast(Tensor *grad, const Tensor *original, Tensor *out) {
    if (!grad || !original || !out) return;
    for (size_t i = 0; i < out->size; i++) out->data[i] = 0.0f;

    int *index = (int *)malloc(sizeof(int) * grad->ndim);
    if (!index) return;

    for (size_t i = 0; i < grad->size; i++) {
        size_t tmp = i;
        for (int d = grad->ndim - 1; d >= 0; d--) {
            index[d] = tmp % grad->shape[d];
            tmp /= grad->shape[d];
        }

        size_t orig_offset = 0;
        for (int d = 0; d < original->ndim; d++) {
            int idx = (original->shape[d] == 1) ? 0 : index[d + (grad->ndim - original->ndim)];
            orig_offset += idx * original->strides[d];
        }
        out->data[orig_offset] += grad->data[i];
    }

    free(index);
}

/********************
 * Debug / Utilities
 ********************/
void tensor_print(const Tensor *t) {
    if (!t) { printf("NULL tensor\n"); return; }
    printf("Tensor(shape=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim-1) printf(", ");
    }
    printf("], data=[");
    for (size_t i = 0; i < t->size; i++) {
        printf("%.4f", t->data[i]);
        if (i < t->size-1) printf(", ");
    }
    printf("], requires_grad=%d)\n", t->requires_grad);
}

int tensor_save(const Tensor *t, const char *path) {
    if (!t || !path) return 0;
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fwrite(&t->ndim, sizeof(int), 1, f);
    fwrite(t->shape, sizeof(int), t->ndim, f);
    fwrite(&t->size, sizeof(size_t), 1, f);
    fwrite(t->data, sizeof(float), t->size, f);
    fclose(f);
    return 1;
}

Tensor *tensor_load(const char *path) {
    if (!path) return NULL;
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    int ndim; if (fread(&ndim, sizeof(int), 1, f)!=1) { fclose(f); return NULL; }
    int *shape = malloc(sizeof(int)*ndim);
    if (fread(shape, sizeof(int), ndim, f)!= (size_t)ndim) { free(shape); fclose(f); return NULL; }
    size_t size; if (fread(&size, sizeof(size_t), 1, f)!=1) { free(shape); fclose(f); return NULL; }
    Tensor *t = tensor_new(ndim, shape);
    if (!t) { free(shape); fclose(f); return NULL; }
    if (fread(t->data, sizeof(float), t->size, f) != t->size) { tensor_free(t); free(shape); fclose(f); return NULL; }
    free(shape);
    fclose(f);
    return t;
}

void tensor_fill(Tensor *t, float value) {
    if (!t || !t->data) return;
    for (size_t i = 0; i < t->size; i++) t->data[i] = value;
}

/* Shape / view helpers: implemented in src/ops_reshape.c to avoid
   duplicate symbol definitions. Keep core tensor utilities here. */

/* ----------------- Basic indexing ----------------- */
float tensor_get(const Tensor *t, const int *idx) {
    if (!t || !idx) return 0.0f;
    int off = 0;
    for (int d = 0; d < t->ndim; d++) off += idx[d] * t->strides[d];
    return t->data[off];
}

int tensor_set(Tensor *t, const int *idx, float value) {
    if (!t || !idx) return 0;
    int off = 0;
    for (int d = 0; d < t->ndim; d++) off += idx[d] * t->strides[d];
    t->data[off] = value;
    return 1;
}

void tensor_set_requires_grad(Tensor *t, int requires_grad) {
    if (!t) return;
    t->requires_grad = requires_grad;
    if (!requires_grad && t->grad) {
        tensor_free(t->grad);
        t->grad = NULL;
    }
}

int tensor_get_requires_grad(const Tensor *t) {
    if (!t) return 0;
    return t->requires_grad;
}

/********************
 * Destructor
 * ********************/
void tensor_free(Tensor *t) {
    if (!t) return;
    if (t->raw_data) free(t->raw_data);
    if (t->grad) tensor_free(t->grad);
    if (t->shape) free(t->shape);
    if (t->strides) free(t->strides);
    free(t);
}
