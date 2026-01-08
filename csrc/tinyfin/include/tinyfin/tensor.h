#ifndef CTORCH_TENSOR_H
#define CTORCH_TENSOR_H

#include <stddef.h>

/* Forward declaration (autograd node is defined elsewhere) */
typedef struct AutogradNode AutogradNode;

/* Tensor structure */
typedef struct Tensor {
	int ndim;          /* number of dimensions */
	int *shape;        /* length = ndim */
	int *strides;      /* length = ndim (in element counts) */
	size_t size;       /* total number of elements */
	float *data;       /* contiguous storage (float32 view for now) */
	void *raw_data;    /* opaque storage pointer (points to allocated buffer) */
	struct Tensor *grad;   /* gradient tensor */
	int requires_grad;     /* whether tensor participates in autograd */
	AutogradNode *grad_fn; /* grad function node */

	/* Device abstraction: currently only CPU (0). Kept as int for future
	 * extensions (GPU, other backends). */
	int device;

	/* DType: default is float32. We keep `data` as float* for backward
	 * compatibility; `raw_data` holds the allocated buffer (float/double).
	 */
	enum { DTYPE_FLOAT32 = 0, DTYPE_FLOAT64 = 1 } dtype;
} Tensor;

/* Device constants */
#define DEVICE_CPU 0
#define DEVICE_GPU 1 /* placeholder for future backends */

/* DType constants (match enum in struct) */
#define DTYPE_FLOAT32 0
#define DTYPE_FLOAT64 1

/***************
 * Constructors *
 ***************/
Tensor *tensor_new(int ndim, const int *shape);
Tensor *tensor_new_like(Tensor *t, int requires_grad);
Tensor *tensor_zeros(int ndim, const int *shape);
/* Serialization */
int tensor_save(const Tensor *t, const char *path);
Tensor *tensor_load(const char *path);

/* Shape / view helpers */
Tensor *tensor_transpose(Tensor *t);
Tensor *tensor_reshape(Tensor *a, int new_ndim, const int *new_shape);
Tensor *tensor_flatten(Tensor *t);

/* Basic indexing */
float tensor_get(const Tensor *t, const int *idx);
int tensor_set(Tensor *t, const int *idx, float value);
void tensor_set_requires_grad(Tensor *t, int requires_grad);
int tensor_get_requires_grad(const Tensor *t);

/***************
 * Broadcasting *
 ***************/
int tensor_broadcast_shape(const Tensor *a, const Tensor *b, int **out_shape, int *out_ndim);
Tensor tensor_expand(const Tensor *t, const int *out_shape, int out_ndim);

/***************
 * Strides *
 ***************/
void tensor_contiguous_strides(const int *shape, int ndim, int *out_strides);

/**********************
 * Gradient Utilities *
 **********************/
void Tensor_attach_gradients(Tensor *t, AutogradNode *node);
void tensor_reduce_sum_broadcast(Tensor *grad, const Tensor *original, Tensor *out);

/* device helpers */
void tensor_set_device(Tensor *t, int device);
int tensor_get_device(const Tensor *t);

/* DType helpers and element access (flat offset in elements) */
int tensor_set_dtype(Tensor *t, int dtype);
float tensor_get_f32_at(const Tensor *t, size_t off);
double tensor_get_f64_at(const Tensor *t, size_t off);
void tensor_set_f32_at(Tensor *t, size_t off, float v);
void tensor_set_f64_at(Tensor *t, size_t off, double v);

/**********************
 * Debug / Utilities *
 **********************/
void tensor_print(const Tensor *t);
void tensor_fill(Tensor *t, float value);
void tensor_free(Tensor *t);

#endif /* CTORCH_TENSOR_H */
