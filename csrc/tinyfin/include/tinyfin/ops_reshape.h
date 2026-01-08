#ifndef CTORCH_OPS_RESHAPE_H
#define CTORCH_OPS_RESHAPE_H

#include "tensor.h"

/* reshape: new_shape must have same total elements as input. 
 * If the input is non-contiguous (e.g. has stride 0 from broadcasting),
 * this implementation falls back to a contiguous copy.
 */
Tensor *tensor_reshape(Tensor *a, int new_ndim, const int *new_shape);

/* permute: reorder dimensions (view, no data copy) */
Tensor *tensor_permute(Tensor *a, const int *order, int ndim);

/* transpose: 2D transpose (alias for permute with [1,0]) */
Tensor *tensor_transpose(Tensor *a);
Tensor *tensor_permute(Tensor *a, const int *order, int ndim);

/* squeeze: remove dimensions of size 1. If dim = -1 remove all singleton dims,
 * otherwise remove only the specified dimension (if its size==1). Returns a view.
 */
Tensor *tensor_squeeze(Tensor *a, int dim);

/* unsqueeze: insert a dimension of size 1 at position `dim` (0..ndim).
 * Returns a view sharing data.
 */
Tensor *tensor_unsqueeze(Tensor *a, int dim);

#endif
