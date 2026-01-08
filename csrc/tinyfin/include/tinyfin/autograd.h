#ifndef CTORCH_AUTOGRAD_H
#define CTORCH_AUTOGRAD_H

#include "tensor.h"
#include <stddef.h>
#include <stdint.h>

/* Autograd node: tracks dependencies and backward function */
typedef struct AutogradNode {
	Tensor *out;                      /* output tensor */
	Tensor *a;                        /* input tensor A (optional) */
	Tensor *b;                        /* input tensor B (optional) */
	Tensor *bias;                     /* optional bias tensor (for layers) */
	Tensor **inputs;                  /* array of input tensors */
	size_t n_inputs;                  /* number of inputs */
	void (*backward)(struct AutogradNode *); /* backward function */
	int visited;                       /* visited flag for topo sort */
	void (*hook)(struct AutogradNode *);  /* optional hook called after backward */
} AutogradNode;

/* Attach an autograd node to a tensor */
void Tensor_attach_gradients(Tensor *t, AutogradNode *node);

/* Reset gradient to zeros */
void tensor_zero_grad(Tensor *t);

/* Trigger backward pass */
void tensor_backward(Tensor *t);

/* Export autograd graph to DOT (malloc'd string; caller frees). */
char *autograd_to_dot(Tensor *root);

/* Global autograd control (for `no_grad` context) */
void autograd_set_enabled(int enabled);
int autograd_get_enabled(void);

/* Control default retain_graph behavior (useful for higher-order experiments). */
void autograd_set_retain_default(int retain);
int autograd_get_retain_default(void);

/* Backward with retain option: if retain==0, free graph nodes after backward */
void tensor_backward_with_retain(Tensor *t, int retain);

/* Allocate a new tensor like `t` but with requires_grad set to the current
	autograd enabled flag. Useful for intermediate buffers that should become
	grad-tracking when higher-order gradients are enabled. */
Tensor *tensor_new_like_autograd(Tensor *t);

#endif /* CTORCH_AUTOGRAD_H */
