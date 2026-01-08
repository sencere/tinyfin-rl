#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "autograd.h"
#include <stdarg.h>

/* -------------------------
 * Reset visited flags recursively
 * ------------------------- */
static void reset_visits(Tensor *t) {
    if (!t || !t->grad_fn) return;

    AutogradNode *node = t->grad_fn;
    if (!node->visited) return;
    node->visited = 0;

    for (size_t i = 0; i < node->n_inputs; i++) {
        reset_visits(node->inputs[i]);
    }
}

/* simple dynamic buffer for graph export */
typedef struct {
    char *data;
    size_t len;
    size_t cap;
} Buf;

static int buf_append(Buf *b, const char *fmt, ...) {
    if (!b) return 0;
    va_list ap;
    va_start(ap, fmt);
    int needed = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (needed < 0) return 0;
    size_t new_len = b->len + (size_t)needed + 1;
    if (new_len > b->cap) {
        size_t new_cap = b->cap ? b->cap * 2 : 256;
        while (new_cap < new_len) new_cap *= 2;
        char *nd = (char *)realloc(b->data, new_cap);
        if (!nd) return 0;
        b->data = nd;
        b->cap = new_cap;
    }
    va_start(ap, fmt);
    vsnprintf(b->data + b->len, b->cap - b->len, fmt, ap);
    va_end(ap);
    b->len += (size_t)needed;
    return 1;
}

/* Export autograd graph to DOT (malloc'd string; caller frees). */
char *autograd_to_dot(Tensor *root) {
    if (!root || !root->grad_fn) return NULL;
    Tensor *nodes[4096];
    int n = 0;
    /* gather nodes via DFS */
    void gather(Tensor *t) {
        if (!t || !t->grad_fn) return;
        AutogradNode *node = t->grad_fn;
        if (node->visited) return;
        node->visited = 1;
        nodes[n++] = t;
        for (size_t i = 0; i < node->n_inputs; i++) {
            gather(node->inputs[i]);
        }
    }
    gather(root);

    Buf b = {0};
    if (!buf_append(&b, "digraph autograd {\n")) return NULL;
    for (int i = 0; i < n; i++) {
        Tensor *t = nodes[i];
        AutogradNode *node = t->grad_fn;
        buf_append(&b, "  \"t%p\" [label=\"%p device=%d requires_grad=%d\" shape=box];\n",
                   (void*)t, (void*)t, t->device, t->requires_grad);
        for (size_t j = 0; j < node->n_inputs; j++) {
            Tensor *inp = node->inputs[j];
            buf_append(&b, "  \"t%p\" -> \"t%p\";\n", (void*)t, (void*)inp);
        }
    }
    buf_append(&b, "}\n");

    /* reset visited flags */
    for (int i = 0; i < n; i++) {
        if (nodes[i] && nodes[i]->grad_fn) nodes[i]->grad_fn->visited = 0;
    }

    return b.data;
}

/* global flag controlling whether autograd records ops */
static int autograd_enabled = 1;
static int autograd_retain_default = 1;

void autograd_set_enabled(int enabled) {
    autograd_enabled = enabled ? 1 : 0;
}

int autograd_get_enabled(void) { return autograd_enabled; }

void autograd_set_retain_default(int retain) {
    autograd_retain_default = retain ? 1 : 0;
}

int autograd_get_retain_default(void) {
    return autograd_retain_default;
}

__attribute__((constructor))
static void autograd_ctor(void) {
    const char *env = getenv("TINYFIN_RETAIN_GRAPH");
    if (env) {
        int v = atoi(env);
        autograd_retain_default = v ? 1 : 0;
    }
    const char *env2 = getenv("TINYFIN_PERSISTENT_TAPE");
    if (env2) {
        int v = atoi(env2);
        autograd_retain_default = v ? 1 : autograd_retain_default;
    }
}

/* -------------------------
 * Build topological order
 * ------------------------- */
static void build_topo(Tensor *t, Tensor **topo, int *count) {
    if (!t || !t->grad_fn) return;

    AutogradNode *node = t->grad_fn;
    if (node->visited) return;
    node->visited = 1;

    for (size_t i = 0; i < node->n_inputs; i++) {
        build_topo(node->inputs[i], topo, count);
    }

    topo[(*count)++] = t;
}

/* -------------------------
 * Ensure tensor has a grad buffer
 * ------------------------- */
static void ensure_grad(Tensor *t) {
    if (!t || !t->requires_grad) return;
    if (!t->grad) {
        t->grad = tensor_zeros(t->ndim, t->shape);
    }
}

/* -------------------------
 * Reset gradient values to zero
 * ------------------------- */
void tensor_zero_grad(Tensor *t) {
    if (!t) return;
    /* Ensure grad buffer exists for tensors that require grad */
    ensure_grad(t);
    if (!t->grad) return;
    for (size_t i = 0; i < t->grad->size; i++) t->grad->data[i] = 0.0f;
}

/* -------------------------
 * Backward pass
 * ------------------------- */
void tensor_backward(Tensor *loss) {
    tensor_backward_with_retain(loss, autograd_retain_default);
}


void tensor_backward_with_retain(Tensor *loss, int retain) {
    if (!loss || !loss->requires_grad) {
        fprintf(stderr, "tensor_backward() called on tensor with requires_grad=0\n");
        return;
    }
        /* debug prints removed for cleanliness */

    ensure_grad(loss);

    /* For scalar, dL/dL = 1 */
    if (loss->size == 1) {
        if (loss->dtype == DTYPE_FLOAT32) loss->grad->data[0] = 1.0f;
        else {
            /* grad stored as f64 */
            tensor_set_f64_at(loss->grad, 0, 1.0);
        }
    }

    /* Build topological order */
    Tensor *topo[4096];
    int n = 0;
    reset_visits(loss);
    build_topo(loss, topo, &n);

    /* Clear intermediate grads (but keep loss->grad which may be set by caller).
       This prevents previous backward runs from polluting intermediate nodes
       and causing double-counting when running backward multiple times. */
    if (n > 0) {
        for (int ii = 0; ii < n - 1; ii++) {
            Tensor *tt = topo[ii];
            if (!tt) continue;
            ensure_grad(tt);
            if (tt->grad) {
                for (size_t k = 0; k < tt->grad->size; k++) tt->grad->data[k] = 0.0f;
            }
        }
    }

    /* Traverse topo order in reverse */
    for (int i = n - 1; i >= 0; i--) {
        Tensor *t = topo[i];
        if (!t->grad_fn || !t->grad_fn->backward) continue;

        /* ensure inputs have grad buffers */
        for (size_t j = 0; j < t->grad_fn->n_inputs; j++) {
            Tensor *in = t->grad_fn->inputs[j];
            ensure_grad(in);
        }

        t->grad_fn->backward(t->grad_fn);
        if (t->grad_fn->hook) t->grad_fn->hook(t->grad_fn);
    }

    /* Optionally free autograd graph nodes (cleanup) */
    if (!retain) {
        for (int i = 0; i < n; i++) {
            Tensor *t = topo[i];
            if (t->grad_fn) {
                /* free inputs array but not tensors */
                if (t->grad_fn->inputs) free(t->grad_fn->inputs);
                free(t->grad_fn);
                t->grad_fn = NULL;
            }
        }
    }
}

/* -------------------------
 * Attach node
 * ------------------------- */
void Tensor_attach_gradients(Tensor *t, AutogradNode *node) {
    if (!t || !node) return;
    /* Ensure node fields are initialized to safe defaults */
    if (!node->inputs) node->n_inputs = 0;
    node->visited = 0;
    node->hook = NULL;
    /* Respect global autograd_enabled: if disabled, drop node and mark tensor not requiring grad. */
    extern int autograd_get_enabled(void);
    if (!autograd_get_enabled()) {
        /* free inputs array if allocated and node memory */
        if (node->inputs) free(node->inputs);
        free(node);
        t->requires_grad = 0;
        t->grad_fn = NULL;
        return;
    }
    t->grad_fn = node;
}

    Tensor *tensor_new_like_autograd(Tensor *t) {
        extern int autograd_get_enabled(void);
        if (!t) return NULL;
        return tensor_new_like(t, autograd_get_enabled());
    }
