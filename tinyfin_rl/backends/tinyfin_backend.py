from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

try:
    import tinyfin as tf
except ImportError:  # tinyfin is optional for tinyfin-rl
    tf = None


@dataclass
class TinyfinBackend:
    device: str = "cpu"

    def __post_init__(self) -> None:
        if tf is None:
            raise RuntimeError("tinyfin is not available; set PYTHONPATH to tinyfin/python")
        if not hasattr(tf, "Tensor"):
            raise RuntimeError("tinyfin Python bindings not found; set PYTHONPATH to tinyfin/python")
        self.tf = tf
        if hasattr(tf, "backend_set"):
            target = "cuda" if self.device == "cuda" else "cpu"
            ok = tf.backend_set(target)
            if target == "cuda" and not ok:
                raise RuntimeError("tinyfin CUDA backend not available")

    def tensor(self, data, requires_grad: bool = False):
        arr = np.asarray(data, dtype=np.float32)
        t = tf.Tensor.from_numpy(arr, requires_grad=requires_grad)
        if self.device == "cuda":
            t.set_device(1)
        return t

    def one_hot(self, index: int, depth: int):
        vec = np.zeros((depth,), dtype=np.float32)
        if 0 <= index < depth:
            vec[index] = 1.0
        return self.tensor(vec, requires_grad=False)

    def matmul(self, a, b):
        return a.matmul(b)

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def log(self, a):
        return a.log()

    def softmax(self, a):
        return a.softmax()

    def sum(self, a):
        return a.sum()

    def zero_grad(self, params: Iterable):
        for p in params:
            p.zero_grad()

    def sgd_step(self, params: Iterable, lr: float):
        for p in params:
            grad = p.grad_numpy()
            if grad is None:
                continue
            if not np.isfinite(grad).all():
                continue
            p.numpy_view()[:] = p.numpy_view() - lr * grad
