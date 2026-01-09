import os
import sys

import pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.backends import TinyfinBackend


def test_backend_optimizer_step_updates_param():
    try:
        backend = TinyfinBackend(device="cpu")
    except RuntimeError:
        pytest.skip("tinyfin backend not available")

    w = backend.tensor([0.0], requires_grad=True)
    x = backend.tensor([1.0], requires_grad=False)
    y = backend.tensor([1.0], requires_grad=False)

    pred = w * x
    diff = pred - y
    loss = diff * diff

    backend.zero_grad([w])
    loss.backward()
    backend.sgd_step([w], lr=0.1)

    assert w.to_numpy()[0] != 0.0
