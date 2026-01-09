import os
import sys

import numpy as np
import pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.policy_tinyfin import PPOPolicy
from tinyfin_rl.spaces import Discrete


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def test_policy_entropy_matches_numpy():
    try:
        backend = TinyfinBackend(device="cpu")
    except RuntimeError:
        pytest.skip("tinyfin backend not available")

    action_space = Discrete(3)
    policy = PPOPolicy(backend=backend, obs_dim=1, action_space=action_space)

    policy.w.numpy_view()[:] = np.array([[1.0, -1.0, 0.5]], dtype=np.float32)
    policy.b.numpy_view()[:] = np.array([0.2, -0.1, 0.3], dtype=np.float32)

    logits = policy.logits(0).to_numpy().reshape(-1)
    probs = _softmax(logits)
    expected = -np.sum(probs * np.log(probs + 1e-8))

    got = policy.entropy(0).to_numpy().item()
    assert abs(got - expected) < 1e-5
