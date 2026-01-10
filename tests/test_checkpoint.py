import os
import sys
import tempfile

import pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.policy_tinyfin import PPOPolicy
from tinyfin_rl.algos.dqn import QNetwork, hard_update
from tinyfin_rl.spaces import Discrete
from tinyfin_rl.checkpoint import save_policy_checkpoint, load_policy_checkpoint, save_q_checkpoint, load_q_checkpoint


def _backend_or_skip():
    try:
        return TinyfinBackend(device="cpu")
    except RuntimeError:
        pytest.skip("tinyfin backend not available")


def test_policy_checkpoint_roundtrip():
    backend = _backend_or_skip()
    policy = PPOPolicy(backend=backend, obs_dim=1, action_space=Discrete(2))
    policy.w.numpy_view()[:] = 1.0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ppo.npz")
        save_policy_checkpoint(path, policy)
        policy2 = PPOPolicy(backend=backend, obs_dim=1, action_space=Discrete(2))
        load_policy_checkpoint(path, policy2)
        assert (policy2.w.numpy_view() == policy.w.numpy_view()).all()


def test_q_checkpoint_roundtrip():
    backend = _backend_or_skip()
    q = QNetwork(backend=backend, obs_dim=1, action_space=Discrete(2))
    t = QNetwork(backend=backend, obs_dim=1, action_space=Discrete(2))
    hard_update(t, q)
    q.w.numpy_view()[:] = 2.0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dqn.npz")
        save_q_checkpoint(path, q, t)
        q2 = QNetwork(backend=backend, obs_dim=1, action_space=Discrete(2))
        t2 = QNetwork(backend=backend, obs_dim=1, action_space=Discrete(2))
        load_q_checkpoint(path, q2, t2)
        assert (q2.w.numpy_view() == q.w.numpy_view()).all()
