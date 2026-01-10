import os
import sys

import pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.algos.dqn import DuelingQNetwork, QNetwork, hard_update, hard_update_dueling, soft_update, soft_update_dueling
from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.spaces import Discrete


def _backend_or_skip():
    try:
        return TinyfinBackend(device="cpu")
    except RuntimeError:
        pytest.skip("tinyfin backend not available")


def test_target_network_updates():
    backend = _backend_or_skip()
    action_space = Discrete(2)
    q = QNetwork(backend=backend, obs_dim=1, action_space=action_space)
    tgt = QNetwork(backend=backend, obs_dim=1, action_space=action_space)
    q.w.numpy_view()[:] = 1.0
    q.b.numpy_view()[:] = 2.0
    hard_update(tgt, q)
    assert (tgt.w.numpy_view() == q.w.numpy_view()).all()
    assert (tgt.b.numpy_view() == q.b.numpy_view()).all()
    q.w.numpy_view()[:] = 0.0
    soft_update(tgt, q, tau=0.5)
    assert (tgt.w.numpy_view() != 1.0).any()


def test_dueling_network_updates():
    backend = _backend_or_skip()
    action_space = Discrete(2)
    q = DuelingQNetwork(backend=backend, obs_dim=1, action_space=action_space)
    tgt = DuelingQNetwork(backend=backend, obs_dim=1, action_space=action_space)
    q.w.numpy_view()[:] = 1.0
    q.b.numpy_view()[:] = 2.0
    q.v_w.numpy_view()[:] = 3.0
    q.v_b.numpy_view()[:] = 4.0
    hard_update_dueling(tgt, q)
    assert (tgt.w.numpy_view() == q.w.numpy_view()).all()
    assert (tgt.v_b.numpy_view() == q.v_b.numpy_view()).all()
    q.w.numpy_view()[:] = 0.0
    soft_update_dueling(tgt, q, tau=0.5)
    assert (tgt.w.numpy_view() != 1.0).any()
