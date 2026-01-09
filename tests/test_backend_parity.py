import os
import sys
import pytest
import numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.policy_tinyfin import SoftmaxPolicy
from tinyfin_rl.spaces import Discrete


def test_policy_inference_cpu_cuda_parity():
    try:
        backend_cpu = TinyfinBackend(device="cpu")
    except RuntimeError:
        pytest.skip("tinyfin backend not available")

    try:
        backend_cuda = TinyfinBackend(device="cuda")
    except RuntimeError:
        pytest.skip("cuda backend not available")

    action_space = Discrete(4)
    policy_cpu = SoftmaxPolicy(backend=backend_cpu, obs_dim=1, action_space=action_space)
    policy_cuda = SoftmaxPolicy(backend=backend_cuda, obs_dim=1, action_space=action_space)

    w = np.random.randn(1, action_space.n).astype(np.float32) * 0.1
    b = np.random.randn(action_space.n).astype(np.float32) * 0.1
    policy_cpu.w.numpy_view()[:] = w
    policy_cpu.b.numpy_view()[:] = b
    policy_cuda.w.numpy_view()[:] = w
    policy_cuda.b.numpy_view()[:] = b

    logits_cpu = policy_cpu.logits(0).to_numpy()
    logits_cuda = policy_cuda.logits(0).to_numpy()
    np.testing.assert_allclose(logits_cpu, logits_cuda, rtol=1e-5, atol=1e-6)
