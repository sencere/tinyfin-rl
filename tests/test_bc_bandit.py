import os
import sys

import pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.algos.bc import BCTrainer
from tinyfin_rl.backends import TinyfinBackend
import pytest
from tinyfin_rl.policy_tinyfin import SoftmaxPolicy
from tinyfin_rl.spaces import Discrete
from tinyfin_rl.seed import set_seed


def _backend_or_skip():
    try:
        return TinyfinBackend(device="cpu")
    except RuntimeError:
        pytest.skip("tinyfin backend not available")


def test_bc_bandit_action_match():
    backend = _backend_or_skip()
    set_seed(0)
    pytest.skip("Python BanditEnv removed; use C env plugin adapter.")
    dataset = []
    for _ in range(200):
        obs = env.reset()
        next_obs, reward, done, _ = env.step(1)
        dataset.append((obs, 1, reward, next_obs, done))

    policy = SoftmaxPolicy(backend=backend, obs_dim=1, action_space=Discrete(2))
    trainer = BCTrainer(policy=policy, epochs=5, lr=0.2)
    trainer.train(dataset)

    action1 = 0
    for _ in range(200):
        if policy.act(0) == 1:
            action1 += 1
    assert action1 / 200 > 0.7
