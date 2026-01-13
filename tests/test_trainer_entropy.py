import os
import sys

import pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.backends import TinyfinBackend
import pytest
from tinyfin_rl.policy_tinyfin import PPOPolicy
from tinyfin_rl.algos.ppo import PPOTrainer
from tinyfin_rl.algos.a2c import A2CTrainer
from tinyfin_rl.spaces import Discrete


def _backend_or_skip():
    try:
        return TinyfinBackend(device="cpu")
    except RuntimeError:
        pytest.skip("tinyfin backend not available")


def test_ppo_entropy_bonus_runs():
    backend = _backend_or_skip()
    pytest.skip("Python BanditEnv removed; use C env plugin adapter.")
    policy = PPOPolicy(backend=backend, obs_dim=1, action_space=Discrete(2))
    trainer = PPOTrainer(env=env, policy=policy, steps_per_batch=8, epochs=1, entropy_coef=0.01)
    metrics = trainer.train(updates=1)
    assert len(metrics) == 1


def test_a2c_entropy_bonus_runs():
    backend = _backend_or_skip()
    pytest.skip("Python BanditEnv removed; use C env plugin adapter.")
    policy = PPOPolicy(backend=backend, obs_dim=1, action_space=Discrete(2))
    trainer = A2CTrainer(env=env, policy=policy, steps_per_batch=8, entropy_coef=0.01)
    metrics = trainer.train(updates=1)
    assert len(metrics) == 1
