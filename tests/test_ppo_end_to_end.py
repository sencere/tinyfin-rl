import os
import sys

import pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.envs import BanditEnv
from tinyfin_rl.policy_tinyfin import PPOPolicy
from tinyfin_rl.algos.ppo import PPOTrainer
from tinyfin_rl.spaces import Discrete
from tinyfin_rl.seed import set_seed
from tinyfin_rl.vector_env import VectorEnv


def _backend_or_skip():
    try:
        return TinyfinBackend(device="cpu")
    except RuntimeError:
        pytest.skip("tinyfin backend not available")


def test_ppo_end_to_end_vector_env():
    backend = _backend_or_skip()
    set_seed(0)
    envs = VectorEnv([BanditEnv([0.2, 0.8]) for _ in range(4)])
    policy = PPOPolicy(backend=backend, obs_dim=1, action_space=Discrete(2))
    trainer = PPOTrainer(env=envs.envs[0], policy=policy, vector_env=envs, steps_per_batch=8, epochs=1)
    metrics = trainer.train(updates=2)
    assert len(metrics) == 2
    assert all("return_mean" in record for record in metrics)


def test_ppo_reaches_bandit_threshold():
    backend = _backend_or_skip()
    set_seed(0)
    envs = VectorEnv([BanditEnv([0.1, 0.9]) for _ in range(8)])
    policy = PPOPolicy(backend=backend, obs_dim=1, action_space=Discrete(2))
    trainer = PPOTrainer(
        env=envs.envs[0],
        policy=policy,
        vector_env=envs,
        steps_per_batch=16,
        epochs=2,
        lr=0.1,
        entropy_coef=0.0,
        adv_norm=True,
    )
    metrics = trainer.train(updates=8)
    assert metrics[-1]["return_mean"] > 0.6
