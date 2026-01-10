import os
import sys

import pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.envs import ContinuousBanditEnv
from tinyfin_rl.algos.td3 import Actor, Critic, TD3Trainer
from tinyfin_rl.replay import ReplayBuffer
from tinyfin_rl.seed import set_seed


def _backend_or_skip():
    try:
        return TinyfinBackend(device="cpu")
    except RuntimeError:
        pytest.skip("tinyfin backend not available")


def test_td3_trainer_runs():
    backend = _backend_or_skip()
    set_seed(0)
    env = ContinuousBanditEnv(target=0.6)
    actor = Actor(backend=backend)
    critic1 = Critic(backend=backend)
    critic2 = Critic(backend=backend)
    actor_t = Actor(backend=backend)
    critic1_t = Critic(backend=backend)
    critic2_t = Critic(backend=backend)
    trainer = TD3Trainer(
        env=env,
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_target=actor_t,
        critic1_target=critic1_t,
        critic2_target=critic2_t,
        replay=ReplayBuffer(capacity=500),
        batch_size=8,
    )
    metrics = trainer.train(steps=120)
    assert len(metrics) > 0
