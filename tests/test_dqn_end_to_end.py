import os
import sys

import pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.algos.dqn import DQNTrainer, QNetwork, hard_update
from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.envs import BanditEnv
from tinyfin_rl.replay import ReplayBuffer, PrioritizedReplayBuffer
from tinyfin_rl.spaces import Discrete
from tinyfin_rl.seed import set_seed


def _backend_or_skip():
    try:
        return TinyfinBackend(device="cpu")
    except RuntimeError:
        pytest.skip("tinyfin backend not available")


def test_dqn_bandit_threshold():
    backend = _backend_or_skip()
    set_seed(0)
    env = BanditEnv([0.1, 0.9])
    action_space = Discrete(2)
    q_net = QNetwork(backend=backend, obs_dim=1, action_space=action_space)
    target_net = QNetwork(backend=backend, obs_dim=1, action_space=action_space)
    hard_update(target_net, q_net)
    replay = ReplayBuffer(capacity=500)
    trainer = DQNTrainer(
        env=env,
        q_net=q_net,
        target_net=target_net,
        replay=replay,
        lr=0.2,
        gamma=0.95,
        batch_size=16,
        eps_start=0.6,
        eps_end=0.05,
        eps_decay_steps=200,
        target_update_steps=50,
    )
    trainer.train(steps=400)
    rewards = []
    for _ in range(200):
        obs = env.reset()
        action = q_net.act(obs, eps=0.0)
        _, reward, _, _ = env.step(action)
        rewards.append(reward)
    assert sum(rewards) / len(rewards) > 0.7


def test_dqn_with_prioritized_replay_runs():
    backend = _backend_or_skip()
    set_seed(1)
    env = BanditEnv([0.2, 0.8])
    action_space = Discrete(2)
    q_net = QNetwork(backend=backend, obs_dim=1, action_space=action_space)
    target_net = QNetwork(backend=backend, obs_dim=1, action_space=action_space)
    hard_update(target_net, q_net)
    replay = PrioritizedReplayBuffer(capacity=200)
    trainer = DQNTrainer(
        env=env,
        q_net=q_net,
        target_net=target_net,
        replay=replay,
        lr=0.15,
        gamma=0.95,
        batch_size=8,
        eps_start=0.6,
        eps_end=0.1,
        eps_decay_steps=100,
        target_update_steps=50,
    )
    trainer.train(steps=120)
