import os
import sys

import pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.multi_agent import SharedPolicyMultiAgent, IndependentMultiAgent
from tinyfin_rl.algos.shared_policy import SharedPolicyTrainer
from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.envs import BanditEnv
from tinyfin_rl.policy_tinyfin import SoftmaxPolicy
from tinyfin_rl.spaces import Discrete
from tinyfin_rl.policy import Policy
from tinyfin_rl.agent import Agent
from tinyfin_rl.replay import ReplayBuffer
from tinyfin_rl.spaces import Discrete


class CountingPolicy(Policy):
    def __init__(self):
        super().__init__(Discrete(2))
        self.calls = 0

    def act(self, observation):
        self.calls += 1
        return 1


def test_shared_policy_multi_agent():
    policy = CountingPolicy()
    multi = SharedPolicyMultiAgent(policy=policy, num_agents=3)
    actions = multi.act_all([0, 1, 2])
    assert actions == [1, 1, 1]
    assert policy.calls == 3


def test_independent_multi_agent_observe():
    policy = CountingPolicy()
    agents = [Agent(policy, ReplayBuffer(capacity=10)) for _ in range(2)]
    multi = IndependentMultiAgent(agents=agents)
    transitions = [(0, 1, 1.0, 0, True), (0, 1, 1.0, 0, True)]
    multi.observe_all(transitions)
    assert len(agents[0].replay_buffer) == 1


def test_shared_policy_trainer_runs():
    try:
        backend = TinyfinBackend(device="cpu")
    except RuntimeError:
        pytest.skip("tinyfin backend not available")
    envs = [BanditEnv([0.2, 0.8]) for _ in range(3)]
    policy = SoftmaxPolicy(backend=backend, obs_dim=1, action_space=Discrete(2))
    trainer = SharedPolicyTrainer(envs=envs, policy=policy, steps_per_batch=8)
    metrics = trainer.train(updates=2)
    assert len(metrics) == 2
