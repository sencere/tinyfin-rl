import os
import sys

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.envs import Env
from tinyfin_rl.rollout import rollout_vector_prealloc, rollout_vector_zero_copy
from tinyfin_rl.spaces import Discrete
from tinyfin_rl.vector_env import VectorEnv


class FixedEnv(Env):
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Discrete(1)
        self._done = False

    def reset(self):
        self._done = False
        return 0

    def step(self, action):
        _ = action
        if self._done:
            raise RuntimeError("step after done")
        self._done = True
        return 0, 1.0, True, {}


class FixedPolicy:
    def act(self, observation):
        _ = observation
        return 0


def test_rollout_prealloc_counts():
    envs = VectorEnv([FixedEnv(), FixedEnv()])
    batch = rollout_vector_prealloc(envs, FixedPolicy(), steps=3)
    assert len(batch.rewards) == 6


def test_rollout_zero_copy_counts():
    envs = VectorEnv([FixedEnv(), FixedEnv()])
    batch, bytes_used = rollout_vector_zero_copy(envs, FixedPolicy(), steps=2, return_bytes=True)
    assert len(batch.actions) == 4
    assert hasattr(batch.observations, "typecode")
    assert bytes_used > 0
