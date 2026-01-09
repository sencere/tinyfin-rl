import os
import sys

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.envs import Env
from tinyfin_rl.rollout import rollout_vector
from tinyfin_rl.spaces import Discrete
from tinyfin_rl.vector_env import VectorEnv


class ResetRequiredEnv(Env):
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Discrete(1)
        self._done = False

    def reset(self):
        self._done = False
        return 0

    def step(self, action):
        if self._done:
            raise RuntimeError("step called after done without reset")
        _ = action
        self._done = True
        return 0, 1.0, True, {}


class FixedPolicy:
    def act(self, observation):
        _ = observation
        return 0


def test_rollout_vector_resets_done_envs():
    envs = VectorEnv([ResetRequiredEnv(), ResetRequiredEnv()])
    batch = rollout_vector(envs, FixedPolicy(), steps=3)
    assert len(batch.rewards) == 6
