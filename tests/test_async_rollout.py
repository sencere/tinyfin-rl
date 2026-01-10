import os
import sys

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.envs import Env
from tinyfin_rl.spaces import Discrete
from tinyfin_rl.vector_env import AsyncVectorEnv
from tinyfin_rl.async_rollout import AsyncRolloutRunner


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


def test_async_rollout_runner():
    envs = AsyncVectorEnv([FixedEnv(), FixedEnv()])
    runner = AsyncRolloutRunner(envs=envs, policy=FixedPolicy())
    batch = runner.rollout(steps=2)
    assert len(batch.rewards) == 4
    runner.close()
