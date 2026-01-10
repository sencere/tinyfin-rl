import os
import sys

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.envs import Env
from tinyfin_rl.spaces import Discrete
from tinyfin_rl.vector_env import VectorEnv
from tinyfin_rl.vector_wrappers import ActionRepeatVec, NormalizeObservationVec, NormalizeRewardVec


class CountingEnv(Env):
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Discrete(10)
        self.steps = 0

    def reset(self):
        self.steps = 0
        return 0

    def step(self, action):
        _ = action
        self.steps += 1
        done = self.steps >= 2
        return self.steps, 1.0, done, {}


def test_action_repeat_vec_accumulates():
    envs = ActionRepeatVec(VectorEnv([CountingEnv(), CountingEnv()]), repeat=2)
    envs.reset()
    obs, rewards, dones, infos = envs.step([0, 0])
    assert rewards == [2.0, 2.0]
    assert dones == [True, True]


def test_normalize_observation_vec():
    envs = NormalizeObservationVec(VectorEnv([CountingEnv(), CountingEnv()]))
    obs = envs.reset()
    assert len(obs) == 2
    obs, rewards, dones, infos = envs.step([0, 0])
    assert len(obs) == 2


def test_normalize_reward_vec():
    envs = NormalizeRewardVec(VectorEnv([CountingEnv(), CountingEnv()]))
    envs.reset()
    obs, rewards, dones, infos = envs.step([0, 0])
    assert len(rewards) == 2
