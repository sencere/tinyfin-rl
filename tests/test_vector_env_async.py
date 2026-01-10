import os
import random
import sys

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.envs import Env
from tinyfin_rl.spaces import Discrete
from tinyfin_rl.vector_env import AsyncVectorEnv, VectorEnv
from tinyfin_rl.vector_wrappers import TimeLimitVec


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
        done = self.steps >= 3
        return self.steps, 1.0, done, {}

class RandomObsEnv(Env):
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Discrete(100)
        self._rng = random.Random(0)

    def reset(self):
        return self._rng.randrange(0, 100)

    def step(self, action):
        _ = action
        return self._rng.randrange(0, 100), 0.0, True, {}

    def seed(self, seed: int) -> int:
        self._rng.seed(seed)
        return seed


def test_async_vector_env_step():
    envs = AsyncVectorEnv([CountingEnv(), CountingEnv()])
    obs = envs.reset()
    assert obs == [0, 0]
    obs, rewards, dones, infos = envs.step([0, 0])
    assert rewards == [1.0, 1.0]
    assert dones == [False, False]
    envs.close()


def test_time_limit_vec_truncates():
    base = VectorEnv([CountingEnv(), CountingEnv()])
    envs = TimeLimitVec(base, max_steps=2)
    envs.reset()
    obs, rewards, dones, infos = envs.step([0, 0])
    assert dones == [False, False]
    obs, rewards, dones, infos = envs.step([0, 0])
    assert dones == [True, True]


def test_vector_env_seed_stable():
    envs = VectorEnv([RandomObsEnv(), RandomObsEnv()])
    envs.seed(123)
    obs1 = envs.reset()
    envs.seed(123)
    obs2 = envs.reset()
    assert obs1 == obs2


def test_async_vector_env_seed_stable():
    envs = AsyncVectorEnv([RandomObsEnv(), RandomObsEnv()])
    envs.seed(321)
    obs1 = envs.reset()
    envs.seed(321)
    obs2 = envs.reset()
    assert obs1 == obs2
    envs.close()
