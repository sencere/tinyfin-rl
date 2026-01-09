import os
import sys

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.envs import Env
from tinyfin_rl.env_wrappers import ActionRepeat, FrameStack, TimeLimit
from tinyfin_rl.spaces import Discrete


class CountingEnv(Env):
    def __init__(self, max_steps: int | None = None):
        self.action_space = Discrete(2)
        self.observation_space = Discrete(10)
        self.max_steps = max_steps
        self.steps = 0

    def reset(self):
        self.steps = 0
        return 0

    def step(self, action):
        _ = action
        self.steps += 1
        done = self.max_steps is not None and self.steps >= self.max_steps
        return self.steps, 1.0, done, {}


def test_time_limit_truncates():
    env = TimeLimit(CountingEnv(max_steps=10), max_steps=3)
    env.reset()
    obs, reward, terminated, truncated, _ = env.step(0)
    assert (terminated, truncated) == (False, False)
    obs, reward, terminated, truncated, _ = env.step(0)
    assert (terminated, truncated) == (False, False)
    obs, reward, terminated, truncated, _ = env.step(0)
    assert truncated is True


def test_frame_stack_shapes():
    env = FrameStack(CountingEnv(max_steps=2), num_frames=4)
    obs = env.reset()
    assert len(obs) == 4
    obs, reward, done, _ = env.step(0)
    assert len(obs) == 4
    assert reward == 1.0


def test_action_repeat_accumulates_reward():
    env = ActionRepeat(CountingEnv(max_steps=3), repeat=2)
    env.reset()
    obs, reward, terminated, truncated, _ = env.step(0)
    assert reward == 2.0
    assert not terminated
    obs, reward, terminated, truncated, _ = env.step(0)
    assert reward == 1.0
    assert terminated
