from __future__ import annotations

from typing import Any, Tuple

from ..envs import Env


class GymnasiumEnv(Env):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return obs, info
        return out

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
            return obs, reward, done, info
        return out


def make_gymnasium(name: str, **kwargs) -> GymnasiumEnv:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError("gymnasium is not installed") from exc
    env = gym.make(name, **kwargs)
    return GymnasiumEnv(env)
