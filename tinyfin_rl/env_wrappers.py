from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Tuple

from .envs import Env
from .spaces import Box, Discrete, Space


def _split_step(step_out):
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
    else:
        obs, reward, done, info = step_out
        terminated = bool(done)
        truncated = False
    return obs, float(reward), bool(terminated), bool(truncated), info


def _split_reset(reset_out):
    if isinstance(reset_out, tuple):
        return reset_out[0], reset_out[1]
    return reset_out, None


def _normalize_obs(obs):
    if isinstance(obs, (list, tuple)):
        return [float(x) for x in obs]
    return float(obs)


@dataclass
class RunningMeanStd:
    mean: List[float]
    var: List[float]
    count: int

    @classmethod
    def from_shape(cls, size: int) -> "RunningMeanStd":
        return cls(mean=[0.0] * size, var=[1.0] * size, count=0)

    def update(self, values: Iterable[float]) -> None:
        vals = list(values)
        if not vals:
            return
        if self.count == 0:
            self.mean = vals[:]
            self.var = [1e-6] * len(vals)
            self.count = 1
            return
        self.count += 1
        for i, v in enumerate(vals):
            delta = v - self.mean[i]
            self.mean[i] += delta / self.count
            delta2 = v - self.mean[i]
            self.var[i] += delta * delta2

    def std(self) -> List[float]:
        if self.count <= 1:
            return [1.0 for _ in self.var]
        return [max(1e-6, (v / (self.count - 1)) ** 0.5) for v in self.var]


class EnvWrapper(Env):
    def __init__(self, env: Env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def seed(self, seed: int) -> int:
        return self.env.seed(seed)


class TimeLimit(EnvWrapper):
    def __init__(self, env: Env, max_steps: int):
        super().__init__(env)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self._steps = 0

    def reset(self):
        self._steps = 0
        return self.env.reset()

    def step(self, action):
        self._steps += 1
        obs, reward, terminated, truncated, info = _split_step(self.env.step(action))
        if not terminated and not truncated and self._steps >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


class FrameStack(EnvWrapper):
    def __init__(self, env: Env, num_frames: int):
        super().__init__(env)
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        self.num_frames = num_frames
        self._frames: Deque = deque(maxlen=num_frames)
        self.observation_space = self._stacked_space(env.observation_space)

    def _stacked_space(self, space: Space) -> Space:
        if isinstance(space, Discrete):
            return Box(low=0.0, high=float(space.n - 1), shape=(self.num_frames,))
        if isinstance(space, Box):
            return Box(low=space.low, high=space.high, shape=(self.num_frames,) + tuple(space.shape))
        return space

    def reset(self):
        obs, info = _split_reset(self.env.reset())
        self._frames.clear()
        for _ in range(self.num_frames):
            self._frames.append(obs)
        stacked = list(self._frames)
        if info is not None:
            return stacked, info
        return stacked

    def step(self, action):
        step_out = self.env.step(action)
        obs, reward, terminated, truncated, info = _split_step(step_out)
        self._frames.append(obs)
        stacked = list(self._frames)
        if len(step_out) == 5:
            return stacked, reward, terminated, truncated, info
        return stacked, reward, terminated or truncated, info


class ActionRepeat(EnvWrapper):
    def __init__(self, env: Env, repeat: int):
        super().__init__(env)
        if repeat <= 0:
            raise ValueError("repeat must be positive")
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        obs = None
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.repeat):
            obs, reward, term, trunc, info = _split_step(self.env.step(action))
            total_reward += reward
            terminated = term
            truncated = trunc
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class NormalizeObservation(EnvWrapper):
    def __init__(self, env: Env, clip: float | None = None):
        super().__init__(env)
        self.clip = clip
        self._stats = None

    def reset(self):
        obs, info = _split_reset(self.env.reset())
        norm = self._normalize(obs)
        if info is not None:
            return norm, info
        return norm

    def step(self, action):
        step_out = self.env.step(action)
        obs, reward, terminated, truncated, info = _split_step(step_out)
        obs = self._normalize(obs)
        return obs, reward, terminated, truncated, info

    def _normalize(self, obs):
        values = _normalize_obs(obs)
        vals = values if isinstance(values, list) else [values]
        if self._stats is None:
            self._stats = RunningMeanStd.from_shape(len(vals))
        self._stats.update(vals)
        std = self._stats.std()
        norm = [(v - m) / s for v, m, s in zip(vals, self._stats.mean, std)]
        if self.clip is not None:
            norm = [max(-self.clip, min(self.clip, v)) for v in norm]
        return norm if isinstance(values, list) else norm[0]


class NormalizeReward(EnvWrapper):
    def __init__(self, env: Env, clip: float | None = None):
        super().__init__(env)
        self.clip = clip
        self._stats = RunningMeanStd.from_shape(1)

    def step(self, action):
        obs, reward, terminated, truncated, info = _split_step(self.env.step(action))
        self._stats.update([reward])
        std = self._stats.std()[0]
        norm = (reward - self._stats.mean[0]) / std
        if self.clip is not None:
            norm = max(-self.clip, min(self.clip, norm))
        return obs, norm, terminated, truncated, info
