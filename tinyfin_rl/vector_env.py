from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Sequence, Tuple

from .envs import Env


class VectorEnv:
    def __init__(self, envs: Sequence[Env]):
        if not envs:
            raise ValueError("envs must be non-empty")
        self.envs = list(envs)

    def reset(self) -> List:
        return [env.reset() for env in self.envs]

    def step(self, actions: Sequence[int]) -> Tuple[List, List[float], List[bool], List[dict]]:
        if len(actions) != len(self.envs):
            raise ValueError("actions length must match envs length")
        obs, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            step_out = env.step(action)
            if len(step_out) == 5:
                o, r, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                o, r, done, info = step_out
            obs.append(o)
            rewards.append(float(r))
            dones.append(bool(done))
            infos.append(info)
        return obs, rewards, dones, infos

    def seed(self, seed: int) -> List[int]:
        seeds = []
        for i, env in enumerate(self.envs):
            seeds.append(env.seed(seed + i))
        return seeds


class AsyncVectorEnv:
    def __init__(self, envs: Sequence[Env], max_workers: int | None = None):
        if not envs:
            raise ValueError("envs must be non-empty")
        self.envs = list(envs)
        self._executor = ThreadPoolExecutor(max_workers=max_workers or len(self.envs))

    def reset(self) -> List:
        return list(self._executor.map(lambda e: e.reset(), self.envs))

    def step(self, actions: Sequence[int]) -> Tuple[List, List[float], List[bool], List[dict]]:
        if len(actions) != len(self.envs):
            raise ValueError("actions length must match envs length")

        def _step(pair: tuple[Env, int]):
            env, action = pair
            step_out = env.step(action)
            if len(step_out) == 5:
                o, r, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                o, r, done, info = step_out
            return o, float(r), bool(done), info

        results = list(self._executor.map(_step, zip(self.envs, actions)))
        obs, rewards, dones, infos = [], [], [], []
        for o, r, d, info in results:
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return obs, rewards, dones, infos

    def seed(self, seed: int) -> List[int]:
        seeds = []
        for i, env in enumerate(self.envs):
            seeds.append(env.seed(seed + i))
        return seeds

    def close(self) -> None:
        self._executor.shutdown(wait=True)
