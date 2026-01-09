from __future__ import annotations

from typing import List, Sequence, Tuple

from .vector_env import VectorEnv, AsyncVectorEnv


class VectorEnvWrapper:
    def __init__(self, envs: VectorEnv | AsyncVectorEnv):
        self.envs = envs

    def reset(self) -> List:
        return self.envs.reset()

    def step(self, actions: Sequence[int]) -> Tuple[List, List[float], List[bool], List[dict]]:
        return self.envs.step(actions)

    def seed(self, seed: int) -> List[int]:
        return self.envs.seed(seed)

    def close(self) -> None:
        if hasattr(self.envs, "close"):
            self.envs.close()


class TimeLimitVec(VectorEnvWrapper):
    def __init__(self, envs: VectorEnv | AsyncVectorEnv, max_steps: int):
        super().__init__(envs)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self._steps: List[int] = []

    def reset(self) -> List:
        obs = self.envs.reset()
        self._steps = [0 for _ in obs]
        return obs

    def step(self, actions: Sequence[int]) -> Tuple[List, List[float], List[bool], List[dict]]:
        obs, rewards, dones, infos = self.envs.step(actions)
        for i in range(len(obs)):
            self._steps[i] += 1
            if not dones[i] and self._steps[i] >= self.max_steps:
                dones[i] = True
                info = dict(infos[i])
                info["time_limit"] = True
                infos[i] = info
        return obs, rewards, dones, infos
