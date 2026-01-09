from __future__ import annotations

from typing import List, Sequence, Tuple

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
