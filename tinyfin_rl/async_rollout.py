from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .vector_env import AsyncVectorEnv
from .policy_tinyfin import SoftmaxPolicy
from .rollout import RolloutStorage, RolloutBatch


@dataclass
class AsyncRolloutRunner:
    envs: AsyncVectorEnv
    policy: SoftmaxPolicy

    def rollout(self, steps: int) -> RolloutBatch:
        obs = self.envs.reset()
        storage = RolloutStorage.prealloc(steps * len(self.envs.envs))
        for _ in range(steps):
            act_list = [self.policy.act(o) for o in obs]
            next_obs, rewards, dones, _ = self.envs.step(act_list)
            for o, a, r, d in zip(obs, act_list, rewards, dones):
                storage.add(o, a, r, d)
            obs = []
            for env, o, done in zip(self.envs.envs, next_obs, dones):
                obs.append(env.reset() if done else o)
        return storage.to_batch()

    def close(self) -> None:
        self.envs.close()
