from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .vector_env import VectorEnv
from .policy_tinyfin import SoftmaxPolicy


@dataclass
class RolloutBatch:
    observations: List[int]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]


def rollout_vector(envs: VectorEnv, policy: SoftmaxPolicy, steps: int) -> RolloutBatch:
    obs = envs.reset()
    observations: List[int] = []
    actions: List[int] = []
    rewards: List[float] = []
    dones: List[bool] = []

    for _ in range(steps):
        act_list = [policy.act(o) for o in obs]
        next_obs, reward_list, done_list, _ = envs.step(act_list)
        observations.extend(int(o) for o in obs)
        actions.extend(int(a) for a in act_list)
        rewards.extend(float(r) for r in reward_list)
        dones.extend(bool(d) for d in done_list)
        obs = next_obs

    return RolloutBatch(observations, actions, rewards, dones)
