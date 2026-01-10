from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import List, Sequence

from .vector_env import VectorEnv
from .policy_tinyfin import SoftmaxPolicy


@dataclass
class RolloutBatch:
    observations: Sequence[int]
    actions: Sequence[int]
    rewards: Sequence[float]
    dones: Sequence[bool]


@dataclass
class RolloutStorage:
    observations: List[int]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]
    idx: int = 0

    @classmethod
    def prealloc(cls, size: int) -> "RolloutStorage":
        return cls(
            observations=[0 for _ in range(size)],
            actions=[0 for _ in range(size)],
            rewards=[0.0 for _ in range(size)],
            dones=[False for _ in range(size)],
            idx=0,
        )

    def add(self, obs: int, action: int, reward: float, done: bool) -> None:
        if self.idx >= len(self.observations):
            raise IndexError("rollout storage overflow")
        self.observations[self.idx] = int(obs)
        self.actions[self.idx] = int(action)
        self.rewards[self.idx] = float(reward)
        self.dones[self.idx] = bool(done)
        self.idx += 1

    def to_batch(self) -> RolloutBatch:
        return RolloutBatch(
            observations=self.observations[: self.idx],
            actions=self.actions[: self.idx],
            rewards=self.rewards[: self.idx],
            dones=self.dones[: self.idx],
        )


@dataclass
class ArrayRolloutStorage:
    observations: array
    actions: array
    rewards: array
    dones: array
    idx: int = 0

    @classmethod
    def prealloc(cls, size: int) -> "ArrayRolloutStorage":
        return cls(
            observations=array("i", [0 for _ in range(size)]),
            actions=array("i", [0 for _ in range(size)]),
            rewards=array("f", [0.0 for _ in range(size)]),
            dones=array("b", [0 for _ in range(size)]),
            idx=0,
        )

    def add(self, obs: int, action: int, reward: float, done: bool) -> None:
        if self.idx >= len(self.observations):
            raise IndexError("rollout storage overflow")
        self.observations[self.idx] = int(obs)
        self.actions[self.idx] = int(action)
        self.rewards[self.idx] = float(reward)
        self.dones[self.idx] = 1 if done else 0
        self.idx += 1

    def to_batch(self) -> RolloutBatch:
        return RolloutBatch(
            observations=self.observations[: self.idx],
            actions=self.actions[: self.idx],
            rewards=self.rewards[: self.idx],
            dones=[bool(x) for x in self.dones[: self.idx]],
        )

    def bytes_used(self) -> int:
        return (
            self.observations.buffer_info()[1] * self.observations.itemsize
            + self.actions.buffer_info()[1] * self.actions.itemsize
            + self.rewards.buffer_info()[1] * self.rewards.itemsize
            + self.dones.buffer_info()[1] * self.dones.itemsize
        )


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
        obs = []
        for env, o, done in zip(envs.envs, next_obs, done_list):
            obs.append(env.reset() if done else o)

    return RolloutBatch(observations, actions, rewards, dones)


def rollout_vector_prealloc(envs: VectorEnv, policy: SoftmaxPolicy, steps: int) -> RolloutBatch:
    obs = envs.reset()
    total = steps * len(envs.envs)
    storage = RolloutStorage.prealloc(total)

    for _ in range(steps):
        act_list = [policy.act(o) for o in obs]
        next_obs, reward_list, done_list, _ = envs.step(act_list)
        for o, a, r, d in zip(obs, act_list, reward_list, done_list):
            storage.add(o, a, r, d)
        obs = []
        for env, o, done in zip(envs.envs, next_obs, done_list):
            obs.append(env.reset() if done else o)

    return storage.to_batch()


def rollout_vector_zero_copy(envs: VectorEnv, policy: SoftmaxPolicy, steps: int, return_bytes: bool = False):
    obs = envs.reset()
    total = steps * len(envs.envs)
    storage = ArrayRolloutStorage.prealloc(total)

    for _ in range(steps):
        act_list = [policy.act(o) for o in obs]
        next_obs, reward_list, done_list, _ = envs.step(act_list)
        for o, a, r, d in zip(obs, act_list, reward_list, done_list):
            storage.add(o, a, r, d)
        obs = []
        for env, o, done in zip(envs.envs, next_obs, done_list):
            obs.append(env.reset() if done else o)

    batch = storage.to_batch()
    if return_bytes:
        return batch, storage.bytes_used()
    return batch
