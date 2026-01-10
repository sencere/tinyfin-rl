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


class ActionRepeatVec(VectorEnvWrapper):
    def __init__(self, envs: VectorEnv | AsyncVectorEnv, repeat: int):
        super().__init__(envs)
        if repeat <= 0:
            raise ValueError("repeat must be positive")
        self.repeat = repeat

    def step(self, actions: Sequence[int]) -> Tuple[List, List[float], List[bool], List[dict]]:
        total_rewards = [0.0 for _ in actions]
        obs = []
        dones = [False for _ in actions]
        infos = [{} for _ in actions]
        for _ in range(self.repeat):
            obs, rewards, step_dones, step_infos = self.envs.step(actions)
            for i, r in enumerate(rewards):
                total_rewards[i] += r
            for i, done in enumerate(step_dones):
                dones[i] = dones[i] or done
            infos = step_infos
            if all(dones):
                break
        return obs, total_rewards, dones, infos


class _RunningMeanStd:
    def __init__(self, size: int):
        self.mean = [0.0] * size
        self.var = [1.0] * size
        self.count = 0

    def update(self, values: Sequence[float]) -> None:
        if not values:
            return
        if self.count == 0:
            self.mean = list(values)
            self.var = [1e-6] * len(values)
            self.count = 1
            return
        self.count += 1
        for i, v in enumerate(values):
            delta = v - self.mean[i]
            self.mean[i] += delta / self.count
            delta2 = v - self.mean[i]
            self.var[i] += delta * delta2

    def std(self) -> List[float]:
        if self.count <= 1:
            return [1.0 for _ in self.var]
        return [max(1e-6, (v / (self.count - 1)) ** 0.5) for v in self.var]


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(value)]


class NormalizeObservationVec(VectorEnvWrapper):
    def __init__(self, envs: VectorEnv | AsyncVectorEnv, clip: float | None = None):
        super().__init__(envs)
        self.clip = clip
        self._stats: _RunningMeanStd | None = None

    def reset(self) -> List:
        obs = self.envs.reset()
        return self._normalize_batch(obs)

    def step(self, actions: Sequence[int]) -> Tuple[List, List[float], List[bool], List[dict]]:
        obs, rewards, dones, infos = self.envs.step(actions)
        obs = self._normalize_batch(obs)
        return obs, rewards, dones, infos

    def _normalize_batch(self, obs_batch: List):
        flat = _as_list(obs_batch[0]) if obs_batch else [0.0]
        if self._stats is None:
            self._stats = _RunningMeanStd(len(flat))
        out = []
        for obs in obs_batch:
            values = _as_list(obs)
            self._stats.update(values)
            std = self._stats.std()
            norm = [(v - m) / s for v, m, s in zip(values, self._stats.mean, std)]
            if self.clip is not None:
                norm = [max(-self.clip, min(self.clip, v)) for v in norm]
            out.append(norm if isinstance(obs, (list, tuple)) else norm[0])
        return out


class NormalizeRewardVec(VectorEnvWrapper):
    def __init__(self, envs: VectorEnv | AsyncVectorEnv, clip: float | None = None):
        super().__init__(envs)
        self.clip = clip
        self._stats = _RunningMeanStd(1)

    def step(self, actions: Sequence[int]) -> Tuple[List, List[float], List[bool], List[dict]]:
        obs, rewards, dones, infos = self.envs.step(actions)
        out_rewards = []
        for r in rewards:
            self._stats.update([r])
            std = self._stats.std()[0]
            norm = (r - self._stats.mean[0]) / std
            if self.clip is not None:
                norm = max(-self.clip, min(self.clip, norm))
            out_rewards.append(norm)
        return obs, out_rewards, dones, infos
