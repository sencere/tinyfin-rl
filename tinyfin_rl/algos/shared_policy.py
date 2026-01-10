from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..envs import Env
from ..logging import setup_logging
from ..policy_tinyfin import SoftmaxPolicy
from .ppo import compute_returns


@dataclass
class SharedPolicyTrainer:
    envs: Sequence[Env]
    policy: SoftmaxPolicy
    gamma: float = 0.99
    lr: float = 0.1
    steps_per_batch: int = 32
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not self.envs:
            raise ValueError("envs must be non-empty")
        self.logger = setup_logging(level=self.log_level)

    def _rollout(self):
        obs_list = [env.reset() for env in self.envs]
        observations: List[int] = []
        actions: List[int] = []
        rewards: List[float] = []
        dones: List[bool] = []

        for _ in range(self.steps_per_batch):
            act_list = [self.policy.act(o) for o in obs_list]
            next_obs_list = []
            for env, action, obs in zip(self.envs, act_list, obs_list):
                next_obs, reward, done, _ = env.step(action)
                observations.append(int(obs))
                actions.append(int(action))
                rewards.append(float(reward))
                dones.append(bool(done))
                next_obs_list.append(env.reset() if done else next_obs)
            obs_list = next_obs_list

        return observations, actions, rewards, dones

    def train(self, updates: int = 5) -> List[dict]:
        metrics = []
        for update in range(updates):
            obs, acts, rewards, dones = self._rollout()
            returns = compute_returns(rewards, dones, self.gamma)
            loss = None
            for obs_i, act_i, ret_i in zip(obs, acts, returns):
                logp = self.policy.log_prob(obs_i, act_i)
                ret_t = self.policy.backend.tensor([ret_i], requires_grad=False)
                sample_loss = logp * ret_t * self.policy.backend.tensor([-1.0], requires_grad=False)
                loss = sample_loss if loss is None else (loss + sample_loss)
            if loss is not None:
                self.policy.backend.zero_grad(self.policy.parameters())
                loss.backward()
                self.policy.backend.sgd_step(self.policy.parameters(), self.lr)
            record = {"update": update, "return_mean": sum(rewards) / len(rewards)}
            metrics.append(record)
            self.logger.info("update=%s return_mean=%.3f", update, record["return_mean"])
        return metrics
