from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..envs import Env
from ..logging import setup_logging
from ..policy_tinyfin import PPOPolicy


def compute_returns(rewards: Sequence[float], dones: Sequence[bool], gamma: float) -> List[float]:
    returns = []
    g = 0.0
    for r, done in zip(reversed(rewards), reversed(dones)):
        if done:
            g = 0.0
        g = r + gamma * g
        returns.append(g)
    returns.reverse()
    return returns


@dataclass
class A2CTrainer:
    env: Env
    policy: PPOPolicy
    gamma: float = 0.99
    lr: float = 0.05
    value_coef: float = 0.5
    steps_per_batch: int = 32
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.logger = setup_logging(level=self.log_level)

    def _rollout(self):
        obs = self.env.reset()
        observations: List[int] = []
        actions: List[int] = []
        rewards: List[float] = []
        dones: List[bool] = []
        values: List[float] = []
        logps: List[float] = []

        for _ in range(self.steps_per_batch):
            action = self.policy.act(obs)
            logp = self.policy.log_prob(obs, action).to_numpy().item()
            val = self.policy.value(obs).to_numpy().item()
            next_obs, reward, done, _ = self.env.step(action)

            observations.append(int(obs))
            actions.append(int(action))
            rewards.append(float(reward))
            dones.append(bool(done))
            values.append(float(val))
            logps.append(float(logp))

            obs = self.env.reset() if done else next_obs

        return observations, actions, rewards, dones, values, logps

    def train(self, updates: int = 5):
        metrics = []
        for update in range(updates):
            obs, acts, rewards, dones, values, logps = self._rollout()
            returns = compute_returns(rewards, dones, self.gamma)
            advantages = [r - v for r, v in zip(returns, values)]

            loss = None
            for obs_i, act_i, adv, ret in zip(obs, acts, advantages, returns):
                logp = self.policy.log_prob(obs_i, act_i)
                adv_t = self.policy.backend.tensor([adv], requires_grad=False)
                policy_loss = logp * adv_t
                policy_loss = policy_loss * self.policy.backend.tensor([-1.0], requires_grad=False)

                value_pred = self.policy.value(obs_i)
                value_target = self.policy.backend.tensor([ret], requires_grad=False)
                vdiff = value_pred - value_target
                value_loss = vdiff * vdiff
                value_loss = value_loss * self.policy.backend.tensor([self.value_coef], requires_grad=False)

                total = policy_loss + value_loss
                loss = total if loss is None else (loss + total)

            if loss is not None:
                self.policy.backend.zero_grad(self.policy.parameters())
                loss.backward()
                self.policy.backend.sgd_step(self.policy.parameters(), self.lr)

            record = {"update": update, "return_mean": sum(rewards) / len(rewards)}
            metrics.append(record)
            self.logger.info("update=%s return_mean=%.3f", update, record["return_mean"])
        return metrics
