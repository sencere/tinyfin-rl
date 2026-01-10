from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..envs import Env
from ..logging import setup_logging
from ..policy_tinyfin import PPOPolicy


@dataclass
class TRPOTrainer:
    env: Env
    policy: PPOPolicy
    gamma: float = 0.99
    lr: float = 0.05
    kl_coef: float = 0.01
    steps_per_batch: int = 32
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.logger = setup_logging(level=self.log_level)

    def _rollout(self):
        obs = self.env.reset()
        observations = []
        actions = []
        rewards = []
        dones = []
        logps = []
        for _ in range(self.steps_per_batch):
            action = self.policy.act(obs)
            logp = self.policy.log_prob(obs, action).to_numpy().item()
            next_obs, reward, done, _ = self.env.step(action)
            observations.append(int(obs))
            actions.append(int(action))
            rewards.append(float(reward))
            dones.append(bool(done))
            logps.append(float(logp))
            obs = self.env.reset() if done else next_obs
        return observations, actions, rewards, dones, logps

    def train(self, updates: int = 5) -> List[dict]:
        metrics = []
        for update in range(updates):
            obs, acts, rewards, dones, logps = self._rollout()
            returns = []
            g = 0.0
            for r, done in zip(reversed(rewards), reversed(dones)):
                if done:
                    g = 0.0
                g = r + self.gamma * g
                returns.append(g)
            returns.reverse()

            loss = None
            for obs_i, act_i, ret_i, old_logp in zip(obs, acts, returns, logps):
                logp = self.policy.log_prob(obs_i, act_i)
                old_logp_t = self.policy.backend.tensor([old_logp], requires_grad=False)
                kl = old_logp_t - logp
                kl_penalty = kl * self.policy.backend.tensor([self.kl_coef], requires_grad=False)
                ret_t = self.policy.backend.tensor([ret_i], requires_grad=False)
                sample_loss = (-logp * ret_t) + kl_penalty
                loss = sample_loss if loss is None else (loss + sample_loss)

            if loss is not None:
                self.policy.backend.zero_grad(self.policy.parameters())
                loss.backward()
                self.policy.backend.sgd_step(self.policy.parameters(), self.lr)

            record = {"update": update, "return_mean": sum(rewards) / len(rewards)}
            metrics.append(record)
            self.logger.info("update=%s return_mean=%.3f", update, record["return_mean"])
        return metrics
