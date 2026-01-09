from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..envs import Env
from ..logging import setup_logging
from ..policy_tinyfin import SoftmaxPolicy


def compute_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()
    return returns


@dataclass
class ReinforceTrainer:
    env: Env
    policy: SoftmaxPolicy
    gamma: float = 0.99
    lr: float = 0.05
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.logger = setup_logging(level=self.log_level)

    def train(self, episodes: int = 10):
        metrics = []
        for ep in range(episodes):
            obs = self.env.reset()
            done = False
            observations: List[int] = []
            actions: List[int] = []
            rewards: List[float] = []

            while not done:
                action = self.policy.act(obs)
                next_obs, reward, done, _ = self.env.step(action)
                observations.append(int(obs))
                actions.append(int(action))
                rewards.append(float(reward))
                obs = next_obs

            returns = compute_returns(rewards, self.gamma)
            loss = None
            for obs_i, act_i, ret in zip(observations, actions, returns):
                logp = self.policy.log_prob(obs_i, act_i)
                scale = self.policy.backend.tensor([-float(ret)], requires_grad=False)
                term = logp * scale
                loss = term if loss is None else (loss + term)

            if loss is not None:
                self.policy.backend.zero_grad(self.policy.parameters())
                loss.backward()
                self.policy.backend.sgd_step(self.policy.parameters(), self.lr)

            record = {"episode": ep, "return": sum(rewards), "steps": len(rewards)}
            metrics.append(record)
            self.logger.info("episode=%s return=%.3f steps=%s", ep, record["return"], record["steps"])
        return metrics
