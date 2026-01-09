from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional

from ..envs import Env
from ..logging import setup_logging
from ..policy_tinyfin import PPOPolicy
from ..vector_env import VectorEnv


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


def compute_gae(rewards: Sequence[float], dones: Sequence[bool], values: Sequence[float], gamma: float, lam: float) -> List[float]:
    advantages = []
    last_adv = 0.0
    next_value = 0.0
    for r, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
        if done:
            next_value = 0.0
            last_adv = 0.0
        delta = r + gamma * next_value - value
        last_adv = delta + gamma * lam * last_adv
        advantages.append(last_adv)
        next_value = value
    advantages.reverse()
    return advantages


@dataclass
class PPOTrainer:
    env: Env
    policy: PPOPolicy
    vector_env: Optional[VectorEnv] = None
    gamma: float = 0.99
    lr: float = 0.03
    clip_eps: float = 0.2
    value_coef: float = 0.5
    epochs: int = 2
    steps_per_batch: int = 32
    use_gae: bool = True
    gae_lambda: float = 0.95
    adv_norm: bool = True
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

    def _rollout_vector(self):
        if self.vector_env is None:
            raise RuntimeError("vector_env is required for vectorized rollout")
        obs = self.vector_env.reset()
        observations: List[int] = []
        actions: List[int] = []
        rewards: List[float] = []
        dones: List[bool] = []
        values: List[float] = []
        logps: List[float] = []

        for _ in range(self.steps_per_batch):
            act_list = [self.policy.act(o) for o in obs]
            logp_list = [self.policy.log_prob(o, a).to_numpy().item() for o, a in zip(obs, act_list)]
            val_list = [self.policy.value(o).to_numpy().item() for o in obs]
            next_obs, reward_list, done_list, _ = self.vector_env.step(act_list)

            observations.extend(int(o) for o in obs)
            actions.extend(int(a) for a in act_list)
            rewards.extend(float(r) for r in reward_list)
            dones.extend(bool(d) for d in done_list)
            values.extend(float(v) for v in val_list)
            logps.extend(float(p) for p in logp_list)

            obs = []
            for env, o, done in zip(self.vector_env.envs, next_obs, done_list):
                obs.append(env.reset() if done else o)

        return observations, actions, rewards, dones, values, logps

    def train(self, updates: int = 5):
        metrics = []
        for update in range(updates):
            if self.vector_env is None:
                obs, acts, rewards, dones, values, logps = self._rollout()
            else:
                obs, acts, rewards, dones, values, logps = self._rollout_vector()
            if self.use_gae:
                advantages = compute_gae(rewards, dones, values, self.gamma, self.gae_lambda)
                returns = [a + v for a, v in zip(advantages, values)]
            else:
                returns = compute_returns(rewards, dones, self.gamma)
                advantages = [r - v for r, v in zip(returns, values)]
            if self.adv_norm:
                mean = sum(advantages) / max(len(advantages), 1)
                var = sum((a - mean) ** 2 for a in advantages) / max(len(advantages), 1)
                std = (var ** 0.5) + 1e-8
                advantages = [(a - mean) / std for a in advantages]

            for _ in range(self.epochs):
                loss = None
                for obs_i, act_i, adv, ret, old_logp in zip(obs, acts, advantages, returns, logps):
                    logp = self.policy.log_prob(obs_i, act_i)
                    old_logp_t = self.policy.backend.tensor([old_logp], requires_grad=False)
                    ratio = (logp - old_logp_t).exp()
                    clipped = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    adv_t = self.policy.backend.tensor([adv], requires_grad=False)
                    policy_loss = clipped * adv_t
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
