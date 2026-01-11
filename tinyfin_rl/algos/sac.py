from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List

from ..envs import Env
from ..logging import setup_logging
from ..backends.tinyfin_backend import TinyfinBackend
from ..replay import ReplayBuffer


@dataclass
class SACActor:
    backend: TinyfinBackend
    log_std_init: float = -0.5

    def __post_init__(self) -> None:
        self.w = self.backend.tensor([0.0], requires_grad=True)
        self.b = self.backend.tensor([0.0], requires_grad=True)
        self.log_std = self.backend.tensor([self.log_std_init], requires_grad=True)

    def parameters(self):
        return [self.w, self.b, self.log_std]

    def forward(self, obs: float):
        obs_t = self.backend.tensor([float(obs)], requires_grad=False)
        return self.backend.add(self.backend.mul(obs_t, self.w), self.b)

    def _log_prob(self, action_t, mu, log_std, std):
        diff = (action_t - mu) / std
        diff2 = diff * diff
        neg_half = self.backend.tensor([-0.5], requires_grad=False)
        log_two_pi = self.backend.tensor([0.5 * math.log(2.0 * math.pi)], requires_grad=False)
        return diff2 * neg_half - log_std - log_two_pi

    def sample(self, obs: float, action_limit: float = 1.0):
        mu = self.forward(obs)
        log_std = self.log_std
        std = log_std.exp()
        eps = random.gauss(0.0, 1.0)
        eps_t = self.backend.tensor([eps], requires_grad=False)
        action_t = mu + std * eps_t
        action_t = action_t.clamp(-action_limit, action_limit)
        log_prob = self._log_prob(action_t, mu, log_std, std)
        action = float(action_t.to_numpy().item())
        return action, log_prob, action_t

    def act(self, obs: float, action_limit: float = 1.0) -> float:
        action, _, _ = self.sample(obs, action_limit=action_limit)
        return action


@dataclass
class SACCritic:
    backend: TinyfinBackend

    def __post_init__(self) -> None:
        self.w_obs = self.backend.tensor([0.0], requires_grad=True)
        self.w_act = self.backend.tensor([0.0], requires_grad=True)
        self.b = self.backend.tensor([0.0], requires_grad=True)

    def parameters(self):
        return [self.w_obs, self.w_act, self.b]

    def forward(self, obs, act):
        if hasattr(obs, "to_numpy"):
            obs_t = obs
        else:
            obs_t = self.backend.tensor([float(obs)], requires_grad=False)
        if hasattr(act, "to_numpy"):
            act_t = act
        else:
            act_t = self.backend.tensor([float(act)], requires_grad=False)
        q = self.backend.add(self.backend.mul(obs_t, self.w_obs), self.backend.mul(act_t, self.w_act))
        return self.backend.add(q, self.b)

    def copy_from(self, other: "SACCritic") -> None:
        self.w_obs.numpy_view()[:] = other.w_obs.numpy_view()
        self.w_act.numpy_view()[:] = other.w_act.numpy_view()
        self.b.numpy_view()[:] = other.b.numpy_view()


def soft_update(target, source, tau: float = 0.005) -> None:
    if tau <= 0 or tau > 1:
        raise ValueError("tau must be in (0, 1]")
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.numpy_view()[:] = (1.0 - tau) * t_param.numpy_view() + tau * s_param.numpy_view()


@dataclass
class SACTrainer:
    env: Env
    actor: SACActor
    critic1: SACCritic
    critic2: SACCritic
    critic1_target: SACCritic
    critic2_target: SACCritic
    replay: ReplayBuffer
    gamma: float = 0.99
    actor_lr: float = 0.05
    critic_lr: float = 0.1
    batch_size: int = 32
    action_limit: float = 1.0
    alpha: float = 0.2
    tau: float = 0.005
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.logger = setup_logging(level=self.log_level)
        self.critic1_target.copy_from(self.critic1)
        self.critic2_target.copy_from(self.critic2)

    def train(self, steps: int = 500) -> List[dict]:
        metrics = []
        obs = self.env.reset()
        for step in range(steps):
            action = self.actor.act(obs, action_limit=self.action_limit)
            next_obs, reward, done, _ = self.env.step(action)
            self.replay.add(obs, action, float(reward), next_obs, bool(done))
            obs = self.env.reset() if done else next_obs

            if len(self.replay) >= self.batch_size:
                batch = self.replay.sample(self.batch_size)
                loss1 = None
                loss2 = None
                for b_obs, b_action, b_reward, b_next_obs, b_done in batch:
                    _, next_logp, next_action_t = self.actor.sample(b_next_obs, action_limit=self.action_limit)
                    q1_t = self.critic1_target.forward(b_next_obs, next_action_t).to_numpy().item()
                    q2_t = self.critic2_target.forward(b_next_obs, next_action_t).to_numpy().item()
                    logp_val = float(next_logp.to_numpy().item())
                    target_q = min(q1_t, q2_t) - self.alpha * logp_val
                    y = float(b_reward) + (0.0 if b_done else self.gamma * target_q)
                    y_t = self.critic1.backend.tensor([y], requires_grad=False)

                    q1 = self.critic1.forward(b_obs, b_action)
                    q2 = self.critic2.forward(b_obs, b_action)
                    diff1 = q1 - y_t
                    diff2 = q2 - y_t
                    sample_loss1 = diff1 * diff1
                    sample_loss2 = diff2 * diff2
                    loss1 = sample_loss1 if loss1 is None else (loss1 + sample_loss1)
                    loss2 = sample_loss2 if loss2 is None else (loss2 + sample_loss2)

                if loss1 is not None:
                    self.critic1.backend.zero_grad(self.critic1.parameters())
                    loss1.backward()
                    self.critic1.backend.sgd_step(self.critic1.parameters(), self.critic_lr)
                if loss2 is not None:
                    self.critic2.backend.zero_grad(self.critic2.parameters())
                    loss2.backward()
                    self.critic2.backend.sgd_step(self.critic2.parameters(), self.critic_lr)

                actor_loss = None
                alpha_t = self.actor.backend.tensor([self.alpha], requires_grad=False)
                for b_obs, _, _, _, _ in batch:
                    _, logp, action_t = self.actor.sample(b_obs, action_limit=self.action_limit)
                    q_val = self.critic1.forward(b_obs, action_t)
                    sample_loss = logp * alpha_t - q_val
                    actor_loss = sample_loss if actor_loss is None else (actor_loss + sample_loss)
                if actor_loss is not None:
                    self.actor.backend.zero_grad(self.actor.parameters())
                    actor_loss.backward()
                    self.actor.backend.sgd_step(self.actor.parameters(), self.actor_lr)

                soft_update(self.critic1_target, self.critic1, self.tau)
                soft_update(self.critic2_target, self.critic2, self.tau)

            if step % 100 == 0:
                metrics.append({"step": step})
                self.logger.info("step=%s", step)
        return metrics
