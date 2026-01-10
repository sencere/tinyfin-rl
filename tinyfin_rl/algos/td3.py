from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random

from ..envs import Env
from ..logging import setup_logging
from ..backends.tinyfin_backend import TinyfinBackend
from ..replay import ReplayBuffer


@dataclass
class Actor:
    backend: TinyfinBackend

    def __post_init__(self) -> None:
        self.w = self.backend.tensor([0.0], requires_grad=True)
        self.b = self.backend.tensor([0.0], requires_grad=True)

    def parameters(self):
        return [self.w, self.b]

    def forward(self, obs: float):
        obs_t = self.backend.tensor([float(obs)], requires_grad=False)
        return self.backend.add(self.backend.mul(obs_t, self.w), self.b)

    def act(self, obs: float, noise_std: float = 0.1, action_limit: float = 1.0) -> float:
        action = float(self.forward(obs).to_numpy().item())
        if noise_std > 0:
            action += random.gauss(0.0, noise_std)
        return max(-action_limit, min(action_limit, action))

    def copy_from(self, other: "Actor") -> None:
        self.w.numpy_view()[:] = other.w.numpy_view()
        self.b.numpy_view()[:] = other.b.numpy_view()


@dataclass
class Critic:
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

    def copy_from(self, other: "Critic") -> None:
        self.w_obs.numpy_view()[:] = other.w_obs.numpy_view()
        self.w_act.numpy_view()[:] = other.w_act.numpy_view()
        self.b.numpy_view()[:] = other.b.numpy_view()


def soft_update(target, source, tau: float = 0.005) -> None:
    if tau <= 0 or tau > 1:
        raise ValueError("tau must be in (0, 1]")
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.numpy_view()[:] = (1.0 - tau) * t_param.numpy_view() + tau * s_param.numpy_view()


@dataclass
class TD3Trainer:
    env: Env
    actor: Actor
    critic1: Critic
    critic2: Critic
    actor_target: Actor
    critic1_target: Critic
    critic2_target: Critic
    replay: ReplayBuffer
    gamma: float = 0.99
    actor_lr: float = 0.05
    critic_lr: float = 0.1
    batch_size: int = 32
    policy_delay: int = 2
    action_limit: float = 1.0
    noise_std: float = 0.1
    target_noise: float = 0.2
    target_noise_clip: float = 0.5
    tau: float = 0.005
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.logger = setup_logging(level=self.log_level)
        self.actor_target.copy_from(self.actor)
        self.critic1_target.copy_from(self.critic1)
        self.critic2_target.copy_from(self.critic2)

    def train(self, steps: int = 500) -> List[dict]:
        metrics = []
        obs = self.env.reset()
        for step in range(steps):
            action = self.actor.act(obs, noise_std=self.noise_std, action_limit=self.action_limit)
            next_obs, reward, done, _ = self.env.step(action)
            self.replay.add(obs, action, float(reward), next_obs, bool(done))
            obs = self.env.reset() if done else next_obs

            if len(self.replay) >= self.batch_size:
                batch = self.replay.sample(self.batch_size)
                loss1 = None
                loss2 = None
                for b_obs, b_action, b_reward, b_next_obs, b_done in batch:
                    target_action = self.actor_target.act(b_next_obs, noise_std=0.0, action_limit=self.action_limit)
                    target_action += random.gauss(0.0, self.target_noise)
                    target_action = max(-self.action_limit, min(self.action_limit, target_action))
                    q1_t = self.critic1_target.forward(b_next_obs, target_action).to_numpy().item()
                    q2_t = self.critic2_target.forward(b_next_obs, target_action).to_numpy().item()
                    target_q = min(q1_t, q2_t)
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

                if step % self.policy_delay == 0:
                    actor_loss = None
                    for b_obs, _, _, _, _ in batch:
                        act = self.actor.forward(b_obs)
                        q_val = self.critic1.forward(b_obs, act)
                        sample_loss = q_val * self.critic1.backend.tensor([-1.0], requires_grad=False)
                        actor_loss = sample_loss if actor_loss is None else (actor_loss + sample_loss)
                    if actor_loss is not None:
                        self.actor.backend.zero_grad(self.actor.parameters())
                        actor_loss.backward()
                        self.actor.backend.sgd_step(self.actor.parameters(), self.actor_lr)
                    soft_update(self.actor_target, self.actor, self.tau)
                    soft_update(self.critic1_target, self.critic1, self.tau)
                    soft_update(self.critic2_target, self.critic2, self.tau)

            if step % 100 == 0:
                metrics.append({"step": step})
                self.logger.info("step=%s", step)
        return metrics
