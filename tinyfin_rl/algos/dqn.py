from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..envs import Env
from ..logging import setup_logging
from ..spaces import Discrete
from ..backends.tinyfin_backend import TinyfinBackend
from ..replay import ReplayBuffer


@dataclass
class QNetwork:
    backend: TinyfinBackend
    obs_dim: int
    action_space: Discrete

    def __post_init__(self) -> None:
        w = self.backend.tensor([[0.0] * self.action_space.n for _ in range(self.obs_dim)], requires_grad=True)
        b = self.backend.tensor([0.0] * self.action_space.n, requires_grad=True)
        self.w = w
        self.b = b

    def parameters(self):
        return [self.w, self.b]

    def q_values(self, obs_index: int):
        x = self.backend.one_hot(obs_index, self.obs_dim).reshape([1, self.obs_dim])
        return self.backend.add(self.backend.matmul(x, self.w), self.b)

    def act(self, observation: int, eps: float = 0.1) -> int:
        import random
        if random.random() < eps:
            return self.action_space.sample()
        q = self.q_values(int(observation)).to_numpy().reshape(-1)
        return int(q.argmax())

    def copy_from(self, other: "QNetwork") -> None:
        self.w.numpy_view()[:] = other.w.numpy_view()
        self.b.numpy_view()[:] = other.b.numpy_view()


def hard_update(target: QNetwork, source: QNetwork) -> None:
    target.copy_from(source)


def soft_update(target: QNetwork, source: QNetwork, tau: float = 0.005) -> None:
    if tau <= 0 or tau > 1:
        raise ValueError("tau must be in (0, 1]")
    target.w.numpy_view()[:] = (1.0 - tau) * target.w.numpy_view() + tau * source.w.numpy_view()
    target.b.numpy_view()[:] = (1.0 - tau) * target.b.numpy_view() + tau * source.b.numpy_view()


@dataclass
class DQNTrainer:
    env: Env
    q_net: QNetwork
    target_net: QNetwork
    replay: ReplayBuffer
    gamma: float = 0.99
    lr: float = 0.1
    batch_size: int = 32
    eps_start: float = 0.5
    eps_end: float = 0.05
    eps_decay_steps: int = 500
    target_update_steps: int = 100
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.logger = setup_logging(level=self.log_level)
        hard_update(self.target_net, self.q_net)

    def _eps_at(self, step: int) -> float:
        if step >= self.eps_decay_steps:
            return self.eps_end
        delta = (self.eps_start - self.eps_end) * (1.0 - step / self.eps_decay_steps)
        return self.eps_end + delta

    def train(self, steps: int = 500) -> List[dict]:
        metrics = []
        obs = self.env.reset()
        for step in range(steps):
            eps = self._eps_at(step)
            action = self.q_net.act(int(obs), eps=eps)
            next_obs, reward, done, _ = self.env.step(action)
            self.replay.add(obs, action, float(reward), next_obs, bool(done))
            obs = self.env.reset() if done else next_obs

            if len(self.replay) >= self.batch_size:
                batch = self.replay.sample(self.batch_size)
                loss = None
                for b_obs, b_action, b_reward, b_next_obs, b_done in batch:
                    q_vals = self.q_net.q_values(int(b_obs))
                    a_one_hot = self.q_net.backend.one_hot(int(b_action), self.q_net.action_space.n).reshape([1, self.q_net.action_space.n])
                    q_pred = self.q_net.backend.sum(self.q_net.backend.mul(q_vals, a_one_hot))
                    next_q = self.target_net.q_values(int(b_next_obs)).to_numpy().reshape(-1)
                    max_next = float(next_q.max())
                    target = float(b_reward) + (0.0 if b_done else self.gamma * max_next)
                    target_t = self.q_net.backend.tensor([target], requires_grad=False)
                    diff = q_pred - target_t
                    sample_loss = diff * diff
                    loss = sample_loss if loss is None else (loss + sample_loss)

                if loss is not None:
                    self.q_net.backend.zero_grad(self.q_net.parameters())
                    loss.backward()
                    self.q_net.backend.sgd_step(self.q_net.parameters(), self.lr)

            if step > 0 and step % self.target_update_steps == 0:
                hard_update(self.target_net, self.q_net)

            if step % 100 == 0:
                metrics.append({"step": step, "eps": eps})
                self.logger.info("step=%s eps=%.3f", step, eps)
        return metrics


def make_dqn_bandit(backend: TinyfinBackend, action_space: Discrete) -> tuple[QNetwork, QNetwork]:
    q_net = QNetwork(backend=backend, obs_dim=1, action_space=action_space)
    target_net = QNetwork(backend=backend, obs_dim=1, action_space=action_space)
    hard_update(target_net, q_net)
    return q_net, target_net
