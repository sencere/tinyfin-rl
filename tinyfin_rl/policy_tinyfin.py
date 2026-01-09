from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .policy import Policy
from .spaces import Discrete
from .backends.tinyfin_backend import TinyfinBackend


@dataclass
class SoftmaxPolicy(Policy):
    backend: TinyfinBackend
    obs_dim: int
    action_space: Discrete

    def __post_init__(self) -> None:
        super().__init__(self.action_space)
        w = np.random.randn(self.obs_dim, self.action_space.n).astype(np.float32) * 0.01
        b = np.zeros((self.action_space.n,), dtype=np.float32)
        self.w = self.backend.tensor(w, requires_grad=True)
        self.b = self.backend.tensor(b, requires_grad=True)

    def parameters(self):
        return [self.w, self.b]

    def logits(self, obs_index: int):
        x = self.backend.one_hot(obs_index, self.obs_dim).reshape([1, self.obs_dim])
        return self.backend.add(self.backend.matmul(x, self.w), self.b)

    def act(self, observation: int):
        logits = self.logits(int(observation))
        probs = self.backend.softmax(logits)
        p = probs.to_numpy().reshape(-1)
        return int(np.random.choice(len(p), p=p))

    def log_prob(self, observation: int, action: int):
        logits = self.logits(int(observation))
        logp = self.backend.log(self.backend.softmax(logits))
        a_one_hot = self.backend.one_hot(action, self.action_space.n).reshape([1, self.action_space.n])
        return self.backend.sum(self.backend.mul(logp, a_one_hot))


@dataclass
class PPOPolicy(Policy):
    backend: TinyfinBackend
    obs_dim: int
    action_space: Discrete

    def __post_init__(self) -> None:
        super().__init__(self.action_space)
        w = np.random.randn(self.obs_dim, self.action_space.n).astype(np.float32) * 0.01
        b = np.zeros((self.action_space.n,), dtype=np.float32)
        v = np.random.randn(self.obs_dim, 1).astype(np.float32) * 0.01
        c = np.zeros((1,), dtype=np.float32)
        self.w = self.backend.tensor(w, requires_grad=True)
        self.b = self.backend.tensor(b, requires_grad=True)
        self.v = self.backend.tensor(v, requires_grad=True)
        self.c = self.backend.tensor(c, requires_grad=True)

    def parameters(self):
        return [self.w, self.b, self.v, self.c]

    def logits(self, obs_index: int):
        x = self.backend.one_hot(obs_index, self.obs_dim).reshape([1, self.obs_dim])
        return self.backend.add(self.backend.matmul(x, self.w), self.b)

    def value(self, obs_index: int):
        x = self.backend.one_hot(obs_index, self.obs_dim).reshape([1, self.obs_dim])
        val = self.backend.add(self.backend.matmul(x, self.v), self.c)
        return self.backend.sum(val)

    def act(self, observation: int):
        logits = self.logits(int(observation))
        probs = self.backend.softmax(logits)
        p = probs.to_numpy().reshape(-1)
        return int(np.random.choice(len(p), p=p))

    def log_prob(self, observation: int, action: int):
        logits = self.logits(int(observation))
        logp = self.backend.log(self.backend.softmax(logits))
        a_one_hot = self.backend.one_hot(action, self.action_space.n).reshape([1, self.action_space.n])
        return self.backend.sum(self.backend.mul(logp, a_one_hot))
