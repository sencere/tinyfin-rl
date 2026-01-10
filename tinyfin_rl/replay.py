import random
from collections import deque
from typing import Deque, List, Tuple, Optional


Transition = Tuple[object, object, float, object, bool]
Sequence = List[Transition]


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, obs, action, reward: float, next_obs, done: bool) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self.buffer):
            raise ValueError("batch_size larger than buffer")
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, eps: float = 1e-5):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer: List[Transition] = []
        self.priorities: List[float] = []

    def add(self, obs, action, reward: float, next_obs, done: bool, priority: Optional[float] = None) -> None:
        if priority is None:
            priority = max(self.priorities, default=1.0)
        priority = float(priority) + self.eps
        if len(self.buffer) < self.capacity:
            self.buffer.append((obs, action, reward, next_obs, done))
            self.priorities.append(priority)
        else:
            idx = len(self.buffer) % self.capacity
            self.buffer[idx] = (obs, action, reward, next_obs, done)
            self.priorities[idx] = priority

    def sample(self, batch_size: int, beta: float = 0.4):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self.buffer):
            raise ValueError("batch_size larger than buffer")
        if beta <= 0:
            raise ValueError("beta must be positive")
        scaled = [p ** self.alpha for p in self.priorities]
        total = sum(scaled)
        probs = [p / total for p in scaled]
        indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)
        samples = [self.buffer[i] for i in indices]
        weights = [(len(self.buffer) * probs[i]) ** (-beta) for i in indices]
        max_w = max(weights)
        weights = [w / max_w for w in weights]
        return samples, indices, weights

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        if len(indices) != len(priorities):
            raise ValueError("indices and priorities length mismatch")
        for idx, priority in zip(indices, priorities):
            if idx < 0 or idx >= len(self.priorities):
                raise ValueError("priority index out of range")
            self.priorities[idx] = float(priority) + self.eps

    def __len__(self) -> int:
        return len(self.buffer)


class NStepReplayBuffer:
    def __init__(self, capacity: int = 10000, n: int = 3, gamma: float = 0.99):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if n <= 0:
            raise ValueError("n must be positive")
        self.capacity = capacity
        self.n = n
        self.gamma = gamma
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.n_step: Deque[Transition] = deque(maxlen=n)

    def _flush(self) -> None:
        while self.n_step:
            obs, action, _, _, _ = self.n_step[0]
            reward_sum = 0.0
            done = False
            next_obs = self.n_step[-1][3]
            for i, (_, _, r, nobs, d) in enumerate(self.n_step):
                reward_sum += (self.gamma ** i) * r
                next_obs = nobs
                done = d
                if d:
                    break
            self.buffer.append((obs, action, reward_sum, next_obs, done))
            self.n_step.popleft()

    def add(self, obs, action, reward: float, next_obs, done: bool) -> None:
        self.n_step.append((obs, action, reward, next_obs, done))
        if len(self.n_step) >= self.n:
            obs0, action0, _, _, _ = self.n_step[0]
            reward_sum = 0.0
            next_obs_n = self.n_step[-1][3]
            done_n = False
            for i, (_, _, r, nobs, d) in enumerate(self.n_step):
                reward_sum += (self.gamma ** i) * r
                next_obs_n = nobs
                done_n = d
                if d:
                    break
            self.buffer.append((obs0, action0, reward_sum, next_obs_n, done_n))
        if done:
            self._flush()

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self.buffer):
            raise ValueError("batch_size larger than buffer")
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class SequenceReplayBuffer:
    def __init__(self, capacity: int = 1000, sequence_length: int = 4):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer: Deque[Sequence] = deque(maxlen=capacity)
        self.window: Deque[Transition] = deque(maxlen=sequence_length)

    def add(self, obs, action, reward: float, next_obs, done: bool) -> None:
        self.window.append((obs, action, reward, next_obs, done))
        if len(self.window) == self.sequence_length:
            self.buffer.append(list(self.window))
        if done:
            self.window.clear()

    def sample(self, batch_size: int) -> List[Sequence]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self.buffer):
            raise ValueError("batch_size larger than buffer")
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
