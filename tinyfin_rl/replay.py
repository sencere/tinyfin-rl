import random
from collections import deque
from typing import Deque, List, Tuple


Transition = Tuple[object, object, float, object, bool]


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
