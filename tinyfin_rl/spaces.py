import random
from typing import Iterable, Tuple


class Space:
    def sample(self):
        raise NotImplementedError

    def contains(self, x) -> bool:
        raise NotImplementedError


class Discrete(Space):
    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n

    def sample(self) -> int:
        return random.randrange(self.n)

    def contains(self, x) -> bool:
        return isinstance(x, int) and 0 <= x < self.n

    def __repr__(self) -> str:
        return f"Discrete(n={self.n})"


class Box(Space):
    def __init__(self, low: float, high: float, shape: Tuple[int, ...]):
        if low >= high:
            raise ValueError("low must be < high")
        self.low = low
        self.high = high
        self.shape = shape

    def sample(self) -> Iterable[float]:
        size = 1
        for dim in self.shape:
            size *= dim
        return [random.uniform(self.low, self.high) for _ in range(size)]

    def contains(self, x) -> bool:
        try:
            values = list(x)
        except TypeError:
            return False
        if len(values) != self._flat_size():
            return False
        return all(self.low <= v <= self.high for v in values)

    def _flat_size(self) -> int:
        size = 1
        for dim in self.shape:
            size *= dim
        return size

    def __repr__(self) -> str:
        return f"Box(low={self.low}, high={self.high}, shape={self.shape})"
