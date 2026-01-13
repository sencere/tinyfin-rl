import random

from .spaces import Space


class Env:
    observation_space: Space
    action_space: Space

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def seed(self, seed: int) -> int:
        random.seed(seed)
        return seed
