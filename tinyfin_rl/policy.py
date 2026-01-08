from typing import Any

from .spaces import Space


class Policy:
    def __init__(self, action_space: Space):
        self.action_space = action_space

    def act(self, observation: Any):
        raise NotImplementedError


class RandomPolicy(Policy):
    def act(self, observation: Any):
        return self.action_space.sample()
