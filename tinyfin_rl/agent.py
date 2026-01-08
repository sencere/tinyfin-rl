from typing import Any

from .policy import Policy
from .replay import ReplayBuffer


class Agent:
    def __init__(self, policy: Policy, replay_buffer: ReplayBuffer | None = None):
        self.policy = policy
        self.replay_buffer = replay_buffer

    def act(self, observation: Any):
        return self.policy.act(observation)

    def observe(self, transition):
        if self.replay_buffer is not None:
            self.replay_buffer.add(*transition)
