import random
from typing import Dict, Tuple

from .spaces import Discrete, Space


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


class BanditEnv(Env):
    def __init__(self, arm_probs, reward_scale: float = 1.0):
        if not arm_probs:
            raise ValueError("arm_probs cannot be empty")
        self.arm_probs = list(arm_probs)
        self.reward_scale = reward_scale
        self.action_space = Discrete(len(self.arm_probs))
        self.observation_space = Discrete(1)

    def reset(self) -> int:
        return 0

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        if not self.action_space.contains(action):
            raise ValueError("invalid action")
        reward = self.reward_scale if random.random() < self.arm_probs[action] else 0.0
        return 0, reward, True, {}


class GridworldEnv(Env):
    def __init__(self, width: int = 5, height: int = 5, max_steps: int = 50):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.action_space = Discrete(4)
        self.observation_space = Discrete(width * height)
        self.reset()

    def reset(self) -> int:
        self.agent_pos = [0, 0]
        self.goal_pos = [self.width - 1, self.height - 1]
        self.steps = 0
        return self._obs()

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        if not self.action_space.contains(action):
            raise ValueError("invalid action")
        self.steps += 1
        if action == 0:
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:
            self.agent_pos[1] = min(self.height - 1, self.agent_pos[1] + 1)
        elif action == 2:
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        else:
            self.agent_pos[0] = min(self.width - 1, self.agent_pos[0] + 1)

        done = self.agent_pos == self.goal_pos or self.steps >= self.max_steps
        reward = 1.0 if self.agent_pos == self.goal_pos else -0.01
        return self._obs(), reward, done, {}

    def _obs(self) -> int:
        return self.agent_pos[1] * self.width + self.agent_pos[0]


class ContinuousBanditEnv(Env):
    def __init__(self, target: float = 0.7, reward_scale: float = 1.0):
        self.target = target
        self.reward_scale = reward_scale
        self.action_space = Box(-1.0, 1.0, (1,))
        self.observation_space = Box(0.0, 1.0, (1,))

    def reset(self) -> float:
        return 0.0

    def step(self, action: float):
        try:
            act = float(action)
        except (TypeError, ValueError):
            raise ValueError("invalid action")
        if act < self.action_space.low or act > self.action_space.high:
            raise ValueError("action out of bounds")
        reward = self.reward_scale * (1.0 - abs(act - self.target))
        return 0.0, reward, True, {}
