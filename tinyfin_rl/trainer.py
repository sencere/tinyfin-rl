from typing import Dict, List

from .agent import Agent
from .envs import Env
from .logging import setup_logging


class Trainer:
    def __init__(self, env: Env, agent: Agent, log_level: str = "INFO"):
        self.env = env
        self.agent = agent
        self.logger = setup_logging(level=log_level)

    def train(self, episodes: int = 1) -> List[Dict[str, float]]:
        metrics = []
        for episode in range(episodes):
            reset_out = self.env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            episode_return = 0.0
            steps = 0
            while not done:
                action = self.agent.act(obs)
                step_out = self.env.step(action)
                if len(step_out) == 5:
                    next_obs, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    next_obs, reward, done, _ = step_out
                self.agent.observe((obs, action, reward, next_obs, done))
                obs = next_obs
                episode_return += reward
                steps += 1
            record = {"episode": episode, "return": episode_return, "steps": steps}
            metrics.append(record)
            self.logger.info("episode=%s return=%.3f steps=%s", episode, episode_return, steps)
        return metrics
