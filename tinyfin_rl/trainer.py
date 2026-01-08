from typing import Dict, List, Optional

from .agent import Agent
from .envs import Env
from .logging import setup_logging
from .visualize import Visualizer


class Trainer:
    def __init__(self, env: Env, agent: Agent, log_level: str = "INFO", visualizer: Optional[Visualizer] = None):
        self.env = env
        self.agent = agent
        self.logger = setup_logging(level=log_level)
        self.visualizer = visualizer

    def train(self, episodes: int = 1) -> List[Dict[str, float]]:
        metrics = []
        for episode in range(episodes):
            reset_out = self.env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            if self.visualizer is not None:
                self.visualizer.on_reset(self.env, obs)
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
                if self.visualizer is not None:
                    self.visualizer.on_step(self.env, obs, action, reward, done)
            record = {"episode": episode, "return": episode_return, "steps": steps}
            metrics.append(record)
            self.logger.info("episode=%s return=%.3f steps=%s", episode, episode_return, steps)
        if self.visualizer is not None:
            self.visualizer.close()
        return metrics
