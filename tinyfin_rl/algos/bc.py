from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..logging import setup_logging
from ..policy_tinyfin import SoftmaxPolicy


@dataclass
class BCTrainer:
    policy: SoftmaxPolicy
    lr: float = 0.1
    epochs: int = 5
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.logger = setup_logging(level=self.log_level)

    def train(self, transitions: Iterable, obs_key: int = 0, action_key: int = 1) -> List[dict]:
        transitions = list(transitions)
        metrics = []
        for epoch in range(self.epochs):
            loss = None
            for t in transitions:
                obs = int(t[obs_key])
                action = int(t[action_key])
                logp = self.policy.log_prob(obs, action)
                sample_loss = logp * self.policy.backend.tensor([-1.0], requires_grad=False)
                loss = sample_loss if loss is None else (loss + sample_loss)
            if loss is not None:
                self.policy.backend.zero_grad(self.policy.parameters())
                loss.backward()
                self.policy.backend.sgd_step(self.policy.parameters(), self.lr)
            metrics.append({"epoch": epoch, "samples": len(transitions)})
            self.logger.info("epoch=%s samples=%s", epoch, len(transitions))
        return metrics
