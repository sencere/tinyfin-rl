import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class ExperimentLogger:
    path: Path | str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            self.path = Path(self.path)

    def log(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")


def run_experiment(trainer, output_path: str, **train_kwargs) -> List[dict]:
    metrics = trainer.train(**train_kwargs)
    logger = ExperimentLogger(Path(output_path))
    for record in metrics:
        logger.log(record)
    return metrics
