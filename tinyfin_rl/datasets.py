import csv
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class Transition:
    obs: float
    action: float
    reward: float
    next_obs: float
    done: bool


@dataclass
class PreferencePair:
    obs_a: float
    obs_b: float
    label: int


def load_transition_csv(path: str) -> List[Transition]:
    rows: List[Transition] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"obs", "action", "reward", "next_obs", "done"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("missing required columns for transitions")
        for row in reader:
            rows.append(
                Transition(
                    obs=float(row["obs"]),
                    action=float(row["action"]),
                    reward=float(row["reward"]),
                    next_obs=float(row["next_obs"]),
                    done=row["done"].strip().lower() in {"1", "true", "yes"},
                )
            )
    return rows


def load_preference_csv(path: str) -> List[PreferencePair]:
    rows: List[PreferencePair] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"obs_a", "obs_b", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("missing required columns for preferences")
        for row in reader:
            rows.append(
                PreferencePair(
                    obs_a=float(row["obs_a"]),
                    obs_b=float(row["obs_b"]),
                    label=int(row["label"]),
                )
            )
    return rows


def iter_batch(data: Iterable, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch = []
    for item in data:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
