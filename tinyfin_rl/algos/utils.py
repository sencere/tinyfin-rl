from __future__ import annotations

from typing import Iterable, List

import math


def normalize_advantages(advantages: Iterable[float], eps: float = 1e-8) -> List[float]:
    values = [float(a) for a in advantages]
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((a - mean) ** 2 for a in values) / len(values)
    std = math.sqrt(var) + eps
    return [(a - mean) / std for a in values]


def policy_entropy(probs: Iterable[float], eps: float = 1e-8) -> float:
    total = 0.0
    for p in probs:
        p = float(p)
        if p <= 0.0:
            continue
        total -= p * math.log(max(p, eps))
    return total


def value_loss(pred: float, target: float, coef: float = 0.5) -> float:
    diff = float(pred) - float(target)
    return float(coef) * diff * diff
