from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple


Spec = Tuple[str, Any]


def infer_spec(value: Any) -> Spec:
    if isinstance(value, list):
        return ("list", [infer_spec(v) for v in value])
    if isinstance(value, tuple):
        return ("tuple", [infer_spec(v) for v in value])
    return ("leaf", None)


def flatten(value: Any, spec: Spec | None = None) -> Tuple[List[float], Spec]:
    if spec is None:
        spec = infer_spec(value)
    tag, data = spec
    if tag == "leaf":
        return [float(value)], spec
    flat: List[float] = []
    if tag == "list":
        for v, child in zip(value, data):
            sub, _ = flatten(v, child)
            flat.extend(sub)
    elif tag == "tuple":
        for v, child in zip(value, data):
            sub, _ = flatten(v, child)
            flat.extend(sub)
    else:
        raise ValueError("unknown spec tag")
    return flat, spec


def unflatten(values: Iterable[float], spec: Spec) -> Tuple[Any, List[float]]:
    values = list(values)
    tag, data = spec
    if tag == "leaf":
        return values[0], values[1:]
    items = []
    rest = values
    for child in data:
        item, rest = unflatten(rest, child)
        items.append(item)
    if tag == "list":
        return items, rest
    if tag == "tuple":
        return tuple(items), rest
    raise ValueError("unknown spec tag")


@dataclass
class EmulationSpec:
    obs_spec: Spec
    action_spec: Spec

    def flatten_obs(self, obs: Any) -> List[float]:
        flat, _ = flatten(obs, self.obs_spec)
        return flat

    def flatten_action(self, action: Any) -> List[float]:
        flat, _ = flatten(action, self.action_spec)
        return flat

    def unflatten_obs(self, values: Iterable[float]) -> Any:
        obs, _ = unflatten(values, self.obs_spec)
        return obs

    def unflatten_action(self, values: Iterable[float]) -> Any:
        action, _ = unflatten(values, self.action_spec)
        return action
