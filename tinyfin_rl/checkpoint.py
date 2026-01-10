import json
from typing import Dict, Optional

import numpy as np

from .policy_tinyfin import PPOPolicy, SoftmaxPolicy
from .algos.dqn import QNetwork, DuelingQNetwork


def _save_npz(path: str, arrays: Dict[str, np.ndarray], meta: Optional[dict] = None) -> None:
    payload = dict(arrays)
    if meta is not None:
        payload["__meta__"] = np.array(json.dumps(meta))
    np.savez(path, **payload)


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_policy_checkpoint(path: str, policy, meta: Optional[dict] = None) -> None:
    arrays: Dict[str, np.ndarray] = {}
    if isinstance(policy, SoftmaxPolicy):
        arrays = {
            "w": policy.w.to_numpy(),
            "b": policy.b.to_numpy(),
            "type": np.array("softmax"),
        }
    elif isinstance(policy, PPOPolicy):
        arrays = {
            "w": policy.w.to_numpy(),
            "b": policy.b.to_numpy(),
            "v": policy.v.to_numpy(),
            "c": policy.c.to_numpy(),
            "type": np.array("ppo"),
        }
    else:
        raise ValueError("unsupported policy type")
    _save_npz(path, arrays, meta=meta)


def load_policy_checkpoint(path: str, policy) -> None:
    data = _load_npz(path)
    if isinstance(policy, SoftmaxPolicy):
        policy.w.numpy_view()[:] = data["w"]
        policy.b.numpy_view()[:] = data["b"]
    elif isinstance(policy, PPOPolicy):
        policy.w.numpy_view()[:] = data["w"]
        policy.b.numpy_view()[:] = data["b"]
        policy.v.numpy_view()[:] = data["v"]
        policy.c.numpy_view()[:] = data["c"]
    else:
        raise ValueError("unsupported policy type")


def save_q_checkpoint(path: str, q_net, target_net=None, meta: Optional[dict] = None) -> None:
    arrays: Dict[str, np.ndarray] = {}
    if isinstance(q_net, DuelingQNetwork):
        arrays.update(
            {
                "q_w": q_net.w.to_numpy(),
                "q_b": q_net.b.to_numpy(),
                "q_v_w": q_net.v_w.to_numpy(),
                "q_v_b": q_net.v_b.to_numpy(),
                "type": np.array("dueling"),
            }
        )
    elif isinstance(q_net, QNetwork):
        arrays.update(
            {
                "q_w": q_net.w.to_numpy(),
                "q_b": q_net.b.to_numpy(),
                "type": np.array("q"),
            }
        )
    else:
        raise ValueError("unsupported q network type")

    if target_net is not None:
        if isinstance(target_net, DuelingQNetwork):
            arrays.update(
                {
                    "t_w": target_net.w.to_numpy(),
                    "t_b": target_net.b.to_numpy(),
                    "t_v_w": target_net.v_w.to_numpy(),
                    "t_v_b": target_net.v_b.to_numpy(),
                }
            )
        elif isinstance(target_net, QNetwork):
            arrays.update(
                {
                    "t_w": target_net.w.to_numpy(),
                    "t_b": target_net.b.to_numpy(),
                }
            )
    _save_npz(path, arrays, meta=meta)


def load_q_checkpoint(path: str, q_net, target_net=None) -> None:
    data = _load_npz(path)
    if isinstance(q_net, DuelingQNetwork):
        q_net.w.numpy_view()[:] = data["q_w"]
        q_net.b.numpy_view()[:] = data["q_b"]
        q_net.v_w.numpy_view()[:] = data["q_v_w"]
        q_net.v_b.numpy_view()[:] = data["q_v_b"]
    elif isinstance(q_net, QNetwork):
        q_net.w.numpy_view()[:] = data["q_w"]
        q_net.b.numpy_view()[:] = data["q_b"]
    else:
        raise ValueError("unsupported q network type")

    if target_net is None:
        return
    if isinstance(target_net, DuelingQNetwork):
        target_net.w.numpy_view()[:] = data["t_w"]
        target_net.b.numpy_view()[:] = data["t_b"]
        target_net.v_w.numpy_view()[:] = data["t_v_w"]
        target_net.v_b.numpy_view()[:] = data["t_v_b"]
    elif isinstance(target_net, QNetwork):
        target_net.w.numpy_view()[:] = data["t_w"]
        target_net.b.numpy_view()[:] = data["t_b"]
