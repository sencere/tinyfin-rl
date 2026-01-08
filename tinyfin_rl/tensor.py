from typing import Any

try:
    import tinyfin as tf
except ImportError:  # tinyfin is in a separate repo; keep optional.
    tf = None


def is_available() -> bool:
    return tf is not None


def to_tensor(data: Any):
    if tf is None:
        return data
    return tf.tensor(data)


def from_tensor(tensor: Any):
    if tf is None:
        return tensor
    try:
        return tensor.numpy()
    except AttributeError:
        return tensor
