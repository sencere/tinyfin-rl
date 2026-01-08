import os
import random
from typing import Optional


def set_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = 0
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    return seed
