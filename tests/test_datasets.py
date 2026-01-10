import os
import sys
import tempfile

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.datasets import load_transition_csv, load_preference_csv, iter_batch


def _write_tmp(contents: str) -> str:
    fd, path = tempfile.mkstemp(text=True)
    with os.fdopen(fd, "w") as handle:
        handle.write(contents)
    return path


def test_load_transition_csv():
    path = _write_tmp("obs,action,reward,next_obs,done\n0,1,1.0,2,True\n")
    rows = load_transition_csv(path)
    assert len(rows) == 1
    assert rows[0].obs == 0.0
    assert rows[0].done is True


def test_load_preference_csv():
    path = _write_tmp("obs_a,obs_b,label\n0.0,1.0,1\n")
    rows = load_preference_csv(path)
    assert len(rows) == 1
    assert rows[0].label == 1


def test_iter_batch():
    data = [1, 2, 3, 4, 5]
    batches = list(iter_batch(data, batch_size=2))
    assert batches == [[1, 2], [3, 4], [5]]
