import os
import sys
import tempfile

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.experiment import ExperimentLogger


def test_experiment_logger_writes():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "metrics.jsonl")
        logger = ExperimentLogger(path=path)
        logger.log({"step": 1})
        with open(path, "r") as handle:
            lines = handle.readlines()
    assert len(lines) == 1
