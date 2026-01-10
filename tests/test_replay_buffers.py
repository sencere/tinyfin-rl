import os
import sys

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.replay import ReplayBuffer, PrioritizedReplayBuffer, NStepReplayBuffer, SequenceReplayBuffer


def test_prioritized_buffer_sampling():
    buf = PrioritizedReplayBuffer(capacity=10, alpha=0.6)
    for i in range(5):
        buf.add(i, 0, 1.0, i + 1, False, priority=i + 1)
    samples, indices, weights = buf.sample(3, beta=0.4)
    assert len(samples) == 3
    assert len(indices) == 3
    assert len(weights) == 3


def test_n_step_buffer_aggregates():
    buf = NStepReplayBuffer(capacity=10, n=3, gamma=0.9)
    buf.add(0, 0, 1.0, 1, False)
    buf.add(1, 0, 1.0, 2, False)
    buf.add(2, 0, 1.0, 3, False)
    assert len(buf) == 1
    sample = buf.sample(1)[0]
    assert abs(sample[2] - (1.0 + 0.9 + 0.9 * 0.9)) < 1e-6


def test_n_step_flush_on_done():
    buf = NStepReplayBuffer(capacity=10, n=3, gamma=0.9)
    buf.add(0, 0, 1.0, 1, False)
    buf.add(1, 0, 1.0, 2, True)
    assert len(buf) >= 1


def test_sequence_buffer_shapes():
    buf = SequenceReplayBuffer(capacity=5, sequence_length=3)
    for i in range(3):
        buf.add(i, 0, 1.0, i + 1, False)
    assert len(buf) == 1
    seq = buf.sample(1)[0]
    assert len(seq) == 3


def test_replay_buffer_sample_size():
    buf = ReplayBuffer(capacity=5)
    for i in range(5):
        buf.add(i, 0, 1.0, i + 1, False)
    batch = buf.sample(3)
    assert len(batch) == 3
