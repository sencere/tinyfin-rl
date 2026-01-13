import os
import sys
import pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.factory import make, emulate
from tinyfin_rl.vector_env import VectorEnv


def test_make_single():
    pytest.skip("Python BanditEnv removed; factory no longer creates Python envs.")


def test_make_vector():
    pytest.skip("Python GridworldEnv removed; factory no longer creates Python envs.")


def test_emulate_requires_factory_for_args():
    pytest.skip("Python BanditEnv removed; emulation of Python envs disabled.")
