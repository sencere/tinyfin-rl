"""Tinyfin-RL: lightweight RL primitives built around Tinyfin tensors."""

from .config import Config
from .envs import Env, BanditEnv, GridworldEnv
from .policy import Policy, RandomPolicy
from .policy_tinyfin import SoftmaxPolicy, PPOPolicy
from .agent import Agent
from .replay import ReplayBuffer
from .trainer import Trainer
from .algos import ReinforceTrainer, PPOTrainer, A2CTrainer, DQNTrainer, DuelingQNetwork, BCTrainer, TRPOTrainer, SharedPolicyTrainer
from .vector_env import VectorEnv, AsyncVectorEnv
from .rollout import RolloutBatch, rollout_vector
from .spaces import Space, Discrete, Box
from .env_wrappers import ActionRepeat, FrameStack, NormalizeObservation, NormalizeReward, TimeLimit
from .vector_wrappers import ActionRepeatVec, NormalizeObservationVec, NormalizeRewardVec, TimeLimitVec
from .seed import set_seed
from .backend_c import CBackend
from .backends import TinyfinBackend
from .datasets import Transition, PreferencePair, load_transition_csv, load_preference_csv, iter_batch
from .multi_agent import SharedPolicyMultiAgent, IndependentMultiAgent
from .visualize import Visualizer, RaylibGridworldVisualizer

__all__ = [
    "Config",
    "Env",
    "BanditEnv",
    "GridworldEnv",
    "Policy",
    "RandomPolicy",
    "SoftmaxPolicy",
    "PPOPolicy",
    "Agent",
    "ReplayBuffer",
    "Trainer",
    "ReinforceTrainer",
    "PPOTrainer",
    "A2CTrainer",
    "DQNTrainer",
    "DuelingQNetwork",
    "BCTrainer",
    "TRPOTrainer",
    "SharedPolicyTrainer",
    "Space",
    "Discrete",
    "Box",
    "set_seed",
    "CBackend",
    "TinyfinBackend",
    "Transition",
    "PreferencePair",
    "load_transition_csv",
    "load_preference_csv",
    "iter_batch",
    "SharedPolicyMultiAgent",
    "IndependentMultiAgent",
    "VectorEnv",
    "AsyncVectorEnv",
    "RolloutBatch",
    "rollout_vector",
    "ActionRepeat",
    "FrameStack",
    "NormalizeObservation",
    "NormalizeReward",
    "TimeLimit",
    "TimeLimitVec",
    "ActionRepeatVec",
    "NormalizeObservationVec",
    "NormalizeRewardVec",
    "Visualizer",
    "RaylibGridworldVisualizer",
]
