"""Tinyfin-RL: lightweight RL primitives built around Tinyfin tensors."""

from .config import Config
from .envs import Env, BanditEnv, GridworldEnv, ContinuousBanditEnv
from .policy import Policy, RandomPolicy
from .policy_tinyfin import SoftmaxPolicy, PPOPolicy
from .agent import Agent
from .replay import ReplayBuffer
from .trainer import Trainer
from .algos import ReinforceTrainer, PPOTrainer, A2CTrainer, DQNTrainer, DuelingQNetwork, BCTrainer, TRPOTrainer, SharedPolicyTrainer, TD3Trainer, Actor, Critic
from .vector_env import VectorEnv, AsyncVectorEnv
from .rollout import RolloutBatch, rollout_vector, rollout_vector_prealloc, rollout_vector_zero_copy
from .async_rollout import AsyncRolloutRunner
from .spaces import Space, Discrete, Box
from .env_wrappers import ActionRepeat, FrameStack, NormalizeObservation, NormalizeReward, TimeLimit
from .vector_wrappers import ActionRepeatVec, NormalizeObservationVec, NormalizeRewardVec, TimeLimitVec
from .seed import set_seed
from .backend_c import CBackend
from .backends import TinyfinBackend
from .datasets import Transition, PreferencePair, load_transition_csv, load_preference_csv, iter_batch
from .multi_agent import SharedPolicyMultiAgent, IndependentMultiAgent
from .emulation import EmulationSpec, flatten, unflatten, infer_spec
from .factory import make, emulate
from .checkpoint import save_policy_checkpoint, load_policy_checkpoint, save_q_checkpoint, load_q_checkpoint
from .eval import evaluate_policy
from .experiment import ExperimentLogger, run_experiment
from .visualize import Visualizer, RaylibGridworldVisualizer

__all__ = [
    "Config",
    "Env",
    "BanditEnv",
    "GridworldEnv",
    "ContinuousBanditEnv",
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
    "TD3Trainer",
    "Actor",
    "Critic",
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
    "EmulationSpec",
    "flatten",
    "unflatten",
    "infer_spec",
    "make",
    "emulate",
    "save_policy_checkpoint",
    "load_policy_checkpoint",
    "save_q_checkpoint",
    "load_q_checkpoint",
    "evaluate_policy",
    "ExperimentLogger",
    "run_experiment",
    "VectorEnv",
    "AsyncVectorEnv",
    "RolloutBatch",
    "rollout_vector",
    "rollout_vector_prealloc",
    "rollout_vector_zero_copy",
    "AsyncRolloutRunner",
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
