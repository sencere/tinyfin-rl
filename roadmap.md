# tinyfin-rl Roadmap (Concise)

**Goal:** A lightweight RL library built on Tinyfin tensors: clean API, fast simulation, and strong algorithm coverage.

## Milestone 0 — Foundations
- Project skeleton, config system, logging, and reproducible seeding.
- Core RL API: `Env`, `Space`, `Agent`, `Policy`, `ReplayBuffer`, `Trainer`.
- Tensor/NN adapters: policy/value heads, distribution helpers, action sampling.
- Baseline examples: bandits, CartPole, simple gridworld.

## Milestone 1 — On-Policy Essentials
- REINFORCE, VPG (vanilla policy gradient), A2C.
- PPO (clip + KL variants), GAE.
- Common utilities: advantage normalization, entropy bonus, value loss, grad clipping.
- Reference configs + unit tests for returns/GAE/advantage math.

## Milestone 2 — Off-Policy Essentials
- DQN family: DQN, Double DQN, Dueling, Prioritized Replay.
- Continuous control: DDPG, TD3, SAC (v1/v2 notes).
- Replay buffers: uniform, prioritized, n-step, sequence replay.
- Target networks, Polyak averaging, exploration noise helpers.

## Milestone 3 — Advanced & Specialized
- Actor-critic variants: TRPO, ACKTR (optional).
- Offline RL: behavior cloning, CQL/BCQ (minimal baseline).
- Imitation + RLHF hooks: dataset loaders, preference model interface.
- Multi-agent primitives: shared policy, independent learners, centralized critic hooks.

## Milestone 4 — Simulation & Env Tooling
- Env wrappers: time-limit, frame stack, normalization, action repeat.
- Vectorized envs: sync/async, multi-process, shared-memory where possible.
- Scene batching: batched stepping for CPU/GPU-friendly sims.
- Gymnasium-compatible adapter and conversion utilities.

## Milestone 5 — Performance & Vectorization
- Emulation wrappers: flatten/unflatten structured obs/action into compact tensors with inverse mapping.
- Vectorization backends: serial + multiprocess; many envs per worker with preallocated buffers.
- Framework adapters: thin shims around common RL trainers + model API (encode/decode, recurrent state packing).
- One-liner UX: `make(name, num_envs, backend=...)` + `emulate(env, num_envs=...)`.
- Vectorized rollout core: batched obs/actions, minimal Python overhead.
- Zero-copy buffers for obs/rew/done; preallocated rollout storage.
- Async stepping and rollout pipelines; configurable worker pools.
- Microbenchmarks for env throughput + policy inference.

## Milestone 6 — Usability & Reliability
- Experiment manager: sweeps, checkpoints, resume, metric dashboards.
- Evaluation suite: standardized eval runs + deterministic seeds.
- Docs and tutorials: PPO/DQN/SAC walkthroughs, env integration guide.
- Cross-backend tests: CPU/CUDA parity for policy inference (as available).

## Current Status
- TBD (placeholder for implementation progress).
