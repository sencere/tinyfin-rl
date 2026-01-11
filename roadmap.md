# tinyfin-rl Roadmap

**Goal:** A modular, C-first RL library with a pluggable tensor backend (Tinyfin or alternatives), fast simulation, and solid algorithm coverage. Python is scaffolding, not the product.

## Immediate Focus (1 sprint)
- Lock C API & ABI (`Env`, `Agent`, `Trainer`, tensor backend).
- Ship one on-policy algorithm (PPO or A2C) end-to-end in C.
- Establish a baseline vectorized rollout benchmark.

## Core Milestones

### 0 — Foundations (Done)
- C-first project skeleton, build, logging, deterministic seeding.
- Core C APIs: envs, agents, policies, buffers, trainers.
- Tensor backend ABI (Tinyfin-compatible).
- Minimal visualization hooks.
- **Result:** Headers compile, backend exercised, deterministic C example runs.

### 1 — On-Policy RL (Done)
- REINFORCE / A2C / PPO (+ GAE).
- Advantage normalization, entropy bonus, value loss, grad clipping.
- Unit tests for returns & GAE math.
- **Result:** PPO/A2C hits reward thresholds on reference envs.

### 2 — Off-Policy RL (Done)
- DQN family (Double, Dueling, Prioritized).
- Continuous control (TD3, SAC baseline).
- Replay buffers (uniform, prioritized, n-step).
- Target networks & Polyak updates.
- **Result:** DQN/TD3 pass end-to-end reward checks.

### 3 — Advanced & Specialized (Done)
- TRPO-style actor–critic.
- Offline RL baseline (BC / CQL).
- Multi-agent primitives (shared policy, centralized critic hooks).
- **Result:** Advanced algorithm runs with example + docs.

## Product & Performance

### 4 — Environments & Simulation (Done)
- Env wrappers (time-limit, frame stack, normalization, action repeat).
- Vectorized envs (sync + async, deterministic seeding).
- C-first env API with optional adapters.
- **Result:** Composable wrappers and vector envs validated.

### 5 — Performance & Vectorization (Done)
- Vectorized rollout core with preallocated, zero-copy buffers.
- Async stepping pipelines and worker pools.
- Flatten/unflatten emulation helpers.
- Benchmarks for env + inference throughput.
- **Result:** Steps/sec targets met and benchmark-gated.

### 6 — Usability & Reliability (Done)
- Experiment manager (checkpoints, resume, sweeps).
- Evaluation harness with deterministic seeds.
- Core docs & walkthroughs (PPO, DQN, SAC).
- **Result:** Reproducible training across multiple algorithms.

## Current State (Summary)
- C API stabilized with multiple working C examples.
- PPO, A2C, DQN, TD3, SAC available (C-first focus, Python where needed).
- Vectorized rollouts, async execution, and benchmarks in place.
- Env plugins, wrappers, Gymnasium/PettingZoo adapters supported.
- Docs and walkthroughs cover core workflows.

## Next Steps
- Expand in-repo C environments.
- Pair each env with a reference algorithm example.
- Reduce Python surface area further in favor of C.
