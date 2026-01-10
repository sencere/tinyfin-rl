# tinyfin-rl Roadmap (Concise)

**Goal:** A modular, C-first RL library with a pluggable tensor backend (Tinyfin or alternatives), fast simulation, and strong algorithm coverage. Python scaffolding is temporary.

**Next up (one sprint):**
- Lock the C API surface for `Env`/`Agent`/`Trainer` and finalize ABI for tensor backend.
- Ship one on-policy algo (PPO or A2C) end-to-end with tests + a C example.
- Baseline throughput benchmark for vectorized rollouts.

## Core Engine

### Milestone 0 — Foundations
- C-first project skeleton, build system, logging, and reproducible seeding.
- Core RL API in C: `Env`, `Space`, `Agent`, `Policy`, `ReplayBuffer`, `Trainer`.
- Tensor backend ABI: minimal tensor ops, optim, and NN adapters (Tinyfin or other).
- Visualization hooks in C with Raylib integration.
- Placeholder samples to validate the pipeline (to be replaced by C examples).
- Acceptance:
  - [x] Core C API headers compile in isolation with a minimal smoke test.
  - [x] Backend ABI exercised by a toy policy + optimizer step.
  - [x] One end-to-end C example runs with deterministic seeding.

### Milestone 1 — On-Policy Essentials
- REINFORCE, VPG (vanilla policy gradient), A2C implemented in C.
- PPO (clip + KL variants), GAE.
- Common utilities: advantage normalization, entropy bonus, value loss, grad clipping.
- Unit tests for returns/GAE/advantage math (C + optional Python harness).
- Acceptance:
  - [x] PPO or A2C reaches a basic reward threshold on a reference env.
  - [x] GAE/returns math validated against a reference implementation.
  - [x] Unit tests cover normalization/entropy/value loss paths.

### Milestone 2 — Off-Policy Essentials
- DQN family: DQN, Double DQN, Dueling, Prioritized Replay.
- Continuous control: DDPG, TD3, SAC (v1/v2 notes).
- Replay buffers: uniform, prioritized, n-step, sequence replay (C implementation).
- Target networks, Polyak averaging, exploration noise helpers.
- Acceptance:
  - [x] DQN or TD3 hits a basic reward threshold on a reference env.
  - [x] Replay buffer variants pass shape and sampling tests.
  - [x] Target network update logic tested (Polyak and hard update).

### Milestone 3 — Advanced & Specialized
- Actor-critic variants: TRPO, ACKTR (optional).
- Offline RL: behavior cloning, CQL/BCQ (minimal baseline).
- Imitation + RLHF hooks: dataset loaders, preference model interface.
- Multi-agent primitives: shared policy, independent learners, centralized critic hooks.
- Acceptance:
  - [x] One offline baseline (BC or CQL) runs end-to-end on a toy dataset.
  - [x] Multi-agent API can train a simple shared-policy task.
  - [x] Advanced algo added with minimal doc + example.

## Product Surface

### Milestone 4 — Simulation & Env Tooling
- Env wrappers: time-limit, frame stack, normalization, action repeat.
- Vectorized envs: sync/async, multi-process, shared-memory where possible.
- Scene batching: batched stepping for CPU/GPU-friendly sims.
- C-first environment API with optional Python adapters (not primary).
- Acceptance:
  - [ ] Wrappers are composable and tested (time-limit + frame stack).
  - [x] Wrappers are composable and tested (time-limit + frame stack).
  - [x] Vectorized envs run sync + async with stable seeding.
  - [x] One environment adapter demoed in C.

## Acceleration

### Milestone 5 — Performance & Vectorization
- Emulation wrappers: flatten/unflatten structured obs/action into compact tensors with inverse mapping.
- Vectorization backends: serial + multiprocess; many envs per worker with preallocated buffers.
- Framework adapters: thin shims around model APIs (encode/decode, recurrent state packing).
- One-liner UX: `make(name, num_envs, backend=...)` + `emulate(env, num_envs=...)`.
- Vectorized rollout core: batched obs/actions with minimal overhead (C focus).
- Zero-copy buffers for obs/rew/done; preallocated rollout storage.
- Async stepping and rollout pipelines; configurable worker pools.
- Microbenchmarks for env throughput + policy inference.
- Acceptance:
  - [ ] Vectorized rollout core hits a target steps/sec on a reference env.
  - [ ] Zero-copy buffers validated with profiling hooks.
  - [ ] Benchmarks track inference + env step throughput.

## Product Surface

### Milestone 6 — Usability & Reliability
- Experiment manager: sweeps, checkpoints, resume, metric dashboards.
- Evaluation suite: standardized eval runs + deterministic seeds.
- Docs and tutorials: PPO/DQN/SAC walkthroughs, env integration guide.
- Cross-backend tests: CPU/CUDA parity for policy inference (as available).
- Acceptance:
  - [ ] Checkpoint/resume works across at least two algos.
  - [ ] Eval harness produces deterministic runs under fixed seeds.
  - [ ] Docs cover one on-policy + one off-policy walkthrough.

## Current Status
- Python scaffolding in place: envs, spaces, policies, replay buffer, agent, trainer, vector env, rollout batch.
- Tinyfin backend adapter and Tinyfin-based policies available.
- On-policy scaffolding: REINFORCE, A2C, PPO trainers in Python.
- Tests cover returns/GAE math for on-policy utilities.
- Advantage normalization, entropy, and value loss utilities covered by tests.
- PPO/A2C now include optional entropy bonus and normalized advantages.
- Added minimal C API header (`tinyfin_rl/rl.h`) with a smoke test stub.
- Vectorized rollout benchmark accepts CLI params.
- Tinyfin codebase added as `tinyfin/` submodule.
- PPO/A2C trainer smoke tests cover entropy bonus.
- PPO/A2C log entropy metrics; vectorized rollout benchmark can emit JSON/CSV.
- Added C bandit REINFORCE example using the C API.
- Added C bandit A2C example using the C API.
- Vectorized rollout resets done envs; benchmark now supports warmup/iterations stats.
- Added PPO vector-env smoke test and rollout reset test.
- Added backend optimizer-step test and a Makefile for C examples/tests.
- Added C bandit PPO example and a minimal CMake build.
- Added env wrappers (time-limit, frame stack, action repeat, normalization) with tests.
- Added raylib-quickstart submodule and first raylib env prototype.
- Added C API overview doc and async/vector wrappers skeletons.
- Added deterministic trainer smoke example in C.
- Added raylib source submodule for direct builds without premake.
- Added Q-learning trainer with metrics export and raylib autoplay.
- Added env plugin API header with a grid plugin + loader example.
- Added vector env wrappers (repeat + normalization) and a one-shot train/run script.
- Added wrapper composition test and vector env seeding tests.
- Added PPO reward-threshold test and GAE reference validation.
- Added replay buffer variants with tests and DQN target update helpers.
- Added DQN bandit end-to-end threshold test.
- Added Double DQN and prioritized replay support in DQN trainer.
- Added dueling DQN network and target update tests.
- Added quickstart docs for C examples, raylib env, Python scaffolding, and plugin loader.
- Added offline dataset loader stubs (transitions + preferences) with tests.
- Added behavior cloning trainer + bandit example/test.
- Added multi-agent API stubs with tests.
- Added minimal TRPO-style trainer with doc + example.
- Added shared-policy trainer for multi-agent bandits.
