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
  - [ ] Core C API headers compile in isolation with a minimal smoke test.
  - [ ] Backend ABI exercised by a toy policy + optimizer step.
  - [ ] One end-to-end C example runs with deterministic seeding.

### Milestone 1 — On-Policy Essentials
- REINFORCE, VPG (vanilla policy gradient), A2C implemented in C.
- PPO (clip + KL variants), GAE.
- Common utilities: advantage normalization, entropy bonus, value loss, grad clipping.
- Unit tests for returns/GAE/advantage math (C + optional Python harness).
- Acceptance:
  - [ ] PPO or A2C reaches a basic reward threshold on a reference env.
  - [ ] GAE/returns math validated against a reference implementation.
  - [ ] Unit tests cover normalization/entropy/value loss paths.

### Milestone 2 — Off-Policy Essentials
- DQN family: DQN, Double DQN, Dueling, Prioritized Replay.
- Continuous control: DDPG, TD3, SAC (v1/v2 notes).
- Replay buffers: uniform, prioritized, n-step, sequence replay (C implementation).
- Target networks, Polyak averaging, exploration noise helpers.
- Acceptance:
  - [ ] DQN or TD3 hits a basic reward threshold on a reference env.
  - [ ] Replay buffer variants pass shape and sampling tests.
  - [ ] Target network update logic tested (Polyak and hard update).

### Milestone 3 — Advanced & Specialized
- Actor-critic variants: TRPO, ACKTR (optional).
- Offline RL: behavior cloning, CQL/BCQ (minimal baseline).
- Imitation + RLHF hooks: dataset loaders, preference model interface.
- Multi-agent primitives: shared policy, independent learners, centralized critic hooks.
- Acceptance:
  - [ ] One offline baseline (BC or CQL) runs end-to-end on a toy dataset.
  - [ ] Multi-agent API can train a simple shared-policy task.
  - [ ] Advanced algo added with minimal doc + example.

## Product Surface

### Milestone 4 — Simulation & Env Tooling
- Env wrappers: time-limit, frame stack, normalization, action repeat.
- Vectorized envs: sync/async, multi-process, shared-memory where possible.
- Scene batching: batched stepping for CPU/GPU-friendly sims.
- C-first environment API with optional Python adapters (not primary).
- Acceptance:
  - [ ] Wrappers are composable and tested (time-limit + frame stack).
  - [ ] Vectorized envs run sync + async with stable seeding.
  - [ ] One environment adapter demoed in C.

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
- TBD (placeholder for implementation progress).
