# Tinyfin-RL Roadmap

**Goal:**  
Build a **C-first reinforcement learning system** with a single executable,
fast deterministic simulation, configurable algorithms, and a clean tensor
backend (Tinyfin). Rendering and Python are optional tools—not architectural
requirements.

---

## Guiding Principles

- **One executable, one control loop**
- **Simulation is the source of truth**
- **Algorithms are configurable, not hardcoded**
- **Tinyfin provides tensors and autograd**
- **Python is optional and never owns execution**
- **Rendering is a parameter, not a mode**
- **Determinism is explicit and documented**

---

## Immediate Focus (Done)

### Unification & Simplification
- Finalize the **single-binary (`tinyfin-rl`) execution model**
- Remove legacy multi-host / multi-executable assumptions
- Lock the **runner-based control flow** (train / eval / replay)

### Core Stability
- Freeze minimal C APIs:
  - Environment API
  - Algorithm API
  - Render snapshot API
  - Trace format
- Ensure Tinyfin integration is clean and isolated

**Result:**  
A stable, unified C-first system with one entrypoint and one example environment.

---

## Core Milestones

---

## 0 — Foundations (Done)

### Scope
- C project skeleton and build system
- Deterministic seeding and logging
- Single executable entrypoint (`main.c`)
- Unified runner (train / eval / replay)
- Tinyfin fully integrated as tensor + autograd backend

### Key Deliverables
- Environment API (renderer-agnostic)
- Algorithm interface (policy + update loop)
- Render snapshot packet (versioned, binary)
- Trace record & replay format

**Result:**  
Tinyfin-RL can step an environment, train an algorithm, and run headless.

---

## 1 — Rendering & Visualization (Done)

### Scope
- Optional raylib viewer module
- Viewer consumes render snapshots (no env coupling)
- Live rendering attached via parameters
- Offline replay via trace files

### Key Deliverables
- `--render off|live`
- `--render-env N`
- `tinyfin-rl replay --trace-in file.tft`
- Viewer lifecycle fully owned outside simulation

**Result:**  
Rendering is safe, optional, and non-invasive.

**Update:**  
Render snapshots now include v3 metadata and entity lists for richer environments.

---

## 2 — Algorithms (Done)

### Scope
- Implement core algorithms in C
- Algorithms use Tinyfin internally for models and optimization
- Algorithm selection via configuration

### Target Algorithms
- REINFORCE (baseline)
- PPO
- DQN (discrete)
- A2C
- TRPO (KL-penalty)
- SAC (discrete + continuous)
- TD3 (discrete + continuous)
- Rainbow DQN
- QR-DQN
- IMPALA / V-trace
- MCTS (planner)

### Key Deliverables
- `--algo ppo | dqn | reinforce | a2c | trpo | sac | td3 | rainbow | qrdqn | impala | mcts`
- Shared algorithm interface
- Reference implementation per algorithm

**Result:**  
C-first training is the default path, with discrete and minimal continuous variants.

---

## 3 — Python (Optional Layer) (Done)

### Scope
- Python bindings for:
  - Environment API
  - Algorithm configuration
  - Checkpoint loading / saving
- Python defines *structure*, not execution

### Constraints
- Python never opens windows
- Python never owns the environment loop
- Python never replaces C trainers

### Key Deliverables
- Optional Python training scripts (done)
- CLI parity between C and Python usage (done)
- Env API bindings (done via `libtfrl_env.so`)
- Checkpoint loading/saving bindings (done via CLI wrapper)

**Result:**  
Python is a convenience layer via CLI and an env shared library.

---

## 4 — Environments & Simulation (Done)

### Scope
- Solidify **one canonical environment**
- Improve env metadata (spaces, shapes, dtypes)
- Ensure env determinism and reset guarantees

### Stretch
- Add second environment only if architecture remains unchanged

**Result:**  
Environments include `maze_rooms`, `lineworld`, `lineworld_cont`, `point1d`, and `coin_maze` with space/dtype metadata in `tfrl_env_spec`.

---

## 5 — Performance & Scaling (Done)

### Scope
- Vectorized environment stepping in C
- Batched inference via Tinyfin
- Benchmark-driven development

### Key Metrics
- Steps/sec (headless)
- Inference throughput
- Training wall-clock stability

**Result:**  
Vectorized stepping, threaded stepping, a benchmark script, and DQN batched inference are implemented.

---

## 6 — Usability & Reliability (Done)

### Scope
- Reproducible training runs
- Stable checkpoint format
- End-to-end documentation

### Key Deliverables
- `ARCHITECTURE.md`
- Algorithm walkthroughs
- Rendering & replay guides

**Result:**  
Docs cover training, replay, CLI, env API, and checkpoints.

---

## Current State (Summary)

- Tinyfin provides a mature tensor + autograd backend
- Tinyfin-RL now builds a single `tinyfin-rl` binary from `src/`
- The runner supports train/eval/replay with configurable algorithms
- Rendering is optional and isolated via render snapshots + a viewer module
- Reference environments: `maze_rooms`, `lineworld`, `lineworld_cont`, `point1d`, `coin_maze`
- Optional Python wrapper calls the C binary and exposes env bindings
- Batched inference for DQN when using `--envs`
- Replay buffer + PER support for DQN family, SAC, and TD3
- MCTS planner for `maze_rooms` and `lineworld`
- Render snapshots v3 with entity lists and richer metadata
- Threaded env stepping via `--threads`
- Deterministic mode via `--deterministic`
- Golden runs regression script

---

## Near-Term Next Steps

1. Improve PPO stability and add advantage normalization
2. Add typed config schema + validation per algorithm (fail fast, versioned defaults)
3. Define determinism contract and add `--deterministic` mode (threading + replay rules)
4. Golden runs regression suite (lineworld/point1d/maze) with thresholds
5. Continuous SAC/TD3: add stochastic policy + log-prob entropy
6. Replay buffer tests, determinism checks, and CI wiring

## Recent Updates

- Config validation for algorithm params and space constraints
- Deterministic mode and fixed seeding behavior
- Golden runs regression script

## Quality Tiers

- **Reference:** tested, documented, stable defaults
- **Experimental:** builds, may learn, no guarantees

## Additional Algorithms (Ideas)

- A3C (async actor-critic)
- Rainbow DQN extensions (C51 + dueling)
- Distributional variants beyond QR-DQN
- IMPALA-scale actor/learner split

## Advanced Vectorization Roadmap

If you want higher throughput beyond simple batching:

1. **Threaded env pool**  
   Spawn N worker threads, each stepping an env instance, and aggregate into a batch.
2. **Lock-free ring buffers**  
   Use single-producer/single-consumer queues to pass obs/actions without mutex overhead.
3. **Fixed-size batch scheduler**  
   Always batch up to `B` envs per inference call to keep GPU/CPU utilization consistent.
4. **Async inference**  
   Overlap env stepping with Tinyfin forward/backward on separate threads.
5. **SIMD kernels**  
   Add a vectorized backend for discrete ops to reduce per-step overhead.
6. **Replay buffer sharding**  
   Shard replay across workers to reduce contention and improve sampling speed.

---

## Long-Term Vision

Tinyfin-RL becomes:

- A **small, fast, inspectable RL system**
- Suitable for:
  - Research
  - Systems experimentation
  - Embedded and constrained environments
- Free of heavy framework assumptions

**One loop. One binary. Tinyfin underneath.**
