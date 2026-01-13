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

### Key Deliverables
- `--algo ppo | dqn | reinforce`
- Shared algorithm interface
- Reference implementation per algorithm

**Result:**  
C-first training is the default path.

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
Two environments exist (`maze_rooms`, `lineworld`) with space/dtype metadata in `tfrl_env_spec`.

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
Vectorized stepping, a benchmark script, and DQN batched inference are implemented.

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
- Two reference environments: `maze_rooms` and `lineworld`
- Optional Python wrapper calls the C binary and exposes env bindings
- Batched inference for DQN when using `--envs`

---

## Near-Term Next Steps

1. Improve PPO stability and add advantage normalization
2. Extend render snapshot specs for richer environments
3. Add richer environment suite
4. Add true parallel stepping (threads)

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
