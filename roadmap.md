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

## 1 — Rendering & Visualization (Partial)

### Scope
- Optional raylib viewer module
- Viewer consumes render snapshots (no env coupling)
- Live rendering attached via parameters
- Offline replay via trace files

### Key Deliverables
- `--render off|live`
- `--render-env N` (missing)
- `tinyfin-rl replay --trace-in file.tft`
- Viewer lifecycle fully owned outside simulation

**Result:**  
Rendering is safe, optional, and non-invasive. `--render-env` still needs wiring.

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

## 3 — Python (Optional Layer) (Partial)

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
- Env API bindings (missing)
- Checkpoint loading/saving bindings (missing)

**Result:**  
Python is a convenience layer via CLI; bindings are still minimal.

---

## 4 — Environments & Simulation (Partial)

### Scope
- Solidify **one canonical environment**
- Improve env metadata (spaces, shapes, dtypes)
- Ensure env determinism and reset guarantees

### Stretch
- Add second environment only if architecture remains unchanged

**Result:**  
Two environments exist (`maze_rooms`, `lineworld`). Metadata beyond the core spec is still missing.

---

## 5 — Performance & Scaling (Partial)

### Scope
- Vectorized environment stepping in C
- Batched inference via Tinyfin
- Benchmark-driven development

### Key Metrics
- Steps/sec (headless)
- Inference throughput
- Training wall-clock stability

**Result:**  
Vectorized stepping and a benchmark script exist. Batched inference is not implemented.

---

## 6 — Usability & Reliability (Partial)

### Scope
- Reproducible training runs
- Stable checkpoint format
- End-to-end documentation

### Key Deliverables
- `ARCHITECTURE.md`
- Algorithm walkthroughs
- Rendering & replay guides

**Result:**  
Core docs exist. Add a dedicated env API doc and a checkpoint format doc for completeness.

---

## Current State (Summary)

- Tinyfin provides a mature tensor + autograd backend
- Tinyfin-RL now builds a single `tinyfin-rl` binary from `src/`
- The runner supports train/eval/replay with configurable algorithms
- Rendering is optional and isolated via render snapshots + a viewer module
- Two reference environments: `maze_rooms` and `lineworld`
- Optional Python wrapper calls the C binary

---

## Near-Term Next Steps

1. Implement `--render-env` (multi-env viewer selection)
2. Add Python bindings for env API + checkpoint IO (not just CLI wrapper)
3. Expand env metadata (spaces, shapes, dtypes)
4. Add batched inference path in Tinyfin-RL
5. Add richer environment suite

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
