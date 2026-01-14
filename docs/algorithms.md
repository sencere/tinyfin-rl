# Algorithms

Tinyfin-RL selects algorithms via `--algo`. All algorithms share the
`tfrl_algo` interface in `src/core/algo_api.h`.

Defaults are versioned and applied per algorithm in `src/core/algo_factory.c`.

Note: Continuous (box) actions are currently supported by `random`, `sac`, and `td3`. MCTS currently supports `maze_rooms` and `lineworld`.

## DQN

Discrete-action DQN with a Tinyfin `Linear` policy head.

```bash
./build/tinyfin-rl train --algo dqn --steps 2000
```

Flags:

- `--gamma`
- `--lr`
- `--epsilon`
- `--replay-size`
- `--batch-size`
- `--per-alpha`
- `--per-beta`
- `--c51-atoms`
- `--c51-vmin`
- `--c51-vmax`
- `--save PATH` / `--load PATH`

## Random

Baseline random policy (supports discrete and box actions).

```bash
./build/tinyfin-rl train --algo random --steps 2000
```

## Rainbow DQN (C51 + Dueling)

Rainbow DQN with n-step returns, dueling heads, and C51-style categorical support.

```bash
./build/tinyfin-rl train --algo rainbow --steps 2000
```

Flags:

- `--gamma`
- `--lr`
- `--epsilon`
- `--replay-size`
- `--batch-size`
- `--per-alpha`
- `--per-beta`
- `--save PATH` / `--load PATH`

## QR-DQN (Minimal)

Quantile-regression DQN with a fixed set of quantiles per action.

```bash
./build/tinyfin-rl train --algo qrdqn --steps 2000
```

Flags:

- `--gamma`
- `--lr`
- `--epsilon`
- `--replay-size`
- `--batch-size`
- `--per-alpha`
- `--per-beta`
- `--save PATH` / `--load PATH`

## IQN (Minimal)

Implicit quantile regression with sampled taus appended to the input features.

```bash
./build/tinyfin-rl train --algo iqn --steps 2000
```

Flags:

- `--gamma`
- `--lr`
- `--epsilon`
- `--replay-size`
- `--batch-size`
- `--per-alpha`
- `--per-beta`
- `--iqn-quantiles`
- `--iqn-tau-samples`
- `--save PATH` / `--load PATH`

## MCTS (Planner)

UCT-style MCTS for discrete environments (`maze_rooms`, `lineworld`).

```bash
./build/tinyfin-rl train --algo mcts --env maze_rooms --steps 2000 --mcts-sims 400 --mcts-depth 80
```

Flags:

- `--mcts-sims`
- `--mcts-depth`

## REINFORCE

Vanilla policy gradient with episodic returns.

```bash
./build/tinyfin-rl train --algo reinforce --steps 2000 --steps-per-batch 512
```

Save a policy:

```bash
./build/tinyfin-rl train --algo reinforce --steps 2000 --save runs/reinforce
```

## PPO

Minimal PPO with a Tinyfin `Linear` policy + value head, clipped ratio loss (min of ratio vs clip), value clipping,
advantage normalization, GAE(λ), entropy bonus, and KL early stopping (`--kl-target`).

```bash
./build/tinyfin-rl train --algo ppo --steps 2000 --steps-per-batch 64 --epochs 2 --clip-eps 0.2 --gae-lambda 0.95 --entropy-coef 0.01
```

## A2C (Minimal)

Advantage actor-critic with a shared optimizer for policy + value heads.

```bash
./build/tinyfin-rl train --algo a2c --steps 2000 --steps-per-batch 64 --entropy-coef 0.01
```

## A3C (Async, Minimal)

Async actor-critic with a shared optimizer guarded by a mutex. Uses multiple envs (`--envs`) to drive concurrent updates.
Rendering and trace output are disabled in async mode.

```bash
./build/tinyfin-rl train --algo a3c --steps 2000 --envs 4 --steps-per-batch 64 --entropy-coef 0.01
```

## TRPO (KL-Penalty, Minimal)

KL-penalized policy updates with a value baseline (approximate TRPO).

```bash
./build/tinyfin-rl train --algo trpo --steps 2000 --steps-per-batch 64 --clip-eps 0.01
```

## SAC (Discrete/Continuous, Minimal)

Soft actor-critic for discrete actions with twin Q networks. For box actions, this uses a stochastic Gaussian policy
(mean/log-std head) with tanh squashing and log-prob correction for the entropy term.

```bash
./build/tinyfin-rl train --algo sac --steps 2000 --entropy-coef 0.2
```

Off-policy flags:

- `--replay-size`
- `--batch-size`
- `--per-alpha`
- `--per-beta`

## TD3 (Discrete/Continuous, Minimal)

Twin Q networks with delayed policy updates for discrete actions. For box actions, this uses a minimal continuous variant
with deterministic actions (no entropy term) and target policy smoothing noise.

```bash
./build/tinyfin-rl train --algo td3 --steps 2000
```

Off-policy flags:

- `--replay-size`
- `--batch-size`
- `--per-alpha`
- `--per-beta`

## IMPALA / V-trace (Minimal)

V-trace corrections over on-policy rollouts (single-process).

```bash
./build/tinyfin-rl train --algo impala --steps 2000 --steps-per-batch 64 --clip-eps 1.0
```

To run with an async actor/learner split:

```bash
./build/tinyfin-rl train --algo impala --steps 2000 --actor-count 4 --queue-capacity 1024
```

Rendering and trace output are disabled when using the split runner.

Checkpoint format:

- DQN: `PATH.q.w.tensor`, `PATH.q.b.tensor`
- Rainbow DQN: `PATH.adv.w.tensor`, `PATH.adv.b.tensor`, `PATH.val.w.tensor`, `PATH.val.b.tensor`
- QR-DQN: `PATH.q.w.tensor`, `PATH.q.b.tensor`
- IQN: `PATH.q.w.tensor`, `PATH.q.b.tensor`
- REINFORCE: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`
- PPO: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`, `PATH.v.w.tensor`, `PATH.v.b.tensor`
- A2C: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`, `PATH.v.w.tensor`, `PATH.v.b.tensor`
- A3C: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`, `PATH.v.w.tensor`, `PATH.v.b.tensor`
- TRPO: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`, `PATH.v.w.tensor`, `PATH.v.b.tensor`
- SAC: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`, `PATH.q1.w.tensor`, `PATH.q1.b.tensor`, `PATH.q2.w.tensor`, `PATH.q2.b.tensor`
- TD3: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`, `PATH.q1.w.tensor`, `PATH.q1.b.tensor`, `PATH.q2.w.tensor`, `PATH.q2.b.tensor`
- IMPALA: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`, `PATH.v.w.tensor`, `PATH.v.b.tensor`

---

## Additional Algorithms (Design Plan)

This section outlines a scoped design plan for future algorithm extensions.

### Goals

- Add A3C (async actor-critic), Rainbow DQN C51 + dueling, distributional variants beyond QR-DQN, and an IMPALA-scale actor/learner split.
- Keep the core runner/trace contract stable; prefer new modules over invasive refactors.

### Non-Goals

- New environments or UI changes.
- Major changes to the Tinyfin tensor core.
- Distributed deployment beyond a single host.

### Baseline Assumptions

- Algorithms implement `tfrl_algo` and are selected via `--algo`.
- Replay buffers already support PER for DQN-family off-policy algorithms.
- IMPALA exists in a single-process V-trace form; PPO/A2C already provide policy/value heads.

### A3C (Async Actor-Critic)

- **Architecture:** N env workers, each with local rollouts; shared global params.
- **Update Flow:** worker computes gradients on its local batch and applies to shared optimizer.
- **Minimal Infra:** add a thread-safe optimizer step or a gradient queue with a learner thread.
- **Config:** `--threads` controls actor count; add `--async` or a new `--algo a3c`.
- **Diagnostics:** per-worker return stats, update lag, gradient norm.

### Rainbow DQN Extensions (C51 + Dueling)

- **Model:** split into value stream V(s) and advantage stream A(s, a) combined as `Q = V + (A - mean(A))`.
- **Distributional Head:** categorical logits over fixed support atoms (C51).
- **Loss:** cross-entropy between projected target distribution and predicted atoms.
- **Replay:** reuse PER; TD error becomes KL/cross-entropy for priorities.
- **Config:** `--atoms`, `--v-min`, `--v-max` (defaults aligned with DQN scale).

### Distributional Beyond QR-DQN

- **Candidate:** IQN (implicit quantile networks) or FQF (fully parameterized quantile function).
- **Model:** quantile embeddings + shared trunk + action head.
- **Loss:** quantile regression with Huber for sampled taus.
- **Replay:** same PER path; priority from quantile TD error.
- **Scope:** pick one variant (IQN preferred) to avoid parallel abstractions.

### IMPALA-Scale Actor/Learner Split

- **Topology:** actor threads/processes collect rollouts; a learner consumes batches.
- **Transport:** shared-memory ring buffer first; optional socket/pipe framing later.
- **Backpressure:** bounded queue with drop or block strategy; expose queue depth.
- **Learner:** uses V-trace; optionally batches across actors for throughput.
- **Config:** `--actor-count`, `--learner-batch`, `--queue-capacity`.

### Shared Infrastructure Changes

- New model components (dueling head, categorical support, quantile embeddings).
- Replay buffer metadata for distributional loss (optional).
- Lightweight threading/queue utilities for async and actor/learner split.
- Algo diagnostics surface (trace metadata + stdout summaries).

### Testing Plan

- Unit tests for C51 projection and quantile loss.
- Deterministic smoke tests with fixed seeds.
- Replay buffer tests for distributional priority updates.
- Minimal convergence checks (short runs, sanity curves).

### Rollout Order

1. Rainbow C51 + dueling (most contained).
2. IQN (or FQF) distributional variant.
3. A3C async (threading).
4. IMPALA-scale actor/learner split (queues/process separation).
