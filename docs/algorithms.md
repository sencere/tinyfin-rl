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
- `--save PATH` / `--load PATH`

## Random

Baseline random policy (supports discrete and box actions).

```bash
./build/tinyfin-rl train --algo random --steps 2000
```

## Rainbow DQN (Minimal)

Rainbow-style DQN with n-step returns and a target network.

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

Minimal PPO with a Tinyfin `Linear` policy + value head, clipped ratio loss (min of ratio vs clip), advantage normalization,
and a 0.5 value-loss coefficient.

```bash
./build/tinyfin-rl train --algo ppo --steps 2000 --steps-per-batch 64 --epochs 2 --clip-eps 0.2
```

## A2C (Minimal)

Advantage actor-critic with a shared optimizer for policy + value heads.

```bash
./build/tinyfin-rl train --algo a2c --steps 2000 --steps-per-batch 64 --entropy-coef 0.01
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

Checkpoint format:

- DQN: `PATH.q.w.tensor`, `PATH.q.b.tensor`
- Rainbow DQN: `PATH.q.w.tensor`, `PATH.q.b.tensor`
- QR-DQN: `PATH.q.w.tensor`, `PATH.q.b.tensor`
- REINFORCE: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`
- PPO: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`, `PATH.v.w.tensor`, `PATH.v.b.tensor`
- A2C: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`, `PATH.v.w.tensor`, `PATH.v.b.tensor`
- TRPO: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`, `PATH.v.w.tensor`, `PATH.v.b.tensor`
- SAC: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`, `PATH.q1.w.tensor`, `PATH.q1.b.tensor`, `PATH.q2.w.tensor`, `PATH.q2.b.tensor`
- TD3: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`, `PATH.q1.w.tensor`, `PATH.q1.b.tensor`, `PATH.q2.w.tensor`, `PATH.q2.b.tensor`
- IMPALA: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`, `PATH.v.w.tensor`, `PATH.v.b.tensor`
