# Algorithms

Tinyfin-RL selects algorithms via `--algo`. All algorithms share the
`tfrl_algo` interface in `src/core/algo_api.h`.

## DQN

Discrete-action DQN with a Tinyfin `Linear` policy head.

```bash
./build/tinyfin-rl train --algo dqn --steps 2000
```

Flags:

- `--gamma`
- `--lr`
- `--epsilon`
- `--save PATH` / `--load PATH`

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

Minimal PPO with a Tinyfin `Linear` policy + value head and clipped ratio loss.

```bash
./build/tinyfin-rl train --algo ppo --steps 2000 --steps-per-batch 64 --epochs 2 --clip-eps 0.2
```

Checkpoint format:

- DQN: `PATH.q.w.tensor`, `PATH.q.b.tensor`
- REINFORCE: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`
- PPO: `PATH.pi.w.tensor`, `PATH.pi.b.tensor`, `PATH.v.w.tensor`, `PATH.v.b.tensor`
