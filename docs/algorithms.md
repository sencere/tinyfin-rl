# Algorithms

Tinyfin-RL v2 exposes a small algorithm set through `--algo`: `dqn`, `ppo`,
`nca`, and `random`. Public consumers should include `tfrl/algo.h`; the old
`src/core/algo_api.h` path is kept as an internal compatibility shim.

Removed v1 placeholder names now fail fast instead of silently mapping to a
fallback policy: `rainbow`, `qrdqn`, `iqn`, `iql`, `a2c`, `a3c`, `trpo`, `sac`,
`td3`, `impala`, and `mcts`.

## DQN

Discrete-action value-policy path. The current implementation is a compact
tabular/simple-Q baseline behind the DQN interface; the v2 target is to replace
this with Tinyfin-backed replay, target-network, and checkpoint support.

```bash
./build/tinyfin-rl train --algo dqn --env lineworld --steps 2000
```

Supported flags:

- `--gamma`
- `--lr`
- `--epsilon`
- `--replay-size`
- `--batch-size`
- `--train-every`
- `--learning-starts`
- `--grad-steps`
- `--per-alpha`
- `--per-beta`
- `--save PATH` / `--load PATH`

## PPO

Policy-gradient path for discrete or box observations. The current
implementation routes through the existing reinforce-style learner; the v2
target is clipped PPO with GAE, value loss, entropy metrics, and checkpoints.

```bash
./build/tinyfin-rl train --algo ppo --env maze_rooms --steps 2000 --steps-per-batch 64
```

Supported flags:

- `--gamma`
- `--lr`
- `--clip-eps`
- `--kl-target`
- `--gae-lambda`
- `--steps-per-batch`
- `--epochs`
- `--entropy-coef`
- `--save PATH` / `--load PATH`

## Tinyfin-NCA

Neural cellular automata policy for discrete-action environments. The current
implementation maps observations into a 1D cell state, applies deterministic
local rollout updates, reads Q-values from the evolved state, and trains the
kernel/readout parameters online with a TD-style update.

```bash
./build/tinyfin-rl train --algo nca --env lineworld --steps 2000 --epsilon 0.1
```

Save and load:

```bash
./build/tinyfin-rl train --algo nca --env lineworld --steps 2000 --save runs/lineworld.nca
./build/tinyfin-rl eval --algo nca --env lineworld --load runs/lineworld.nca --episodes 10 --deterministic
```

Supported flags:

- `--gamma`
- `--lr`
- `--epsilon`
- `--save PATH` / `--load PATH`

Remaining NCA work: Tinyfin autograd-backed optimization, diversity metrics,
and PPO integration.

## Random

Baseline random policy for discrete and box actions.

```bash
./build/tinyfin-rl train --algo random --env lineworld --steps 2000
```

Use this for smoke tests and environment sanity checks, not as a training
baseline for completed algorithm claims.
