# tinyfin-rl

<p align="center">
  <img src="resources/logo/logo.png" alt="tinyfin logo" />
</p>


Tinyfin-RL is a modular reinforcement learning library intended to be built around a tensor backend (Tinyfin or another library). The core goal is a C-first, embeddable architecture with clean interfaces, optional backends, and room for higher-performance training loops and visualization.

## Status

This repository currently provides foundational building blocks (envs, spaces, policies, agents, replay buffer, trainer) plus a minimal Tinyfin backend adapter and REINFORCE scaffolding. Python examples are temporary scaffolding and will be replaced by C modules and bindings.

## Goals

- Minimal, readable RL primitives that translate to C modules.
- Optional tensor backend integration without hard coupling.
- Training visualization via Raylib (C integration).

## Direction

This library is intended to be used as a project foundation rather than a small Python-only teaching repo. The Python files in `examples/` are placeholders that will be superseded by C implementations and project templates.

## Structure

- `tinyfin_rl/` core library
- `examples/` temporary scaffolding (will be replaced by C-based samples)
- `tinyfin/` Tinyfin submodule (tensor core + C backend source)
- `env/` standalone environment projects (Raylib, etc.)
- `raylib/` raylib-quickstart submodule for C visualization and env prototypes
- `raylib-src/` raylib source submodule for direct builds without premake
- `docs/` project notes and usage guides
- `roadmap.md` milestone plan
 - `tests/` backend parity checks (CPU/CUDA) when Tinyfin is available

## Docs

- `docs/overview.md` project overview
- `docs/install.md` setup and optional dependencies
- `docs/first_algorithm.md` REINFORCE sketch
- `docs/api_conventions.md` API expectations
- `docs/c_api.md` C API overview
- `docs/trpo.md` minimal TRPO placeholder
- `docs/ppo_walkthrough.md` PPO walkthrough
- `docs/dqn_walkthrough.md` DQN walkthrough
- `docs/sac_walkthrough.md` SAC walkthrough
- `docs/adapters.md` Gymnasium + PettingZoo adapters
- `env/first-env/README.md` raylib gridworld + training notes
- `env/lineworld/README.md` headless lineworld + Q-learning

## Quickstart (C examples)

Build the C examples and run the deterministic trainer smoke test:

```bash
cd tinyfin-rl
make
./build/trainer_smoke
```

## Raylib env (train + autoplay)

Train the Q-learning policy and watch it in the raylib viewer:

```bash
cd tinyfin-rl/env/first-env
make train_q
./train_q --episodes 2000 --report 200 --out q_table.csv
make
./first_env
```

Press `P` to toggle autoplay and `L` to reload `q_table.csv`.

## Headless env (lineworld)

Train a Q-learning policy in a simple 1D lineworld:

```bash
cd tinyfin-rl/env/lineworld
make
./train_q --episodes 2000 --report 200 --out q_table.csv
```

## Python scaffolding (PPO/A2C/DQN)

The Python trainers require Tinyfin. Point `PYTHONPATH` at `tinyfin/python`:

```bash
export PYTHONPATH=/path/to/tinyfin/python
python examples/ppo_bandit.py
python examples/a2c_bandit.py
python examples/bc_bandit.py
python examples/trpo_bandit.py
python examples/ppo_checkpoint.py
python examples/dqn_checkpoint.py
python examples/td3_bandit.py
python examples/sac_bandit.py
python examples/gym_cartpole_dqn.py
python examples/gym_cartpole_ppo.py
python examples/gym_pendulum_td3.py
python examples/gym_pendulum_sac.py
python examples/pettingzoo_coop_pong_random.py
```

Examples auto-add the repo root to `sys.path`, so you can also run them from `examples/` without setting `PYTHONPATH` to the repo.

Gymnasium/PettingZoo adapters require those packages to be installed.

## C plugin loader

Build and run the plugin loader against the grid env:

```bash
cd tinyfin-rl
make env_plugin_loader
cd env/first-env
make libfirst_env.so
../../build/env_plugin_loader ./libfirst_env.so
```

## Benchmark rollout throughput

Run the vectorized rollout benchmark and optionally gate on steps/sec:

```bash
cd tinyfin-rl
python examples/benchmark_rollout.py --storage array --num-envs 64 --steps 200 --min-steps-per-s 5000
scripts/bench_rollout.sh array 5000 64 200
```

Reference benchmark (env-var configured):

```bash
cd tinyfin-rl
MIN_STEPS_PER_S=5000 ENVS=64 STEPS_PER_ENV=200 scripts/bench_reference.sh
```

Compare benchmark results (exit non-zero if below threshold):

```bash
cd tinyfin-rl
python scripts/bench_compare.py --storage array --num-envs 64 --steps 200 --min-steps-per-s 5000
```

Targets file (default threshold values):

```bash
cd tinyfin-rl
cat benchmarks/rollout_targets.json
python scripts/bench_compare.py
```

## Experiment logging

Write metrics to JSONL with the experiment helper:

```bash
cd tinyfin-rl
python - <<'PY'
from tinyfin_rl.experiment import run_experiment
from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.envs import BanditEnv
from tinyfin_rl.policy_tinyfin import PPOPolicy
from tinyfin_rl.algos.ppo import PPOTrainer
from tinyfin_rl.spaces import Discrete

backend = TinyfinBackend(device="cpu")
env = BanditEnv([0.2, 0.8])
policy = PPOPolicy(backend=backend, obs_dim=1, action_space=Discrete(2))
trainer = PPOTrainer(env=env, policy=policy, steps_per_batch=8, epochs=1)
run_experiment(trainer, "runs/ppo_metrics.jsonl", updates=2)
PY
```

## Evaluation harness

Deterministic evaluation under fixed seeds:

```bash
cd tinyfin-rl
python - <<'PY'
from tinyfin_rl.envs import BanditEnv
from tinyfin_rl.policy import RandomPolicy
from tinyfin_rl.eval import evaluate_policy

env = BanditEnv([0.2, 0.8])
policy = RandomPolicy(env.action_space)
out = evaluate_policy(env, policy, episodes=5, seed=123)
print(out["return_mean"])
PY
```

## Dashboard (D3)

Serve a simple D3 dashboard for JSONL metrics:

```bash
cd tinyfin-rl
python scripts/dashboard_server.py --metrics runs/ppo_metrics.jsonl --port 8000
```

Note: the dashboard loads D3 from a CDN; for offline use, vendor a local copy in `dashboard/`.

## C backend and Tinyfin

Tinyfin lives in a separate repository and is included here as a submodule. The key requirement is a stable C-facing ABI. The `CBackend` loader is a placeholder for that integration.

For Python scaffolding, `TinyfinBackend` wires the Tinyfin Python bindings into a minimal policy/optimizer loop. Run examples from this repo with:

```bash
PYTHONPATH=/path/to/tinyfin/python python examples/reinforce_bandit.py
PYTHONPATH=/path/to/tinyfin/python python examples/ppo_bandit.py
PYTHONPATH=/path/to/tinyfin/python python examples/ppo_bandit_vector.py
PYTHONPATH=/path/to/tinyfin/python python examples/a2c_bandit.py
PYTHONPATH=/path/to/tinyfin/python python examples/benchmark_rollout.py
```

## Raylib visualization

Visualization is intended to be a C/Raylib integration. The current Python hooks only exist to keep the training loop modular until the C modules land.

## License

See `LICENSE`.
