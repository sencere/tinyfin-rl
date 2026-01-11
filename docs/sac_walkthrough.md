# SAC Walkthrough (Continuous Bandit)

This walkthrough runs the SAC trainer on a simple continuous bandit environment.

1) Set Tinyfin on PYTHONPATH:

```bash
export PYTHONPATH=/path/to/tinyfin/python
```

2) Run the SAC example:

```bash
python examples/sac_bandit.py
```

3) Optional: run the scalarized Pendulum example:

```bash
python examples/gym_pendulum_sac.py
```

4) Note: SAC currently uses scalar obs/action. Gymnasium examples use a scalar wrapper until vector/MLP policy support lands.
