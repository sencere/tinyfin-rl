# cartpole_simple (C env)

Lightweight cartpole with box observation (x, x_dot, theta, theta_dot).

## Build

```bash
cd tinyfin-rl/environments/cartpole_simple
make
```

## Run viewer

```bash
./cartpole_simple
```

## Train (C trainers)

Current C trainers expect discrete observations. Cartpole is `Box`-valued, so you must discretize it first or use the Gymnasium Python examples for now.

## Renderer sample (dlopen)

```bash
cd tinyfin-rl/environments/common
make
./plugin_viewer ../cartpole_simple
```
