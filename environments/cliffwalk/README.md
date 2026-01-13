# cliffwalk (C env)

Classic cliff-walking grid with large penalty for stepping on the cliff.

## Build

```bash
cd tinyfin-rl/environments/cliffwalk
make
```

## Run viewer

```bash
./cliffwalk
```

## Train (C trainers)

```bash
cd tinyfin-rl
make train_dqn train_ppo
./build/train_dqn --env environments/cliffwalk/libcliffwalk.so --obs-n 48 --action-n 4
./build/train_ppo --env environments/cliffwalk/libcliffwalk.so --obs-n 48 --action-n 4
```

## Renderer sample (dlopen)

```bash
cd tinyfin-rl/environments/common
make
./plugin_viewer ../cliffwalk
```
