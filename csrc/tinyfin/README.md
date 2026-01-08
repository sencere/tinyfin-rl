# Embedded Tinyfin subset

This folder holds a minimal, self-contained subset of Tinyfin's C core so tinyfin-rl can embed a tensor backend locally.

Included today:
- `src/tensor.c`
- `src/autograd.c`
- `src/ops_reshape.c`
- `include/tinyfin/tensor.h`
- `include/tinyfin/autograd.h`
- `include/tinyfin/ops_reshape.h`

Notes:
- This is a snapshot intended for early integration and can be expanded as the RL core needs more ops.
- `tensor_flatten` is implemented locally in `ops_reshape.c` for convenience.
