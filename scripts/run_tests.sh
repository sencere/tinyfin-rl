#!/usr/bin/env sh
set -eu

make tests
./build/test_replay_buffer
./build/test_point1d_smoke
