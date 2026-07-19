#!/usr/bin/env sh
set -eu

make tests
./build/test_replay_buffer
./build/test_point1d_smoke
./build/test_maze_rooms_smoke
./build/test_lineworld_duo_smoke
./build/test_coin_maze_duo_smoke
./build/test_public_headers
./build/test_nca_smoke
./build/test_env_modules
./build/test_env_dynamic_load
