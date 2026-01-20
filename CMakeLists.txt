cmake_minimum_required(VERSION 3.13)
project(tinyfin_rl C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(TFRL_WITH_RAYLIB "Enable raylib viewer" OFF)
option(TFRL_PERF_FLAGS "Enable aggressive local perf flags" OFF)
set(TINYFIN_DIR "tinyfin" CACHE PATH "Path to tinyfin")

set(SRC_ENVS
    src/envs/envs.c
    src/envs/registry.c
    src/envs/maze/maze.c
    src/envs/maze/render.c
    src/envs/lineworld/lineworld.c
    src/envs/lineworld/render.c
    src/envs/point1d/point1d.c
    src/envs/point1d/render.c
    src/envs/coin_maze/coin_maze.c
    src/envs/coin_maze/render.c
    src/envs/snake/snake_env.c
    src/envs/snake/render.c
    src/envs/floppy/floppy_env.c
    src/envs/floppy/render.c
    src/envs/tetris/tetris_env.c
    src/envs/tetris/render.c
    src/envs/pang/pang_env.c
    src/envs/pang/render.c
    src/envs/py_bridge/py_bridge.c
    src/envs/render.c
    src/envs/render_grid.c
)

set(SRC_CORE
    ${SRC_ENVS}
    src/core/algo_factory.c
    src/core/algo_random.c
    src/core/algo_dqn.c
    src/core/algo_rainbow.c
    src/core/algo_qrdqn.c
    src/core/algo_iqn.c
    src/core/algo_a2c.c
    src/core/algo_a3c.c
    src/core/algo_trpo.c
    src/core/algo_sac.c
    src/core/algo_td3.c
    src/core/algo_impala.c
    src/core/algo_mcts.c
    src/core/algo_reinforce.c
    src/core/algo_ppo.c
    src/core/algo_tetris_plan.c
    src/core/replay_buffer.c
    src/core/trace.c
)

set(SRC_APP
    src/main.c
    src/cli/cli.c
    src/runner/runner.c
)

if (TFRL_WITH_RAYLIB)
    set(SRC_VIEWER src/envs/viewer_raylib.c)
else()
    set(SRC_VIEWER src/envs/viewer_stub.c)
endif()

add_executable(tinyfin-rl ${SRC_CORE} ${SRC_APP} ${SRC_VIEWER})
target_include_directories(tinyfin-rl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(tinyfin-rl PRIVATE ${TINYFIN_DIR}/include)
if (TFRL_PERF_FLAGS)
    target_compile_options(tinyfin-rl PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
endif()

find_library(TINYFIN_LIB tinyfin PATHS ${TINYFIN_DIR} NO_DEFAULT_PATH)
if (NOT TINYFIN_LIB)
    set(TINYFIN_LIB ${TINYFIN_DIR}/libtinyfin.so)
endif()
target_link_libraries(tinyfin-rl ${TINYFIN_LIB} m pthread)
target_link_libraries(tinyfin-rl rt)

if (TFRL_WITH_RAYLIB)
    target_include_directories(tinyfin-rl PRIVATE raylib-src/src)
    target_link_libraries(tinyfin-rl raylib)
endif()

add_library(tfrl_env SHARED ${SRC_ENVS})
target_include_directories(tfrl_env PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if (TFRL_PERF_FLAGS)
    target_compile_options(tfrl_env PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
endif()

enable_testing()

add_executable(test_replay_buffer tests/test_replay_buffer.c src/core/replay_buffer.c)
target_include_directories(test_replay_buffer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_replay_buffer m)
if (TFRL_PERF_FLAGS)
    target_compile_options(test_replay_buffer PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
endif()
add_test(NAME replay_buffer COMMAND test_replay_buffer)

add_executable(test_point1d_smoke tests/test_point1d_smoke.c ${SRC_ENVS})
target_include_directories(test_point1d_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_point1d_smoke m)
if (TFRL_PERF_FLAGS)
    target_compile_options(test_point1d_smoke PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
endif()
add_test(NAME point1d_smoke COMMAND test_point1d_smoke)

add_executable(test_lineworld_duo_smoke tests/test_lineworld_duo_smoke.c ${SRC_ENVS})
target_include_directories(test_lineworld_duo_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_lineworld_duo_smoke m)
if (TFRL_PERF_FLAGS)
    target_compile_options(test_lineworld_duo_smoke PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
endif()
add_test(NAME lineworld_duo_smoke COMMAND test_lineworld_duo_smoke)

add_executable(test_coin_maze_duo_smoke tests/test_coin_maze_duo_smoke.c ${SRC_ENVS})
target_include_directories(test_coin_maze_duo_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_coin_maze_duo_smoke m)
if (TFRL_PERF_FLAGS)
    target_compile_options(test_coin_maze_duo_smoke PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
endif()
add_test(NAME coin_maze_duo_smoke COMMAND test_coin_maze_duo_smoke)
