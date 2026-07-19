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
    src/envs/breakout/breakout_env.c
    src/envs/breakout/render.c
    src/envs/breakout_atari_env.c
    src/envs/breakout_atari_render.c
    src/envs/pang/pang_env.c
    src/envs/pang/render.c
    src/envs/seq_pixels_env.c
    src/envs/py_bridge/py_bridge.c
    src/envs/render.c
    src/envs/render_grid.c
)

set(SRC_ENV_SUPPORT
    src/core/obs_layout.c
)

set(SRC_ENV_MODULE_CORE
    src/envs/envs.c
    src/envs/registry.c
    src/envs/render.c
    src/envs/render_grid.c
    ${SRC_ENV_SUPPORT}
)

set(SRC_CORE
    ${SRC_ENVS}
    ${SRC_ENV_SUPPORT}
    src/core/algo_factory.c
    src/core/algo_random.c
    src/core/algo_dqn.c
    src/core/algo_nca.c
    src/core/algo_reinforce.c
    src/core/algo_ppo.c
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
target_include_directories(tinyfin-rl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(tinyfin-rl PRIVATE ${TINYFIN_DIR}/include)
if (TFRL_PERF_FLAGS)
    target_compile_options(tinyfin-rl PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
endif()

find_library(TINYFIN_LIB tinyfin PATHS ${TINYFIN_DIR} NO_DEFAULT_PATH)
if (NOT TINYFIN_LIB)
    set(TINYFIN_LIB ${TINYFIN_DIR}/libtinyfin.so)
endif()
target_link_libraries(tinyfin-rl ${TINYFIN_LIB} m pthread dl)
target_link_libraries(tinyfin-rl rt)

if (TFRL_WITH_RAYLIB)
    target_include_directories(tinyfin-rl PRIVATE raylib-src/src)
    target_link_libraries(tinyfin-rl raylib)
endif()

add_library(tfrl_env SHARED ${SRC_ENVS} ${SRC_ENV_SUPPORT})
target_include_directories(tfrl_env PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(tfrl_env dl)
if (TFRL_PERF_FLAGS)
    target_compile_options(tfrl_env PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
endif()

function(add_tfrl_env_module target module_define)
    add_library(${target} SHARED ${ARGN})
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(${target} PRIVATE ${module_define} TFRL_ENV_NO_PYBRIDGE TFRL_ENV_NO_DYNAMIC)
    if (TFRL_PERF_FLAGS)
        target_compile_options(${target} PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
    endif()
endfunction()

add_tfrl_env_module(tfrl_env_lineworld TFRL_ENV_MODULE_LINEWORLD
    ${SRC_ENV_MODULE_CORE}
    src/envs/lineworld/lineworld.c
    src/envs/lineworld/render.c
)
add_tfrl_env_module(tfrl_env_maze_rooms TFRL_ENV_MODULE_MAZE_ROOMS
    ${SRC_ENV_MODULE_CORE}
    src/envs/maze/maze.c
    src/envs/maze/render.c
)
add_tfrl_env_module(tfrl_env_coin_maze TFRL_ENV_MODULE_COIN_MAZE
    ${SRC_ENV_MODULE_CORE}
    src/envs/coin_maze/coin_maze.c
    src/envs/coin_maze/render.c
)

add_library(tfrl_public_api INTERFACE)
target_include_directories(tfrl_public_api INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

enable_testing()

add_executable(test_replay_buffer tests/test_replay_buffer.c src/core/replay_buffer.c)
target_include_directories(test_replay_buffer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_replay_buffer m)
if (TFRL_PERF_FLAGS)
    target_compile_options(test_replay_buffer PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
endif()
add_test(NAME replay_buffer COMMAND test_replay_buffer)

add_executable(test_point1d_smoke tests/test_point1d_smoke.c ${SRC_ENVS} ${SRC_ENV_SUPPORT})
target_include_directories(test_point1d_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_point1d_smoke m)
target_link_libraries(test_point1d_smoke dl)
if (TFRL_PERF_FLAGS)
    target_compile_options(test_point1d_smoke PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
endif()
add_test(NAME point1d_smoke COMMAND test_point1d_smoke)

add_executable(test_maze_rooms_smoke tests/test_maze_rooms_smoke.c ${SRC_ENVS} ${SRC_ENV_SUPPORT})
target_include_directories(test_maze_rooms_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_maze_rooms_smoke dl m)
add_dependencies(test_maze_rooms_smoke tfrl_env_maze_rooms)
if (TFRL_PERF_FLAGS)
    target_compile_options(test_maze_rooms_smoke PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
endif()
add_test(NAME maze_rooms_smoke COMMAND test_maze_rooms_smoke)

add_executable(test_lineworld_duo_smoke tests/test_lineworld_duo_smoke.c ${SRC_ENVS} ${SRC_ENV_SUPPORT})
target_include_directories(test_lineworld_duo_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_lineworld_duo_smoke m)
target_link_libraries(test_lineworld_duo_smoke dl)
if (TFRL_PERF_FLAGS)
    target_compile_options(test_lineworld_duo_smoke PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
endif()
add_test(NAME lineworld_duo_smoke COMMAND test_lineworld_duo_smoke)

add_executable(test_coin_maze_duo_smoke tests/test_coin_maze_duo_smoke.c ${SRC_ENVS} ${SRC_ENV_SUPPORT})
target_include_directories(test_coin_maze_duo_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_coin_maze_duo_smoke m)
target_link_libraries(test_coin_maze_duo_smoke dl)
if (TFRL_PERF_FLAGS)
    target_compile_options(test_coin_maze_duo_smoke PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
endif()
add_test(NAME coin_maze_duo_smoke COMMAND test_coin_maze_duo_smoke)

add_executable(test_public_headers tests/test_public_headers.c)
target_link_libraries(test_public_headers PRIVATE tfrl_public_api)
add_test(NAME public_headers COMMAND test_public_headers)

add_executable(test_nca_smoke tests/test_nca_smoke.c src/core/algo_nca.c ${SRC_ENVS} ${SRC_ENV_SUPPORT})
target_include_directories(test_nca_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_nca_smoke m)
target_link_libraries(test_nca_smoke dl)
add_test(NAME nca_smoke COMMAND test_nca_smoke)

add_executable(test_env_modules tests/test_env_modules.c)
target_include_directories(test_env_modules PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_env_modules dl m)
add_dependencies(test_env_modules tfrl_env_lineworld tfrl_env_maze_rooms tfrl_env_coin_maze)
add_test(NAME env_modules COMMAND test_env_modules)

add_executable(test_env_dynamic_load tests/test_env_dynamic_load.c ${SRC_ENVS} ${SRC_ENV_SUPPORT})
target_include_directories(test_env_dynamic_load PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_env_dynamic_load dl m)
add_dependencies(test_env_dynamic_load tfrl_env_lineworld tfrl_env_maze_rooms tfrl_env_coin_maze)
add_test(NAME env_dynamic_load COMMAND test_env_dynamic_load)

install(DIRECTORY include/tfrl DESTINATION include)
install(TARGETS tinyfin-rl tfrl_env tfrl_env_lineworld tfrl_env_maze_rooms tfrl_env_coin_maze
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)
