cmake_minimum_required(VERSION 3.13)
project(tinyfin_rl C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(TFRL_WITH_RAYLIB "Enable raylib viewer" OFF)
set(TINYFIN_DIR "tinyfin" CACHE PATH "Path to tinyfin")

set(SRC_CORE
    src/core/env_example.c
    src/core/algo_factory.c
    src/core/algo_random.c
    src/core/algo_dqn.c
    src/core/algo_reinforce.c
    src/core/algo_ppo.c
    src/core/trace.c
)

set(SRC_APP
    src/main.c
    src/cli/cli.c
    src/runner/runner.c
)

if (TFRL_WITH_RAYLIB)
    set(SRC_VIEWER src/viewer/viewer_raylib.c)
else()
    set(SRC_VIEWER src/viewer/viewer_stub.c)
endif()

add_executable(tinyfin-rl ${SRC_CORE} ${SRC_APP} ${SRC_VIEWER})
target_include_directories(tinyfin-rl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(tinyfin-rl PRIVATE ${TINYFIN_DIR}/include)

find_library(TINYFIN_LIB tinyfin PATHS ${TINYFIN_DIR} NO_DEFAULT_PATH)
if (NOT TINYFIN_LIB)
    set(TINYFIN_LIB ${TINYFIN_DIR}/libtinyfin.so)
endif()
target_link_libraries(tinyfin-rl ${TINYFIN_LIB} m)

if (TFRL_WITH_RAYLIB)
    target_include_directories(tinyfin-rl PRIVATE raylib-src/src)
    target_link_libraries(tinyfin-rl raylib)
endif()
