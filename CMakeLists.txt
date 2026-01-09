cmake_minimum_required(VERSION 3.13)
project(tinyfin_rl_c C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(c_api_smoke tests/c_api_smoke.c)
target_include_directories(c_api_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(bandit_reinforce examples/c/bandit_reinforce.c)
target_include_directories(bandit_reinforce PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bandit_reinforce m)

add_executable(a2c_bandit examples/c/a2c_bandit.c)
target_include_directories(a2c_bandit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(a2c_bandit m)

add_executable(ppo_bandit examples/c/ppo_bandit.c)
target_include_directories(ppo_bandit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ppo_bandit m)

add_executable(trainer_smoke examples/c/trainer_smoke.c)
target_include_directories(trainer_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trainer_smoke m)

add_executable(env_plugin_loader examples/c/env_plugin_loader.c)
target_include_directories(env_plugin_loader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(UNIX)
    target_link_libraries(env_plugin_loader dl)
endif()
