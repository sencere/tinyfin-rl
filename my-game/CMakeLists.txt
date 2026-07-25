cmake_minimum_required(VERSION 3.24)

project(my_game LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(FetchContent)

# Do not build raylib's example programs.
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    raylib
    URL https://github.com/raysan5/raylib/archive/refs/tags/6.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
)

FetchContent_MakeAvailable(raylib)

add_executable(my_game
    src/main.c
)

target_link_libraries(my_game PRIVATE raylib)

# Matches the official raylib game template's macOS setup.
if(APPLE)
    target_link_libraries(my_game PRIVATE
        "-framework IOKit"
        "-framework Cocoa"
        "-framework OpenGL"
    )
endif()
