CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra
CFLAGS += -Iinclude
PERF ?= 0

ifeq ($(PERF),1)
CFLAGS += -O3 -march=native -ffast-math -fno-math-errno
endif

TINYFIN_DIR ?= tinyfin
TINYFIN_INC ?= $(TINYFIN_DIR)/include
TINYFIN_LIB ?= $(TINYFIN_DIR)/libtinyfin.so
TINYFIN_RPATH ?= -Wl,-rpath,$(abspath $(TINYFIN_DIR))

USE_RAYLIB ?= 0
RAYLIB_MODE ?= vendored
RAYLIB_DIR ?= raylib-src
RAYLIB_INC ?= $(RAYLIB_DIR)/src
RAYLIB_LIB ?= $(RAYLIB_DIR)/src/libraylib.so
RAYLIB_LDFLAGS ?= -L$(RAYLIB_DIR)/src -lraylib -lm -ldl -lpthread -lX11 -lrt -Wl,-rpath,$(abspath $(RAYLIB_DIR)/src)
RAYLIB_CFLAGS ?= -I$(RAYLIB_INC)

ifeq ($(RAYLIB_MODE),system)
RAYLIB_CFLAGS = $(shell pkg-config --cflags raylib)
RAYLIB_LDFLAGS = $(shell pkg-config --libs raylib)
endif

ENV_SRCS = \
	src/envs/envs.c \
	src/envs/registry.c \
	envs/maze_rooms/maze.c \
	envs/maze_rooms/render.c \
	envs/lineworld/lineworld.c \
	envs/lineworld/render.c \
	src/envs/point1d/point1d.c \
	src/envs/point1d/render.c \
	envs/coin_maze/coin_maze.c \
	envs/coin_maze/render.c \
	src/envs/snake/snake_env.c \
	src/envs/snake/render.c \
	src/envs/floppy/floppy_env.c \
	src/envs/floppy/render.c \
	envs/tetris/tetris_env.c \
	envs/tetris/render.c \
	src/envs/breakout/breakout_env.c \
	src/envs/breakout/render.c \
	src/envs/breakout_atari_env.c \
	src/envs/breakout_atari_render.c \
	src/envs/seq_pixels_env.c \
	src/envs/pang/pang_env.c \
	src/envs/pang/render.c \
	src/envs/py_bridge/py_bridge.c \
	src/envs/render.c \
	src/envs/render_grid.c

ENV_SUPPORT_SRCS = \
	src/core/obs_layout.c

ENV_MODULE_CORE_SRCS = \
	src/envs/envs.c \
	src/envs/registry.c \
	src/envs/render.c \
	src/envs/render_grid.c \
	$(ENV_SUPPORT_SRCS)

ENV_MODULE_LINEWORLD_SRCS = \
	$(ENV_MODULE_CORE_SRCS) \
	envs/lineworld/lineworld.c \
	envs/lineworld/render.c

ENV_MODULE_MAZE_ROOMS_SRCS = \
	$(ENV_MODULE_CORE_SRCS) \
	envs/maze_rooms/maze.c \
	envs/maze_rooms/render.c

ENV_MODULE_COIN_MAZE_SRCS = \
	$(ENV_MODULE_CORE_SRCS) \
	envs/coin_maze/coin_maze.c \
	envs/coin_maze/render.c

SRC_LIB = \
	$(ENV_SRCS) \
	$(ENV_SUPPORT_SRCS) \
	src/core/algo_factory.c \
	src/core/algo_random.c \
	src/core/algo_dqn.c \
	src/core/algo_nca.c \
	src/core/algo_reinforce.c \
	src/core/algo_ppo.c \
	src/core/replay_buffer.c \
	src/core/trace.c

SRC_CORE = $(SRC_LIB)

SRC_APP = \
	src/main.c \
	src/cli/cli.c \
	src/runner/runner.c

SRC_VIEWER_STUB = src/envs/viewer_stub.c
SRC_VIEWER_RAYLIB = src/envs/viewer_raylib.c

BIN_DIR ?= build
BIN ?= $(BIN_DIR)/tinyfin-rl
PROD_BIN ?= $(BIN_DIR)/tinyfin-prod
ENV_LIB ?= $(BIN_DIR)/libtfrl_env.so
ENV_LIB_LINEWORLD ?= $(BIN_DIR)/libtfrl_env_lineworld.so
ENV_LIB_MAZE_ROOMS ?= $(BIN_DIR)/libtfrl_env_maze_rooms.so
ENV_LIB_COIN_MAZE ?= $(BIN_DIR)/libtfrl_env_coin_maze.so
ENV_LIB_TETRIS ?= $(BIN_DIR)/libtfrl_env_tetris.so
TEST_REPLAY ?= $(BIN_DIR)/test_replay_buffer
TEST_POINT1D ?= $(BIN_DIR)/test_point1d_smoke
TEST_MAZE_ROOMS ?= $(BIN_DIR)/test_maze_rooms_smoke
TEST_LINEWORLD_DUO ?= $(BIN_DIR)/test_lineworld_duo_smoke
TEST_COIN_MAZE_DUO ?= $(BIN_DIR)/test_coin_maze_duo_smoke
TEST_CORE_ENVS ?= $(BIN_DIR)/test_core_env_smoke
TEST_TRACE_REPLAY ?= $(BIN_DIR)/test_trace_replay
TEST_PUBLIC_HEADERS ?= $(BIN_DIR)/test_public_headers
TEST_NCA ?= $(BIN_DIR)/test_nca_smoke
TEST_ENV_MODULES ?= $(BIN_DIR)/test_env_modules
TEST_ENV_DYNAMIC_LOAD ?= $(BIN_DIR)/test_env_dynamic_load

.PHONY: all clean raylib

all: $(BIN) $(ENV_LIB) $(ENV_LIB_LINEWORLD) $(ENV_LIB_MAZE_ROOMS) $(ENV_LIB_COIN_MAZE) $(ENV_LIB_TETRIS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

ifeq ($(USE_RAYLIB),1)
VIEWER_SRC = $(SRC_VIEWER_RAYLIB)
VIEWER_INC = $(RAYLIB_CFLAGS)
VIEWER_LIBS = $(RAYLIB_LDFLAGS)
VIEWER_DEFS = -DUSE_RAYLIB
else
VIEWER_SRC = $(SRC_VIEWER_STUB)
VIEWER_INC =
VIEWER_LIBS =
VIEWER_DEFS =
endif

$(BIN): $(SRC_CORE) $(SRC_APP) $(VIEWER_SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(VIEWER_DEFS) -I$(TINYFIN_INC) -Isrc $(VIEWER_INC) \
		-o $@ $(SRC_CORE) $(SRC_APP) $(VIEWER_SRC) \
		-L$(TINYFIN_DIR) -ltinyfin $(TINYFIN_RPATH) $(VIEWER_LIBS) -lpthread -ldl -lm -lrt

$(PROD_BIN): $(SRC_LIB) src/prod/prod_main.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(TINYFIN_INC) -Isrc \
		-o $@ $(SRC_LIB) src/prod/prod_main.c \
		-L$(TINYFIN_DIR) -ltinyfin $(TINYFIN_RPATH) -lpthread -ldl -lm -lrt

$(ENV_LIB): $(ENV_SRCS) $(ENV_SUPPORT_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -fPIC -shared -Isrc -o $@ $^ -ldl

$(ENV_LIB_LINEWORLD): $(ENV_MODULE_LINEWORLD_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -DTFRL_ENV_MODULE_LINEWORLD -DTFRL_ENV_NO_PYBRIDGE -DTFRL_ENV_NO_DYNAMIC -fPIC -shared -Isrc -o $@ $^

$(ENV_LIB_MAZE_ROOMS): $(ENV_MODULE_MAZE_ROOMS_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -DTFRL_ENV_MODULE_MAZE_ROOMS -DTFRL_ENV_NO_PYBRIDGE -DTFRL_ENV_NO_DYNAMIC -fPIC -shared -Isrc -o $@ $^

$(ENV_LIB_COIN_MAZE): $(ENV_MODULE_COIN_MAZE_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -DTFRL_ENV_MODULE_COIN_MAZE -DTFRL_ENV_NO_PYBRIDGE -DTFRL_ENV_NO_DYNAMIC -fPIC -shared -Isrc -o $@ $^

$(ENV_LIB_TETRIS): $(ENV_MODULE_CORE_SRCS) envs/tetris/tetris_env.c envs/tetris/render.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -DTFRL_ENV_MODULE_TETRIS -DTFRL_ENV_NO_PYBRIDGE -DTFRL_ENV_NO_DYNAMIC -fPIC -shared -Isrc -o $@ $^

tests: $(TEST_REPLAY) $(TEST_POINT1D) $(TEST_MAZE_ROOMS) $(TEST_LINEWORLD_DUO) $(TEST_COIN_MAZE_DUO) $(TEST_CORE_ENVS) $(TEST_TRACE_REPLAY) $(TEST_PUBLIC_HEADERS) $(TEST_NCA) $(TEST_ENV_MODULES) $(TEST_ENV_DYNAMIC_LOAD)

$(TEST_REPLAY): tests/test_replay_buffer.c src/core/replay_buffer.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -Isrc -o $@ $^ -lm

$(TEST_POINT1D): tests/test_point1d_smoke.c $(ENV_SRCS) $(ENV_SUPPORT_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -Isrc -o $@ $^ -ldl -lm

$(TEST_MAZE_ROOMS): tests/test_maze_rooms_smoke.c $(ENV_SRCS) $(ENV_SUPPORT_SRCS) $(ENV_LIB_MAZE_ROOMS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -Isrc -o $@ tests/test_maze_rooms_smoke.c $(ENV_SRCS) $(ENV_SUPPORT_SRCS) -ldl -lm

$(TEST_LINEWORLD_DUO): tests/test_lineworld_duo_smoke.c $(ENV_SRCS) $(ENV_SUPPORT_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -Isrc -o $@ $^ -ldl -lm

$(TEST_COIN_MAZE_DUO): tests/test_coin_maze_duo_smoke.c $(ENV_SRCS) $(ENV_SUPPORT_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -Isrc -o $@ $^ -ldl -lm

$(TEST_CORE_ENVS): tests/test_core_env_smoke.c $(ENV_SRCS) $(ENV_SUPPORT_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -Isrc -o $@ $^ -ldl -lm

$(TEST_TRACE_REPLAY): tests/test_trace_replay.c src/core/trace.c $(ENV_SRCS) $(ENV_SUPPORT_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -Isrc -o $@ $^ -ldl -lm

$(TEST_PUBLIC_HEADERS): tests/test_public_headers.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^

$(TEST_NCA): tests/test_nca_smoke.c src/core/algo_nca.c $(ENV_SRCS) $(ENV_SUPPORT_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -Isrc -o $@ $^ -ldl -lm

$(TEST_ENV_MODULES): tests/test_env_modules.c $(ENV_LIB_LINEWORLD) $(ENV_LIB_MAZE_ROOMS) $(ENV_LIB_COIN_MAZE) | $(BIN_DIR)
	$(CC) $(CFLAGS) -Isrc -o $@ tests/test_env_modules.c -ldl -lm

$(TEST_ENV_DYNAMIC_LOAD): tests/test_env_dynamic_load.c $(ENV_SRCS) $(ENV_SUPPORT_SRCS) $(ENV_LIB_LINEWORLD) $(ENV_LIB_MAZE_ROOMS) $(ENV_LIB_COIN_MAZE) | $(BIN_DIR)
	$(CC) $(CFLAGS) -Isrc -o $@ tests/test_env_dynamic_load.c $(ENV_SRCS) $(ENV_SUPPORT_SRCS) -ldl -lm

raylib:
	$(MAKE) -C $(RAYLIB_DIR)/src

clean:
	rm -rf $(BIN_DIR)
