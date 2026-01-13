CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra

TINYFIN_DIR ?= tinyfin
TINYFIN_INC ?= $(TINYFIN_DIR)/include
TINYFIN_LIB ?= $(TINYFIN_DIR)/libtinyfin.so
TINYFIN_RPATH ?= -Wl,-rpath,$(abspath $(TINYFIN_DIR))

USE_RAYLIB ?= 0
RAYLIB_DIR ?= raylib-src
RAYLIB_INC ?= $(RAYLIB_DIR)/src
RAYLIB_LIB ?= $(RAYLIB_DIR)/src/libraylib.so
RAYLIB_LDFLAGS ?= -L$(RAYLIB_DIR)/src -lraylib -lm -ldl -lpthread -lX11 -lrt

SRC_CORE = \
	src/core/env_example.c \
	src/core/algo_factory.c \
	src/core/algo_random.c \
	src/core/algo_dqn.c \
	src/core/algo_reinforce.c \
	src/core/algo_ppo.c \
	src/core/trace.c

SRC_APP = \
	src/main.c \
	src/cli/cli.c \
	src/runner/runner.c

SRC_VIEWER_STUB = src/viewer/viewer_stub.c
SRC_VIEWER_RAYLIB = src/viewer/viewer_raylib.c

BIN_DIR ?= build
BIN ?= $(BIN_DIR)/tinyfin-rl

.PHONY: all clean raylib

all: $(BIN)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

ifeq ($(USE_RAYLIB),1)
VIEWER_SRC = $(SRC_VIEWER_RAYLIB)
VIEWER_INC = -I$(RAYLIB_INC)
VIEWER_LIBS = $(RAYLIB_LDFLAGS)
else
VIEWER_SRC = $(SRC_VIEWER_STUB)
VIEWER_INC =
VIEWER_LIBS =
endif

$(BIN): $(SRC_CORE) $(SRC_APP) $(VIEWER_SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(TINYFIN_INC) -Isrc $(VIEWER_INC) \
		-o $@ $(SRC_CORE) $(SRC_APP) $(VIEWER_SRC) \
		-L$(TINYFIN_DIR) -ltinyfin $(TINYFIN_RPATH) $(VIEWER_LIBS)

raylib:
	$(MAKE) -C $(RAYLIB_DIR)/src

clean:
	rm -rf $(BIN_DIR)
