CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -I../..
LDFLAGS ?=
LDLIBS ?=

APP := maze_rooms
SRC := view.c
TRAIN_APP := train_q
PLUGIN_LIB := libmaze_rooms.so
PLUGIN_SRC := plugin_maze_rooms.c
RENDER_LIB := libmaze_rooms_render.so
RENDER_SRC := render_maze_rooms.c

RAYLIB_DIR ?= ../../raylib-src
RAYLIB_CFLAGS := $(shell pkg-config --cflags raylib 2>/dev/null)
RAYLIB_LIBS := $(shell pkg-config --libs raylib 2>/dev/null)
RAYLIB_LIBDIR := $(shell pkg-config --variable=libdir raylib 2>/dev/null)
RAYLIB_SHARED := $(shell test -n "$(RAYLIB_LIBDIR)" && test -f "$(RAYLIB_LIBDIR)/libraylib.so" && echo 1)
RAYLIB_BUILD_DEPS :=
RAYLIB_RPATH :=
UNAME_S := $(shell uname -s 2>/dev/null)

ifneq ($(strip $(RAYLIB_SHARED)),1)
RAYLIB_LIBS :=
RAYLIB_CFLAGS :=
endif

ifeq ($(strip $(RAYLIB_LIBS)),)
RAYLIB_SRC_LIB := $(RAYLIB_DIR)/src/libraylib.so
RAYLIB_CFLAGS := -I$(RAYLIB_DIR)/src -I$(RAYLIB_DIR)/src/external
RAYLIB_LIBS := -L$(RAYLIB_DIR)/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
RAYLIB_BUILD_DEPS := $(RAYLIB_SRC_LIB)
RAYLIB_RPATH := -Wl,-rpath,$(abspath $(RAYLIB_DIR)/src)
else
ifeq ($(UNAME_S),Linux)
RAYLIB_LIBS := $(RAYLIB_LIBS) -lm -ldl -lpthread -lX11 -lrt
endif
endif

.PHONY: all clean

all: $(APP) $(PLUGIN_LIB) $(RENDER_LIB)

$(APP): $(SRC) $(PLUGIN_SRC) $(RENDER_SRC) $(RAYLIB_BUILD_DEPS)
	$(CC) $(CFLAGS) $(RAYLIB_CFLAGS) -o $@ $(SRC) $(PLUGIN_SRC) $(RENDER_SRC) $(LDFLAGS) $(RAYLIB_LIBS) $(RAYLIB_RPATH)

$(PLUGIN_LIB): $(PLUGIN_SRC)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(PLUGIN_SRC)

$(RENDER_LIB): $(RENDER_SRC) $(RAYLIB_BUILD_DEPS)
	$(CC) $(CFLAGS) $(RAYLIB_CFLAGS) -fPIC -shared -o $@ $(RENDER_SRC) $(LDFLAGS) $(RAYLIB_LIBS) $(RAYLIB_RPATH)

$(RAYLIB_SRC_LIB):
ifneq ($(strip $(RAYLIB_SHARED)),1)
	$(MAKE) -C $(RAYLIB_DIR)/src RAYLIB_LIBTYPE=SHARED
endif

clean:
	rm -f $(APP) $(PLUGIN_LIB) $(RENDER_LIB)
