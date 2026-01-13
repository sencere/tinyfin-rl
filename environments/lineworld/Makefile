CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -I../..
LDFLAGS ?=
LDLIBS ?=

APP := lineworld
SRC := main.c
VIEW_APP := view
VIEW_SRC := view.c
TRAIN_APP := train_q
TRAIN_SRC := train_q.c
PLUGIN_LIB := liblineworld.so
PLUGIN_SRC := plugin_lineworld.c
RENDER_LIB := liblineworld_render.so
RENDER_SRC := render_lineworld.c

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

all: $(APP) $(VIEW_APP) $(TRAIN_APP) $(PLUGIN_LIB) $(RENDER_LIB)

$(APP): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS) $(LDLIBS)

$(VIEW_APP): $(VIEW_SRC) $(PLUGIN_SRC) $(RENDER_SRC) $(RAYLIB_BUILD_DEPS)
	$(CC) $(CFLAGS) $(RAYLIB_CFLAGS) -o $@ $(VIEW_SRC) $(PLUGIN_SRC) $(RENDER_SRC) $(LDFLAGS) $(RAYLIB_LIBS) $(RAYLIB_RPATH)

$(TRAIN_APP): $(TRAIN_SRC)
	$(CC) $(CFLAGS) -o $@ $(TRAIN_SRC) -lm

$(PLUGIN_LIB): $(PLUGIN_SRC)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(PLUGIN_SRC)

$(RENDER_LIB): $(RENDER_SRC) $(RAYLIB_BUILD_DEPS)
	$(CC) $(CFLAGS) $(RAYLIB_CFLAGS) -fPIC -shared -o $@ $(RENDER_SRC) $(LDFLAGS) $(RAYLIB_LIBS) $(RAYLIB_RPATH)

$(RAYLIB_SRC_LIB):
ifneq ($(strip $(RAYLIB_SHARED)),1)
	$(MAKE) -C $(RAYLIB_DIR)/src RAYLIB_LIBTYPE=SHARED
endif

clean:
	rm -f $(APP) $(VIEW_APP) $(TRAIN_APP) $(PLUGIN_LIB) $(RENDER_LIB)
