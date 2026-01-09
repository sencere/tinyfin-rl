CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -I../..
LDFLAGS ?=
LDLIBS ?=

APP := first_env
SRC := main.c
TRAIN_APP := train_q
TRAIN_SRC := train_q.c
PLUGIN_LIB := libfirst_env.so
PLUGIN_SRC := plugin_grid.c

RAYLIB_DIR ?= ../../raylib-src

RAYLIB_CFLAGS := $(shell pkg-config --cflags raylib 2>/dev/null)
RAYLIB_LIBS := $(shell pkg-config --libs raylib 2>/dev/null)
RAYLIB_BUILD_DEPS :=
UNAME_S := $(shell uname -s 2>/dev/null)

ifeq ($(strip $(RAYLIB_LIBS)),)
RAYLIB_SRC_LIB := $(RAYLIB_DIR)/src/libraylib.a
RAYLIB_CFLAGS := -I$(RAYLIB_DIR)/src -I$(RAYLIB_DIR)/src/external
RAYLIB_LIBS := -L$(RAYLIB_DIR)/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
RAYLIB_BUILD_DEPS := $(RAYLIB_SRC_LIB)
else
ifeq ($(UNAME_S),Linux)
RAYLIB_LIBS := $(RAYLIB_LIBS) -lm -ldl -lpthread -lX11 -lrt
endif
endif

.PHONY: all clean raylib

all: $(APP) $(TRAIN_APP) $(PLUGIN_LIB)

$(APP): $(SRC) $(RAYLIB_BUILD_DEPS)
	$(CC) $(CFLAGS) $(RAYLIB_CFLAGS) -o $@ $(SRC) $(LDFLAGS) $(RAYLIB_LIBS)

$(TRAIN_APP): $(TRAIN_SRC)
	$(CC) $(CFLAGS) -o $@ $(TRAIN_SRC) -lm

$(PLUGIN_LIB): $(PLUGIN_SRC)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(PLUGIN_SRC)

$(RAYLIB_SRC_LIB):
ifeq ($(strip $(shell pkg-config --libs raylib 2>/dev/null)),)
	$(MAKE) -C $(RAYLIB_DIR)/src
endif

clean:
	rm -f $(APP) $(TRAIN_APP) $(PLUGIN_LIB)
