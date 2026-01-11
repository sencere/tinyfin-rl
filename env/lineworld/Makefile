CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -I../..
LDFLAGS ?=
LDLIBS ?=

APP := lineworld
SRC := main.c
TRAIN_APP := train_q
TRAIN_SRC := train_q.c
PLUGIN_LIB := liblineworld.so
PLUGIN_SRC := plugin_lineworld.c

.PHONY: all clean

all: $(APP) $(TRAIN_APP) $(PLUGIN_LIB)

$(APP): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS) $(LDLIBS)

$(TRAIN_APP): $(TRAIN_SRC)
	$(CC) $(CFLAGS) -o $@ $(TRAIN_SRC) -lm

$(PLUGIN_LIB): $(PLUGIN_SRC)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(PLUGIN_SRC)

clean:
	rm -f $(APP) $(TRAIN_APP) $(PLUGIN_LIB)
