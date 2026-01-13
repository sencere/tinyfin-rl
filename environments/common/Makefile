CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -I../..
LDFLAGS ?=
LDLIBS ?= -ldl

APP := plugin_viewer
SRC := plugin_viewer.c

.PHONY: all clean

all: $(APP)

$(APP): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(APP)
