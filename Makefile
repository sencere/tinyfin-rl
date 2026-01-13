CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -I.
LDLIBS ?= -lm
TINYFIN_DIR ?= tinyfin
TINYFIN_INC ?= $(TINYFIN_DIR)/include
TINYFIN_LIB ?= $(TINYFIN_DIR)/libtinyfin.so
TINYFIN_RPATH ?= -Wl,-rpath,$(abspath $(TINYFIN_DIR))
BUILD_DIR ?= build

.PHONY: all clean c_api_smoke trainer_smoke env_plugin_loader train_dqn train_ppo play_dqn tfrl_train

all: c_api_smoke trainer_smoke env_plugin_loader train_dqn train_ppo play_dqn tfrl_train

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/c_api_smoke: tests/c_api_smoke.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD_DIR)/trainer_smoke: examples/c/trainer_smoke.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

$(BUILD_DIR)/env_plugin_loader: examples/c/env_plugin_loader.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< -ldl

$(BUILD_DIR)/libtfrl_train.so: tinyfin_rl/rl_train.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(TINYFIN_INC) -fPIC -shared -o $@ $< -L$(TINYFIN_DIR) -ltinyfin $(TINYFIN_RPATH)

$(BUILD_DIR)/train_dqn: examples/c/train_dqn.c $(BUILD_DIR)/libtfrl_train.so | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(TINYFIN_INC) -o $@ $< -L$(BUILD_DIR) -ltfrl_train -L$(TINYFIN_DIR) -ltinyfin -ldl $(TINYFIN_RPATH) -Wl,-rpath,$(abspath $(BUILD_DIR))

$(BUILD_DIR)/train_ppo: examples/c/train_ppo.c $(BUILD_DIR)/libtfrl_train.so | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(TINYFIN_INC) -o $@ $< -L$(BUILD_DIR) -ltfrl_train -L$(TINYFIN_DIR) -ltinyfin -ldl $(TINYFIN_RPATH) -Wl,-rpath,$(abspath $(BUILD_DIR))

$(BUILD_DIR)/play_dqn: examples/c/play_dqn.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(TINYFIN_INC) -o $@ $< -L$(TINYFIN_DIR) -ltinyfin -ldl $(TINYFIN_RPATH)

c_api_smoke: $(BUILD_DIR)/c_api_smoke

trainer_smoke: $(BUILD_DIR)/trainer_smoke

env_plugin_loader: $(BUILD_DIR)/env_plugin_loader

train_dqn: $(BUILD_DIR)/train_dqn

train_ppo: $(BUILD_DIR)/train_ppo

play_dqn: $(BUILD_DIR)/play_dqn

tfrl_train: $(BUILD_DIR)/libtfrl_train.so

clean:
	rm -rf $(BUILD_DIR)
