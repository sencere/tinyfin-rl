CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -I.
LDLIBS ?= -lm
BUILD_DIR ?= build

.PHONY: all clean c_api_smoke bandit_reinforce a2c_bandit ppo_bandit trainer_smoke env_plugin_loader

all: c_api_smoke bandit_reinforce a2c_bandit ppo_bandit trainer_smoke env_plugin_loader

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/c_api_smoke: tests/c_api_smoke.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD_DIR)/bandit_reinforce: examples/c/bandit_reinforce.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

$(BUILD_DIR)/a2c_bandit: examples/c/a2c_bandit.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

$(BUILD_DIR)/ppo_bandit: examples/c/ppo_bandit.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

$(BUILD_DIR)/trainer_smoke: examples/c/trainer_smoke.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

$(BUILD_DIR)/env_plugin_loader: examples/c/env_plugin_loader.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< -ldl

c_api_smoke: $(BUILD_DIR)/c_api_smoke

bandit_reinforce: $(BUILD_DIR)/bandit_reinforce

a2c_bandit: $(BUILD_DIR)/a2c_bandit

ppo_bandit: $(BUILD_DIR)/ppo_bandit

trainer_smoke: $(BUILD_DIR)/trainer_smoke

env_plugin_loader: $(BUILD_DIR)/env_plugin_loader

clean:
	rm -rf $(BUILD_DIR)
