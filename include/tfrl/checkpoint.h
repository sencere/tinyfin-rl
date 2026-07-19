#ifndef TFRL_PUBLIC_CHECKPOINT_H
#define TFRL_PUBLIC_CHECKPOINT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define TFRL_CHECKPOINT_MANIFEST_VERSION 2u

typedef struct {
    uint32_t manifest_version;
    const char *algorithm;
    uint32_t algorithm_version;
    const char *environment;
    const char *environment_spec_hash;
    uint64_t training_step;
    uint64_t rng_state;
    const char *policy_path;
    const char *target_policy_path;
    const char *optimizer_path;
    const char *replay_meta_path;
} tfrl_checkpoint_manifest;

#ifdef __cplusplus
}
#endif

#endif
