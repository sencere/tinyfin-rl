#ifndef TINYFIN_RL_ENV_INFO_H
#define TINYFIN_RL_ENV_INFO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define TFRL_ENV_INFO_API_VERSION 0x0001

typedef enum {
    TFRL_SPACE_UNKNOWN = 0,
    TFRL_SPACE_DISCRETE = 1,
    TFRL_SPACE_BOX = 2
} tfrl_space_type;

typedef struct {
    uint32_t api_version;
    const char *name;
    tfrl_space_type observation_type;
    tfrl_space_type action_type;
    int64_t observation_n;
    int64_t action_n;
    int64_t observation_dims;
    int64_t action_dims;
    double observation_low;
    double observation_high;
    double action_low;
    double action_high;
} tfrl_env_metadata;

// Optional symbol provided by env plugins for space introspection.
const tfrl_env_metadata *tfrl_env_metadata_get(void);

#ifdef __cplusplus
}
#endif

#endif
