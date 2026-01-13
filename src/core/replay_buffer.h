#ifndef TFRL_REPLAY_BUFFER_H
#define TFRL_REPLAY_BUFFER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "algo_api.h"

typedef struct tfrl_replay_buffer tfrl_replay_buffer;

tfrl_replay_buffer *tfrl_replay_create(int capacity, float alpha);
void tfrl_replay_free(tfrl_replay_buffer *buf);
void tfrl_replay_push(tfrl_replay_buffer *buf, const tfrl_transition *transition, float priority);
int tfrl_replay_sample(tfrl_replay_buffer *buf, int batch, int *out_idx, float *out_weights, float beta);
void tfrl_replay_update_priority(tfrl_replay_buffer *buf, int idx, float priority);
const tfrl_transition *tfrl_replay_get(const tfrl_replay_buffer *buf, int idx);
int tfrl_replay_size(const tfrl_replay_buffer *buf);

#ifdef __cplusplus
}
#endif

#endif
