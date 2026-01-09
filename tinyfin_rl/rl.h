#ifndef TINYFIN_RL_RL_H
#define TINYFIN_RL_RL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define TFRL_API_VERSION 0x0001

typedef enum {
    TFRL_OK = 0,
    TFRL_ERR = 1,
    TFRL_ERR_INVALID = 2,
    TFRL_ERR_NOOP = 3
} tfrl_status;

typedef enum {
    TFRL_VALUE_EMPTY = 0,
    TFRL_VALUE_I64 = 1,
    TFRL_VALUE_F64 = 2,
    TFRL_VALUE_BUFFER = 3
} tfrl_value_type;

typedef struct {
    void *data;
    size_t len;
    size_t elem_size;
} tfrl_buffer;

typedef struct {
    tfrl_value_type type;
    union {
        int64_t i64;
        double f64;
        tfrl_buffer buffer;
    } as;
} tfrl_value;

typedef struct {
    tfrl_value observation;
    double reward;
    int terminated;
    int truncated;
} tfrl_step;

static inline int tfrl_step_done(const tfrl_step *step) {
    return step->terminated || step->truncated;
}

typedef struct {
    tfrl_value observation;
    tfrl_value action;
    double reward;
    tfrl_value next_observation;
    int terminated;
    int truncated;
} tfrl_transition;

typedef struct tfrl_env_vtable {
    int (*reset)(void *ctx, tfrl_value *out_obs);
    int (*step)(void *ctx, const tfrl_value *action, tfrl_step *out_step);
    int (*seed)(void *ctx, uint64_t seed);
    void (*destroy)(void *ctx);
} tfrl_env_vtable;

typedef struct {
    void *ctx;
    const tfrl_env_vtable *vtable;
} tfrl_env;

static inline int tfrl_env_reset(tfrl_env *env, tfrl_value *out_obs) {
    return env->vtable->reset(env->ctx, out_obs);
}

static inline int tfrl_env_step(tfrl_env *env, const tfrl_value *action, tfrl_step *out_step) {
    return env->vtable->step(env->ctx, action, out_step);
}

static inline int tfrl_env_seed(tfrl_env *env, uint64_t seed) {
    if (!env->vtable->seed) {
        return TFRL_ERR_NOOP;
    }
    return env->vtable->seed(env->ctx, seed);
}

static inline void tfrl_env_destroy(tfrl_env *env) {
    if (env->vtable->destroy) {
        env->vtable->destroy(env->ctx);
    }
}

typedef struct tfrl_agent_vtable {
    int (*act)(void *ctx, const tfrl_value *obs, tfrl_value *out_action);
    int (*observe)(void *ctx, const tfrl_transition *transition);
    void (*destroy)(void *ctx);
} tfrl_agent_vtable;

typedef struct {
    void *ctx;
    const tfrl_agent_vtable *vtable;
} tfrl_agent;

static inline int tfrl_agent_act(tfrl_agent *agent, const tfrl_value *obs, tfrl_value *out_action) {
    return agent->vtable->act(agent->ctx, obs, out_action);
}

static inline int tfrl_agent_observe(tfrl_agent *agent, const tfrl_transition *transition) {
    if (!agent->vtable->observe) {
        return TFRL_ERR_NOOP;
    }
    return agent->vtable->observe(agent->ctx, transition);
}

static inline void tfrl_agent_destroy(tfrl_agent *agent) {
    if (agent->vtable->destroy) {
        agent->vtable->destroy(agent->ctx);
    }
}

typedef struct tfrl_trainer_vtable {
    int (*train)(void *ctx, int episodes);
    void (*destroy)(void *ctx);
} tfrl_trainer_vtable;

typedef struct {
    void *ctx;
    const tfrl_trainer_vtable *vtable;
} tfrl_trainer;

static inline int tfrl_trainer_train(tfrl_trainer *trainer, int episodes) {
    return trainer->vtable->train(trainer->ctx, episodes);
}

static inline void tfrl_trainer_destroy(tfrl_trainer *trainer) {
    if (trainer->vtable->destroy) {
        trainer->vtable->destroy(trainer->ctx);
    }
}

#ifdef __cplusplus
}
#endif

#endif
