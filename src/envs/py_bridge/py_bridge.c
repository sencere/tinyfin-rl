#define _POSIX_C_SOURCE 200809L

#include "envs/envs_internal.h"

#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

int tfrl_env_py_connect(tfrl_env *env, const char *socket_path, const char *kind, const char *env_id, uint64_t seed) {
    if (!env || !socket_path || !kind || !env_id) return 0;
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return 0;
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return 0;
    }
    FILE *fp = fdopen(fd, "r+");
    if (!fp) {
        close(fd);
        return 0;
    }
    setvbuf(fp, NULL, _IOLBF, 0);
    fprintf(fp, "INIT %s %s %llu\n", kind, env_id, (unsigned long long)seed);
    char line[1024];
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return 0;
    }
    int obs_type = 0;
    int obs_n = 0;
    int obs_dims = 0;
    int action_type = 0;
    int action_n = 0;
    int action_dims = 0;
    double obs_low = 0.0;
    double obs_high = 0.0;
    double action_low = 0.0;
    double action_high = 0.0;
    int max_steps = 0;
    int agent_count = 0;
    if (sscanf(line, "OK SPEC %d %d %d %d %d %d %lf %lf %lf %lf %d %d",
               &obs_type, &obs_n, &obs_dims, &action_type, &action_n, &action_dims,
               &obs_low, &obs_high, &action_low, &action_high, &max_steps, &agent_count) != 12) {
        fclose(fp);
        return 0;
    }
    env->py_fd = fd;
    env->py_fp = fp;
    env->py_agent_count = agent_count > 0 ? agent_count : 1;
    snprintf(env->py_name, sizeof(env->py_name), "py:%s:%s", kind, env_id);
    env->py_spec.name = env->py_name;
    env->py_spec.obs_n = obs_n;
    env->py_spec.action_n = action_n;
    env->py_spec.max_steps = max_steps;
    env->py_spec.width = 0;
    env->py_spec.height = 0;
    env->py_spec.obs_type = (tfrl_space_type)obs_type;
    env->py_spec.action_type = (tfrl_space_type)action_type;
    env->py_spec.obs_dims = obs_dims;
    env->py_spec.action_dims = action_dims;
    env->py_spec.obs_shape[0] = obs_type == TFRL_SPACE_DISCRETE ? obs_n : obs_dims;
    env->py_spec.obs_shape[1] = 0;
    env->py_spec.action_shape[0] = action_type == TFRL_SPACE_DISCRETE ? action_n : action_dims;
    env->py_spec.action_shape[1] = 0;
    env->py_spec.obs_dtype = obs_type == TFRL_SPACE_DISCRETE ? TFRL_DTYPE_INT32 : TFRL_DTYPE_FLOAT32;
    env->py_spec.action_dtype = action_type == TFRL_SPACE_DISCRETE ? TFRL_DTYPE_INT32 : TFRL_DTYPE_FLOAT32;
    env->py_spec.obs_low = obs_low;
    env->py_spec.obs_high = obs_high;
    env->py_spec.action_low = action_low;
    env->py_spec.action_high = action_high;
    env->py_spec.agent_count = env->py_agent_count;
    env->py_spec.obs_layout = NULL;
    return 1;
}

void tfrl_env_py_close(tfrl_env *env) {
    if (!env || !env->py_fp) return;
    fprintf(env->py_fp, "CLOSE\n");
    fclose(env->py_fp);
    env->py_fp = NULL;
    env->py_fd = -1;
}

static int py_read_line(FILE *fp, char *buf, size_t buf_len) {
    if (!fp || !buf || buf_len == 0) return 0;
    if (!fgets(buf, buf_len, fp)) return 0;
    return 1;
}

static int py_parse_tokens(char *line, char **out, int max) {
    int count = 0;
    char *tok = strtok(line, " \t\r\n");
    while (tok && count < max) {
        out[count++] = tok;
        tok = strtok(NULL, " \t\r\n");
    }
    return count;
}

int tfrl_env_py_reset_multi(tfrl_env *env, uint64_t seed, tfrl_obs *out_obs, int max_agents) {
    if (!env || !out_obs || max_agents <= 0) return 0;
    if (!env->py_fp) return 0;
    fprintf(env->py_fp, "RESET %llu\n", (unsigned long long)seed);
    char line[2048];
    if (!py_read_line(env->py_fp, line, sizeof(line))) return 0;
    char *tokens[256];
    int count = py_parse_tokens(line, tokens, 256);
    if (count < 4) return 0;
    if (strcmp(tokens[0], "OK") != 0 || strcmp(tokens[1], "RESET") != 0) return 0;
    int agent_count = atoi(tokens[2]);
    int obs_type = atoi(tokens[3]);
    int expected = obs_type == TFRL_SPACE_DISCRETE ? agent_count : agent_count * env->py_spec.obs_dims;
    if (count < 4 + expected) return 0;
    if (agent_count > max_agents) return 0;
    int idx = 4;
    for (int i = 0; i < agent_count; i++) {
        if (obs_type == TFRL_SPACE_DISCRETE) {
            out_obs[i].index = atoi(tokens[idx++]);
            out_obs[i].data_len = 0;
        } else {
            out_obs[i].data_len = env->py_spec.obs_dims;
            for (int d = 0; d < env->py_spec.obs_dims; d++) {
                out_obs[i].data[d] = (float)atof(tokens[idx++]);
            }
        }
    }
    return agent_count;
}

int tfrl_env_py_step_multi(tfrl_env *env, const tfrl_action *actions, int action_count,
                           tfrl_step_result *out_steps, int max_agents) {
    if (!env || !actions || !out_steps || max_agents <= 0) return 0;
    if (!env->py_fp) return 0;
    int agent_count = env->py_agent_count;
    if (action_count < agent_count || max_agents < agent_count) return 0;
    fprintf(env->py_fp, "STEP %d %d", agent_count, env->py_spec.action_type);
    for (int i = 0; i < agent_count; i++) {
        if (env->py_spec.action_type == TFRL_SPACE_DISCRETE) {
            fprintf(env->py_fp, " %d", actions[i].index);
        } else {
            for (int d = 0; d < env->py_spec.action_dims; d++) {
                float v = actions[i].data[d];
                fprintf(env->py_fp, " %.6f", v);
            }
        }
    }
    fprintf(env->py_fp, "\n");
    char line[4096];
    if (!py_read_line(env->py_fp, line, sizeof(line))) return 0;
    char *tokens[512];
    int count = py_parse_tokens(line, tokens, 512);
    if (count < 5) return 0;
    if (strcmp(tokens[0], "OK") != 0 || strcmp(tokens[1], "STEP") != 0) return 0;
    int resp_agents = atoi(tokens[2]);
    int obs_type = atoi(tokens[3]);
    if (resp_agents != agent_count) return 0;
    int obs_count = obs_type == TFRL_SPACE_DISCRETE ? resp_agents : resp_agents * env->py_spec.obs_dims;
    int rewards_count = resp_agents;
    int dones_count = resp_agents;
    int min_tokens = 4 + obs_count + rewards_count + dones_count;
    if (count < min_tokens) return 0;
    int idx = 4;
    for (int i = 0; i < resp_agents; i++) {
        if (obs_type == TFRL_SPACE_DISCRETE) {
            out_steps[i].observation.index = atoi(tokens[idx++]);
            out_steps[i].observation.data_len = 0;
        } else {
            out_steps[i].observation.data_len = env->py_spec.obs_dims;
            for (int d = 0; d < env->py_spec.obs_dims; d++) {
                out_steps[i].observation.data[d] = (float)atof(tokens[idx++]);
            }
        }
    }
    for (int i = 0; i < resp_agents; i++) {
        out_steps[i].reward = atof(tokens[idx++]);
    }
    for (int i = 0; i < resp_agents; i++) {
        out_steps[i].done = atoi(tokens[idx++]);
    }
    env->last_reward = (float)out_steps[0].reward;
    env->last_done = out_steps[0].done;
    return resp_agents;
}
