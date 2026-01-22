#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include "runner/runner.h"
#include "core/algo_api.h"
#include "core/env_api.h"
#include "core/trace.h"
#include "envs/viewer.h"
#include "tinyfin/backend.h"
#include "tinyfin/tensor.h"

static void seed_rng(int seed) {
    if (seed <= 0) {
        seed = (int)time(NULL);
    }
    srand((unsigned)seed);
}

static int should_render(int step, int render_every) {
    if (render_every <= 1) return 1;
    return (step % render_every) == 0;
}

static int parse_device_name(const char *name, int *out_device) {
    if (!name || !out_device) return 0;
    if (strcmp(name, "0") == 0 || strcmp(name, "cpu") == 0) {
        *out_device = DEVICE_CPU;
        return 1;
    }
    if (strcmp(name, "1") == 0) {
        *out_device = DEVICE_GPU;
        return 1;
    }
    char buf[16];
    size_t n = strlen(name);
    if (n >= sizeof(buf)) n = sizeof(buf) - 1;
    for (size_t i = 0; i < n; i++) buf[i] = (char)tolower((unsigned char)name[i]);
    buf[n] = '\0';
    if (strcmp(buf, "gpu") == 0 || strcmp(buf, "cuda") == 0) {
        *out_device = DEVICE_GPU;
        return 1;
    }
    if (strcmp(buf, "cpu") == 0) {
        *out_device = DEVICE_CPU;
        return 1;
    }
    return 0;
}

static void configure_tinyfin(const tfrl_runner_config *cfg) {
    if (!cfg) return;
    if (cfg->backend && cfg->backend[0]) {
        if (!backend_set_by_name(cfg->backend)) {
            fprintf(stderr, "unknown backend '%s' (expected cpu|cuda|blas)\n", cfg->backend);
        }
    }
    if (cfg->device && cfg->device[0]) {
        int device = DEVICE_CPU;
        if (!parse_device_name(cfg->device, &device)) {
            fprintf(stderr, "unknown device '%s' (expected cpu|gpu)\n", cfg->device);
        } else {
            tensor_set_default_device(device);
        }
    } else if (cfg->backend && strcmp(cfg->backend, "cuda") == 0) {
        tensor_set_default_device(DEVICE_GPU);
    }
}

static void append_meta_kv(char *buffer, size_t buffer_len, const char *key, const char *value) {
    if (!buffer || buffer_len == 0 || !key || !value) return;
    size_t len = strlen(buffer);
    if (len >= buffer_len - 1) return;
    int wrote = snprintf(buffer + len, buffer_len - len, "%s%s=%s", len ? " " : "", key, value);
    if (wrote < 0) buffer[buffer_len - 1] = '\0';
}

static double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

static void fill_algo_config(tfrl_algo_config *out,
                             const tfrl_runner_config *cfg,
                             const tfrl_env_spec *spec,
                             int seed,
                             const char *load_path) {
    if (!out || !cfg || !spec) return;
    *out = (tfrl_algo_config){
        .name = cfg->algo ? cfg->algo : "dqn",
        .obs_n = spec->obs_n,
        .action_n = spec->action_n,
        .obs_type = spec->obs_type,
        .action_type = spec->action_type,
        .obs_dims = spec->obs_dims,
        .action_dims = spec->action_dims,
        .obs_shape = {spec->obs_shape[0], spec->obs_shape[1]},
        .action_shape = {spec->action_shape[0], spec->action_shape[1]},
        .obs_dtype = spec->obs_dtype,
        .action_dtype = spec->action_dtype,
        .obs_low = spec->obs_low,
        .obs_high = spec->obs_high,
        .action_low = spec->action_low,
        .action_high = spec->action_high,
        .gamma = cfg->gamma,
        .lr = cfg->lr,
        .epsilon = cfg->mode == TFRL_MODE_TRAIN ? cfg->epsilon : 0.0f,
        .entropy_coef = cfg->entropy_coef,
        .clip_eps = cfg->clip_eps,
        .kl_target = cfg->kl_target,
        .gae_lambda = cfg->gae_lambda,
        .replay_size = cfg->replay_size,
        .batch_size = cfg->batch_size,
        .train_every = cfg->train_every,
        .learning_starts = cfg->learning_starts,
        .grad_steps = cfg->grad_steps,
        .per_alpha = cfg->per_alpha,
        .per_beta = cfg->per_beta,
        .mcts_sims = cfg->mcts_sims,
        .mcts_depth = cfg->mcts_depth,
        .steps_per_batch = cfg->steps_per_batch,
        .epochs = cfg->epochs,
        .c51_atoms = cfg->c51_atoms,
        .c51_vmin = cfg->c51_vmin,
        .c51_vmax = cfg->c51_vmax,
        .iqn_quantiles = cfg->iqn_quantiles,
        .iqn_tau_samples = cfg->iqn_tau_samples,
        .save_path = cfg->save_path,
        .load_path = load_path,
        .env_name = cfg->env_name,
        .seed = seed,
        .deterministic = cfg->deterministic,
    };
}

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    pthread_mutex_t step_lock;
    int total_steps;
    int steps_done;
    int capacity;
    int head;
    int tail;
    int count;
    int closed;
    tfrl_transition items[];
} tfrl_mp_queue;

static size_t mp_queue_bytes(int capacity) {
    return sizeof(tfrl_mp_queue) + sizeof(tfrl_transition) * (size_t)capacity;
}

static tfrl_mp_queue *mp_queue_create(int capacity, int total_steps, char *name_out, size_t name_len, int *fd_out) {
    if (capacity <= 0 || !name_out || name_len == 0 || !fd_out) return NULL;
    snprintf(name_out, name_len, "/tfrl_mpq_%ld_%u", (long)getpid(), (unsigned)rand());
    int fd = shm_open(name_out, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) return NULL;
    size_t bytes = mp_queue_bytes(capacity);
    if (ftruncate(fd, (off_t)bytes) != 0) {
        close(fd);
        shm_unlink(name_out);
        return NULL;
    }
    void *mem = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED) {
        close(fd);
        shm_unlink(name_out);
        return NULL;
    }
    tfrl_mp_queue *q = (tfrl_mp_queue *)mem;
    memset(q, 0, bytes);
    q->capacity = capacity;
    q->total_steps = total_steps;
    q->steps_done = 0;
    q->closed = 0;

    pthread_mutexattr_t mattr;
    pthread_condattr_t cattr;
    pthread_mutexattr_init(&mattr);
    pthread_condattr_init(&cattr);
    pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&q->lock, &mattr);
    pthread_mutex_init(&q->step_lock, &mattr);
    pthread_cond_init(&q->not_empty, &cattr);
    pthread_cond_init(&q->not_full, &cattr);
    pthread_mutexattr_destroy(&mattr);
    pthread_condattr_destroy(&cattr);

    *fd_out = fd;
    return q;
}

static void mp_queue_close(tfrl_mp_queue *q) {
    if (!q) return;
    pthread_mutex_lock(&q->lock);
    q->closed = 1;
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->lock);
}

static void mp_queue_destroy(tfrl_mp_queue *q, const char *name, int fd) {
    if (!q) return;
    size_t bytes = mp_queue_bytes(q->capacity);
    pthread_mutex_destroy(&q->lock);
    pthread_mutex_destroy(&q->step_lock);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
    munmap(q, bytes);
    if (fd >= 0) close(fd);
    if (name && name[0]) shm_unlink(name);
}

static int mp_queue_next_step(tfrl_mp_queue *q) {
    int step = -1;
    if (!q) return -1;
    pthread_mutex_lock(&q->step_lock);
    if (q->steps_done < q->total_steps) {
        step = q->steps_done;
        q->steps_done++;
    }
    pthread_mutex_unlock(&q->step_lock);
    return step;
}

static int mp_queue_push(tfrl_mp_queue *q, const tfrl_transition *tr) {
    if (!q || !tr) return 0;
    pthread_mutex_lock(&q->lock);
    while (!q->closed && q->count >= q->capacity) {
        pthread_cond_wait(&q->not_full, &q->lock);
    }
    if (q->closed) {
        pthread_mutex_unlock(&q->lock);
        return 0;
    }
    q->items[q->tail] = *tr;
    q->tail = (q->tail + 1) % q->capacity;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->lock);
    return 1;
}

static int mp_queue_pop(tfrl_mp_queue *q, tfrl_transition *out) {
    if (!q || !out) return 0;
    pthread_mutex_lock(&q->lock);
    while (!q->closed && q->count == 0) {
        pthread_cond_wait(&q->not_empty, &q->lock);
    }
    if (q->count == 0 && q->closed) {
        pthread_mutex_unlock(&q->lock);
        return 0;
    }
    *out = q->items[q->head];
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->lock);
    return 1;
}

static int read_marker_version(const char *path, long long *out_version) {
    if (!path || !path[0] || !out_version) return 0;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    char buf[64] = {0};
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return 0;
    char *end = NULL;
    long long v = strtoll(buf, &end, 10);
    if (end == buf) return 0;
    *out_version = v;
    return 1;
}

static int write_marker_version(const char *path, long long version) {
    if (!path || !path[0]) return 0;
    char tmp[256];
    size_t path_len = strlen(path);
    if (path_len + 4 >= sizeof(tmp)) return 0;
    memcpy(tmp, path, path_len);
    memcpy(tmp + path_len, ".tmp", 5);
    int fd = open(tmp, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) return 0;
    char buf[64];
    int written = snprintf(buf, sizeof(buf), "%lld\n", version);
    if (write(fd, buf, (size_t)written) != written) {
        close(fd);
        return 0;
    }
    close(fd);
    if (rename(tmp, path) != 0) return 0;
    return 1;
}

static void ensure_parent_dir(const char *path) {
    if (!path) return;
    const char *slash = strrchr(path, '/');
    if (!slash || slash == path) return;
    size_t len = (size_t)(slash - path);
    if (len >= 256) return;
    char dir[256];
    memcpy(dir, path, len);
    dir[len] = '\0';
    mkdir(dir, 0755);
}

static int dashboard_enabled(void) {
    const char *env = getenv("TFRL_DASHBOARD");
    if (env && strcmp(env, "0") == 0) return 0;
    if (env && strcmp(env, "1") == 0) return 1;
    return isatty(fileno(stdout));
}

static void dashboard_render(const tfrl_runner_config *cfg,
                             int step,
                             int total_steps,
                             int env_count,
                             int agent_count,
                             int total_episodes,
                             double total_return_sum,
                             double last_return,
                             double elapsed_seconds,
                             double env_seconds,
                             double update_seconds,
                             double render_seconds) {
    long long total_env_steps = (long long)(step + 1) * env_count * agent_count;
    double steps_per_sec = elapsed_seconds > 0.0 ? (double)(step + 1) / elapsed_seconds : 0.0;
    double samples_per_sec = elapsed_seconds > 0.0 ? (double)total_env_steps / elapsed_seconds : 0.0;
    double avg_return = total_episodes > 0 ? total_return_sum / (double)total_episodes : 0.0;
    double avg_ep_len = total_episodes > 0 ? (double)total_env_steps / (double)total_episodes : 0.0;
    double est_eps_left = 0.0;
    if (avg_ep_len > 0.0 && total_steps > step + 1) {
        long long remaining_env_steps = (long long)(total_steps - (step + 1)) * env_count * agent_count;
        est_eps_left = (double)remaining_env_steps / avg_ep_len;
    }
    double progress = total_steps > 0 ? (double)(step + 1) * 100.0 / (double)total_steps : 0.0;
    int steps_left = total_steps - (step + 1);

    fprintf(stdout, "\033[2J\033[H");
    fprintf(stdout, "\033[1;34mTINYFIN 🐟\033[0m\n");
    fprintf(stdout, "progress: %6.2f%%  steps: %d/%d  left: %d\n",
            progress, step + 1, total_steps, steps_left);
    fprintf(stdout, "episodes: %d  est_left: %.0f  last_return: %.4f  avg_return: %.4f\n",
            total_episodes, est_eps_left, last_return, avg_return);
    fprintf(stdout, "perf: elapsed %.1fs  steps/s %.1f  sps %.1f\n",
            elapsed_seconds, steps_per_sec, samples_per_sec);
    if (elapsed_seconds > 0.0) {
        double other = elapsed_seconds - env_seconds - update_seconds - render_seconds;
        if (other < 0.0) other = 0.0;
        fprintf(stdout, "timers: env %.1fs (%.0f%%)  update %.1fs (%.0f%%)  render %.1fs (%.0f%%)  other %.1fs (%.0f%%)\n",
                env_seconds, (env_seconds / elapsed_seconds) * 100.0,
                update_seconds, (update_seconds / elapsed_seconds) * 100.0,
                render_seconds, (render_seconds / elapsed_seconds) * 100.0,
                other, (other / elapsed_seconds) * 100.0);
    }
    fprintf(stdout, "config: algo=%s env=%s envs=%d agents=%d threads=%d backend=%s device=%s\n",
            cfg->algo ? cfg->algo : "dqn",
            cfg->env_name ? cfg->env_name : "maze_rooms",
            env_count,
            agent_count,
            cfg->threads > 0 ? cfg->threads : env_count,
            cfg->backend ? cfg->backend : "cpu",
            cfg->device ? cfg->device : "cpu");
    fflush(stdout);
}

static void profile_write_json(const tfrl_runner_config *cfg,
                               const char *path,
                               int steps,
                               int env_count,
                               int agent_count,
                               int total_episodes,
                               double total_return_sum,
                               double elapsed_seconds,
                               double env_seconds,
                               double update_seconds,
                               double render_seconds) {
    if (!path || !path[0]) return;
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "profile: failed to write '%s': %s\n", path, strerror(errno));
        return;
    }
    long long total_env_steps = (long long)steps * env_count * agent_count;
    double steps_per_sec = elapsed_seconds > 0.0 ? (double)steps / elapsed_seconds : 0.0;
    double samples_per_sec = elapsed_seconds > 0.0 ? (double)total_env_steps / elapsed_seconds : 0.0;
    double avg_return = total_episodes > 0 ? total_return_sum / (double)total_episodes : 0.0;
    double other = elapsed_seconds - env_seconds - update_seconds - render_seconds;
    if (other < 0.0) other = 0.0;

    fprintf(fp, "{\n");
    fprintf(fp, "  \"algo\": \"%s\",\n", cfg->algo ? cfg->algo : "dqn");
    fprintf(fp, "  \"env\": \"%s\",\n", cfg->env_name ? cfg->env_name : "maze_rooms");
    fprintf(fp, "  \"backend\": \"%s\",\n", cfg->backend ? cfg->backend : "cpu");
    fprintf(fp, "  \"device\": \"%s\",\n", cfg->device ? cfg->device : "cpu");
    fprintf(fp, "  \"steps\": %d,\n", steps);
    fprintf(fp, "  \"envs\": %d,\n", env_count);
    fprintf(fp, "  \"agents\": %d,\n", agent_count);
    fprintf(fp, "  \"threads\": %d,\n", cfg->threads > 0 ? cfg->threads : env_count);
    fprintf(fp, "  \"episodes\": %d,\n", total_episodes);
    fprintf(fp, "  \"avg_return\": %.6f,\n", avg_return);
    fprintf(fp, "  \"elapsed_seconds\": %.6f,\n", elapsed_seconds);
    fprintf(fp, "  \"env_seconds\": %.6f,\n", env_seconds);
    fprintf(fp, "  \"update_seconds\": %.6f,\n", update_seconds);
    fprintf(fp, "  \"render_seconds\": %.6f,\n", render_seconds);
    fprintf(fp, "  \"other_seconds\": %.6f,\n", other);
    fprintf(fp, "  \"steps_per_sec\": %.6f,\n", steps_per_sec);
    fprintf(fp, "  \"samples_per_sec\": %.6f\n", samples_per_sec);
    fprintf(fp, "}\n");
    fclose(fp);
}

static void json_write_string(FILE *out, const char *s, size_t len) {
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '\"':
                fputs("\\\"", out);
                break;
            case '\\':
                fputs("\\\\", out);
                break;
            case '\n':
                fputs("\\n", out);
                break;
            case '\r':
                fputs("\\r", out);
                break;
            case '\t':
                fputs("\\t", out);
                break;
            default:
                if (c < 0x20) {
                    fprintf(out, "\\u%04x", c);
                } else {
                    fputc(c, out);
                }
                break;
        }
    }
}

static int json_is_number(const char *s, size_t len) {
    if (len == 0 || len >= 64) return 0;
    char tmp[64];
    memcpy(tmp, s, len);
    tmp[len] = '\0';
    char *end = NULL;
    (void)strtod(tmp, &end);
    return end && *end == '\0';
}

static void dump_meta_json(const char *meta, FILE *out) {
    fputc('{', out);
    if (!meta || meta[0] == '\0') {
        fputs("}\n", out);
        return;
    }
    int first = 1;
    const char *p = meta;
    while (*p) {
        while (*p == ' ') p++;
        if (!*p) break;
        const char *key_start = p;
        while (*p && *p != '=' && *p != ' ') p++;
        if (*p != '=') {
            while (*p && *p != ' ') p++;
            continue;
        }
        size_t key_len = (size_t)(p - key_start);
        p++;
        const char *val_start = p;
        while (*p && *p != ' ') p++;
        size_t val_len = (size_t)(p - val_start);
        if (key_len == 0) continue;
        if (!first) fputc(',', out);
        fputc('\"', out);
        json_write_string(out, key_start, key_len);
        fputc('\"', out);
        fputc(':', out);
        if (json_is_number(val_start, val_len)) {
            fwrite(val_start, 1, val_len, out);
        } else {
            fputc('\"', out);
            json_write_string(out, val_start, val_len);
            fputc('\"', out);
        }
        first = 0;
    }
    fputs("}\n", out);
}

static void build_trace_meta(char *buffer,
                             size_t buffer_len,
                             const tfrl_algo_config *cfg,
                             int agent_count,
                             int share_policy,
                             const char *diag) {
    if (!buffer || buffer_len == 0 || !cfg) return;
    snprintf(buffer,
             buffer_len,
             "algo=%s defaults=v%d env=%s agents=%d share_policy=%d seed=%d deterministic=%d gamma=%.3f lr=%.6f "
             "epsilon=%.3f entropy=%.3f clip_eps=%.3f gae_lambda=%.3f replay=%d batch=%d per_alpha=%.3f "
             "per_beta=%.3f steps_per_batch=%d epochs=%d c51_atoms=%d c51_vmin=%.3f c51_vmax=%.3f iqn_quantiles=%d "
             "iqn_tau_samples=%d mcts_sims=%d mcts_depth=%d",
             cfg->name ? cfg->name : "null",
             cfg->defaults_version,
             cfg->env_name ? cfg->env_name : "null",
             agent_count,
             share_policy,
             cfg->seed,
             cfg->deterministic,
             cfg->gamma,
             cfg->lr,
             cfg->epsilon,
             cfg->entropy_coef,
             cfg->clip_eps,
             cfg->gae_lambda,
             cfg->replay_size,
             cfg->batch_size,
             cfg->per_alpha,
             cfg->per_beta,
             cfg->steps_per_batch,
             cfg->epochs,
             cfg->c51_atoms,
             cfg->c51_vmin,
             cfg->c51_vmax,
             cfg->iqn_quantiles,
             cfg->iqn_tau_samples,
             cfg->mcts_sims,
             cfg->mcts_depth);
    if (diag && diag[0] != '\0') {
        size_t len = strlen(buffer);
        if (len < buffer_len - 1) {
            snprintf(buffer + len, buffer_len - len, " %s", diag);
        }
    }
}

typedef struct {
    tfrl_env **envs;
    const tfrl_action *actions;
    tfrl_step_result *steps;
    int start;
    int count;
} tfrl_step_job;

static void *step_worker(void *arg) {
    tfrl_step_job *job = (tfrl_step_job *)arg;
    int end = job->start + job->count;
    for (int i = job->start; i < end; i++) {
        job->steps[i] = tfrl_env_step(job->envs[i], job->actions[i]);
    }
    return NULL;
}

static void step_envs_threaded(tfrl_env **envs, int env_count, const tfrl_action *actions, tfrl_step_result *steps, int threads) {
    if (env_count <= 0) return;
    if (threads <= 1 || env_count == 1) {
        tfrl_env_step_batch(envs, env_count, actions, steps);
        return;
    }
    if (threads > env_count) threads = env_count;
    pthread_t *tids = (pthread_t *)calloc((size_t)threads, sizeof(pthread_t));
    tfrl_step_job *jobs = (tfrl_step_job *)calloc((size_t)threads, sizeof(tfrl_step_job));
    if (!tids || !jobs) {
        free(tids);
        free(jobs);
        tfrl_env_step_batch(envs, env_count, actions, steps);
        return;
    }

    int base = env_count / threads;
    int extra = env_count % threads;
    int start = 0;
    for (int t = 0; t < threads; t++) {
        int count = base + (t < extra ? 1 : 0);
        jobs[t].envs = envs;
        jobs[t].actions = actions;
        jobs[t].steps = steps;
        jobs[t].start = start;
        jobs[t].count = count;
        pthread_create(&tids[t], NULL, step_worker, &jobs[t]);
        start += count;
    }
    for (int t = 0; t < threads; t++) {
        pthread_join(tids[t], NULL);
    }
    free(tids);
    free(jobs);
}

typedef struct {
    tfrl_env *env;
    tfrl_algo *algo;
    int total_steps;
    int *steps_done;
    pthread_mutex_t *step_lock;
    uint64_t seed;
} tfrl_a3c_worker;

typedef struct {
    tfrl_transition *items;
    int capacity;
    int head;
    int tail;
    int count;
    int max_count;
    long long push_count;
    long long pop_count;
    int closed;
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} tfrl_transition_queue;

static int queue_init(tfrl_transition_queue *q, int capacity) {
    if (!q || capacity <= 0) return 0;
    q->items = (tfrl_transition *)calloc((size_t)capacity, sizeof(tfrl_transition));
    if (!q->items) return 0;
    q->capacity = capacity;
    q->head = 0;
    q->tail = 0;
    q->count = 0;
    q->max_count = 0;
    q->push_count = 0;
    q->pop_count = 0;
    q->closed = 0;
    pthread_mutex_init(&q->lock, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
    return 1;
}

static void queue_close(tfrl_transition_queue *q) {
    pthread_mutex_lock(&q->lock);
    q->closed = 1;
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->lock);
}

static void queue_destroy(tfrl_transition_queue *q) {
    if (!q) return;
    queue_close(q);
    pthread_mutex_destroy(&q->lock);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
    free(q->items);
}

static int queue_push(tfrl_transition_queue *q, const tfrl_transition *tr) {
    pthread_mutex_lock(&q->lock);
    while (!q->closed && q->count == q->capacity) {
        pthread_cond_wait(&q->not_full, &q->lock);
    }
    if (q->closed) {
        pthread_mutex_unlock(&q->lock);
        return 0;
    }
    q->items[q->tail] = *tr;
    q->tail = (q->tail + 1) % q->capacity;
    q->count++;
    if (q->count > q->max_count) q->max_count = q->count;
    q->push_count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->lock);
    return 1;
}

static int queue_pop(tfrl_transition_queue *q, tfrl_transition *out) {
    pthread_mutex_lock(&q->lock);
    while (!q->closed && q->count == 0) {
        pthread_cond_wait(&q->not_empty, &q->lock);
    }
    if (q->count == 0 && q->closed) {
        pthread_mutex_unlock(&q->lock);
        return 0;
    }
    *out = q->items[q->head];
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    q->pop_count++;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->lock);
    return 1;
}

typedef struct {
    int capacity;
    int count;
    int max_count;
    long long push_count;
    long long pop_count;
} tfrl_queue_stats;

static tfrl_queue_stats queue_stats(const tfrl_transition_queue *q) {
    tfrl_queue_stats stats = {0};
    if (!q) return stats;
    pthread_mutex_lock((pthread_mutex_t *)&q->lock);
    stats.count = q->count;
    stats.max_count = q->max_count;
    stats.capacity = q->capacity;
    stats.push_count = q->push_count;
    stats.pop_count = q->pop_count;
    pthread_mutex_unlock((pthread_mutex_t *)&q->lock);
    return stats;
}

static void *a3c_worker_main(void *arg) {
    tfrl_a3c_worker *worker = (tfrl_a3c_worker *)arg;
    int local_steps = 0;
    tfrl_obs obs = tfrl_env_reset(worker->env, worker->seed);
    while (1) {
        pthread_mutex_lock(worker->step_lock);
        if (*worker->steps_done >= worker->total_steps) {
            pthread_mutex_unlock(worker->step_lock);
            break;
        }
        (*worker->steps_done)++;
        pthread_mutex_unlock(worker->step_lock);

        tfrl_action action = tfrl_algo_act(worker->algo, worker->env, obs);
        tfrl_step_result step = tfrl_env_step(worker->env, action);
        tfrl_transition transition = {
            .obs = obs,
            .action = action,
            .reward = step.reward,
            .next_obs = step.observation,
            .done = step.done,
        };
        worker->algo->vtable->update(worker->algo->ctx, &transition);
        obs = step.observation;
        local_steps++;
        if (step.done) {
            obs = tfrl_env_reset(worker->env, worker->seed + (uint64_t)local_steps + 1u);
        }
    }
    return NULL;
}

static int run_a3c_async(const tfrl_runner_config *cfg,
                         tfrl_env **envs,
                         int env_count,
                         tfrl_algo *algo) {
    if (env_count <= 0 || !algo) return 1;
    int total_steps = cfg->steps > 0 ? cfg->steps : 1000;
    int steps_done = 0;
    pthread_mutex_t step_lock;
    pthread_mutex_init(&step_lock, NULL);

    pthread_t *threads = (pthread_t *)calloc((size_t)env_count, sizeof(pthread_t));
    tfrl_a3c_worker *workers = (tfrl_a3c_worker *)calloc((size_t)env_count, sizeof(tfrl_a3c_worker));
    if (!threads || !workers) {
        free(threads);
        free(workers);
        pthread_mutex_destroy(&step_lock);
        return 1;
    }

    for (int i = 0; i < env_count; i++) {
        workers[i].env = envs[i];
        workers[i].algo = algo;
        workers[i].total_steps = total_steps;
        workers[i].steps_done = &steps_done;
        workers[i].step_lock = &step_lock;
        workers[i].seed = (uint64_t)(cfg->seed + i + 1);
        pthread_create(&threads[i], NULL, a3c_worker_main, &workers[i]);
    }

    for (int i = 0; i < env_count; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(workers);
    pthread_mutex_destroy(&step_lock);
    return 0;
}

typedef struct {
    tfrl_env *env;
    tfrl_algo *algo;
    tfrl_transition_queue *queue;
    int total_steps;
    int *steps_done;
    pthread_mutex_t *step_lock;
    pthread_mutex_t *policy_lock;
    uint64_t seed;
} tfrl_impala_actor;

static void *impala_actor_main(void *arg) {
    tfrl_impala_actor *actor = (tfrl_impala_actor *)arg;
    tfrl_obs obs = tfrl_env_reset(actor->env, actor->seed);
    while (1) {
        pthread_mutex_lock(actor->step_lock);
        if (*actor->steps_done >= actor->total_steps) {
            pthread_mutex_unlock(actor->step_lock);
            break;
        }
        (*actor->steps_done)++;
        pthread_mutex_unlock(actor->step_lock);

        pthread_mutex_lock(actor->policy_lock);
        tfrl_action action = tfrl_algo_act(actor->algo, actor->env, obs);
        pthread_mutex_unlock(actor->policy_lock);
        tfrl_step_result step = tfrl_env_step(actor->env, action);
        tfrl_transition transition = {
            .obs = obs,
            .action = action,
            .reward = step.reward,
            .next_obs = step.observation,
            .done = step.done,
        };
        if (!queue_push(actor->queue, &transition)) break;
        obs = step.observation;
        if (step.done) {
            obs = tfrl_env_reset(actor->env, actor->seed + 1u);
        }
    }
    return NULL;
}

typedef struct {
    tfrl_algo *algo;
    tfrl_transition_queue *queue;
    int total_steps;
    pthread_mutex_t *policy_lock;
    int learner_batch;
    int log_every;
    int profile;
    double start_time;
    double update_seconds;
    int processed;
    int learner_batch_auto;
} tfrl_impala_learner;

static void *impala_learner_main(void *arg) {
    tfrl_impala_learner *learner = (tfrl_impala_learner *)arg;
    int processed = 0;
    tfrl_transition tr;
    int base_batch = learner->learner_batch > 0 ? learner->learner_batch : 1;
    int batch = base_batch;
    learner->start_time = now_seconds();
    learner->update_seconds = 0.0;
    while (processed < learner->total_steps) {
        if (learner->learner_batch_auto) {
            tfrl_queue_stats stats = queue_stats(learner->queue);
            int cap = stats.capacity > 0 ? stats.capacity : 1;
            if (stats.count > (cap * 3) / 4) {
                batch = base_batch * 4;
            } else if (stats.count > cap / 2) {
                batch = base_batch * 2;
            } else {
                batch = base_batch;
            }
            if (batch > 1024) batch = 1024;
        }
        for (int i = 0; i < batch && processed < learner->total_steps; i++) {
            if (!queue_pop(learner->queue, &tr)) return NULL;
            double t0 = now_seconds();
            pthread_mutex_lock(learner->policy_lock);
            learner->algo->vtable->update(learner->algo->ctx, &tr);
            pthread_mutex_unlock(learner->policy_lock);
            learner->update_seconds += now_seconds() - t0;
            processed++;
            if (learner->log_every > 0 && (processed % learner->log_every) == 0) {
                double elapsed = now_seconds() - learner->start_time;
                double sps = elapsed > 0.0 ? (double)processed / elapsed : 0.0;
                tfrl_queue_stats stats = queue_stats(learner->queue);
                fprintf(stdout,
                        "impala: steps=%d sps=%.1f queue=%d/%d push=%lld pop=%lld update=%.2fs\n",
                        processed, sps, stats.count, stats.max_count, stats.push_count, stats.pop_count,
                        learner->update_seconds);
            }
        }
    }
    learner->processed = processed;
    return NULL;
}

static int run_impala_split(const tfrl_runner_config *cfg,
                            tfrl_env **envs,
                            int env_count,
                            tfrl_algo *algo) {
    int total_steps = cfg->steps > 0 ? cfg->steps : 1000;
    int steps_done = 0;
    int queue_capacity = cfg->queue_capacity > 0 ? cfg->queue_capacity : 1024;
    tfrl_transition_queue queue;
    if (!queue_init(&queue, queue_capacity)) return 1;

    pthread_mutex_t step_lock;
    pthread_mutex_t policy_lock;
    pthread_mutex_init(&step_lock, NULL);
    pthread_mutex_init(&policy_lock, NULL);

    pthread_t *actors = (pthread_t *)calloc((size_t)env_count, sizeof(pthread_t));
    tfrl_impala_actor *actor_cfg = (tfrl_impala_actor *)calloc((size_t)env_count, sizeof(tfrl_impala_actor));
    if (!actors || !actor_cfg) {
        free(actors);
        free(actor_cfg);
        queue_destroy(&queue);
        pthread_mutex_destroy(&step_lock);
        pthread_mutex_destroy(&policy_lock);
        return 1;
    }

    for (int i = 0; i < env_count; i++) {
        actor_cfg[i].env = envs[i];
        actor_cfg[i].algo = algo;
        actor_cfg[i].queue = &queue;
        actor_cfg[i].total_steps = total_steps;
        actor_cfg[i].steps_done = &steps_done;
        actor_cfg[i].step_lock = &step_lock;
        actor_cfg[i].policy_lock = &policy_lock;
        actor_cfg[i].seed = (uint64_t)(cfg->seed + i + 1);
        pthread_create(&actors[i], NULL, impala_actor_main, &actor_cfg[i]);
    }

    pthread_t learner_thread;
    tfrl_impala_learner learner = {
        .algo = algo,
        .queue = &queue,
        .total_steps = total_steps,
        .policy_lock = &policy_lock,
        .learner_batch = cfg->learner_batch > 0 ? cfg->learner_batch : 1,
        .log_every = cfg->log_every,
        .profile = cfg->profile,
        .learner_batch_auto = cfg->learner_batch_auto,
    };
    pthread_create(&learner_thread, NULL, impala_learner_main, &learner);

    for (int i = 0; i < env_count; i++) {
        pthread_join(actors[i], NULL);
    }
    queue_close(&queue);
    pthread_join(learner_thread, NULL);

    free(actors);
    free(actor_cfg);
    if (cfg->profile) {
        double elapsed = now_seconds() - learner.start_time;
        double sps = elapsed > 0.0 ? (double)learner.processed / elapsed : 0.0;
        tfrl_queue_stats stats = queue_stats(&queue);
        fprintf(stdout,
                "impala_profile: steps=%d elapsed=%.2fs sps=%.1f update=%.2fs queue_max=%d push=%lld pop=%lld\n",
                learner.processed, elapsed, sps, learner.update_seconds, stats.max_count,
                stats.push_count, stats.pop_count);
    }
    queue_destroy(&queue);
    pthread_mutex_destroy(&step_lock);
    pthread_mutex_destroy(&policy_lock);
    return 0;
}

typedef struct {
    tfrl_runner_config cfg;
    int actor_id;
    int total_steps;
    tfrl_mp_queue *queue;
    const char *sync_path;
    const char *sync_marker;
    int sync_every;
} tfrl_mp_actor;

static int is_dqn_family(const char *name) {
    if (!name) return 0;
    return strcmp(name, "dqn") == 0 || strcmp(name, "rainbow") == 0 || strcmp(name, "qrdqn") == 0 || strcmp(name, "iqn") == 0;
}

static void mp_actor_main(const tfrl_mp_actor *actor) {
    tfrl_env_config env_cfg = {.name = actor->cfg.env_name, .seed = (uint64_t)(actor->cfg.seed + actor->actor_id + 1)};
    tfrl_env *env = tfrl_env_create(&env_cfg);
    if (!env) _exit(1);
    const tfrl_env_spec *spec = tfrl_env_get_spec(env);
    if (!spec) {
        tfrl_env_destroy(env);
        _exit(1);
    }
    tfrl_obs obs = tfrl_env_reset(env, (uint64_t)(actor->cfg.seed + actor->actor_id + 1));
    tfrl_algo algo = {0};
    int loaded = 0;
    long long last_version = -1;
    while (1) {
        int step_id = mp_queue_next_step(actor->queue);
        if (step_id < 0) break;
        if (actor->sync_every > 0 && (step_id % actor->sync_every) == 0) {
            long long version = -1;
            if (read_marker_version(actor->sync_marker, &version) && version > last_version) {
                tfrl_algo_destroy(&algo);
                tfrl_algo_config algo_cfg;
                fill_algo_config(&algo_cfg, &actor->cfg, spec, actor->cfg.seed, actor->sync_path);
                tfrl_algo_config_apply_defaults(&algo_cfg);
                algo = tfrl_algo_create(&algo_cfg, spec);
                loaded = 1;
                last_version = version;
            }
        }
        if (!algo.ctx) {
            const char *load_path = loaded ? actor->sync_path : actor->cfg.load_path;
            tfrl_algo_config algo_cfg;
            fill_algo_config(&algo_cfg, &actor->cfg, spec, actor->cfg.seed, load_path);
            tfrl_algo_config_apply_defaults(&algo_cfg);
            algo = tfrl_algo_create(&algo_cfg, spec);
        }
        tfrl_action action = tfrl_algo_act(&algo, env, obs);
        tfrl_step_result step = tfrl_env_step(env, action);
        tfrl_transition transition = {
            .obs = obs,
            .action = action,
            .reward = step.reward,
            .next_obs = step.observation,
            .done = step.done,
        };
        if (!mp_queue_push(actor->queue, &transition)) break;
        obs = step.observation;
        if (step.done) {
            obs = tfrl_env_reset(env, (uint64_t)(actor->cfg.seed + actor->actor_id + 1));
        }
    }
    tfrl_algo_destroy(&algo);
    tfrl_env_destroy(env);
    _exit(0);
}

static void mp_actor_main_ppo(const tfrl_mp_actor *actor) {
    tfrl_env_config env_cfg = {.name = actor->cfg.env_name, .seed = (uint64_t)(actor->cfg.seed + actor->actor_id + 1)};
    tfrl_env *env = tfrl_env_create(&env_cfg);
    if (!env) _exit(1);
    const tfrl_env_spec *spec = tfrl_env_get_spec(env);
    if (!spec) {
        tfrl_env_destroy(env);
        _exit(1);
    }
    tfrl_obs obs = tfrl_env_reset(env, (uint64_t)(actor->cfg.seed + actor->actor_id + 1));
    tfrl_algo algo = {0};
    long long last_version = -1;
    int policy_version = 0;

    while (1) {
        int step_id = mp_queue_next_step(actor->queue);
        if (step_id < 0) break;
        long long version = -1;
        if (read_marker_version(actor->sync_marker, &version) && version > last_version) {
            tfrl_algo_destroy(&algo);
            tfrl_algo_config algo_cfg;
            fill_algo_config(&algo_cfg, &actor->cfg, spec, actor->cfg.seed, actor->sync_path);
            tfrl_algo_config_apply_defaults(&algo_cfg);
            algo = tfrl_algo_create(&algo_cfg, spec);
            last_version = version;
            policy_version = (int)version;
        }
        if (!algo.ctx) {
            tfrl_algo_config algo_cfg;
            fill_algo_config(&algo_cfg, &actor->cfg, spec, actor->cfg.seed, actor->sync_path);
            tfrl_algo_config_apply_defaults(&algo_cfg);
            algo = tfrl_algo_create(&algo_cfg, spec);
        }
        tfrl_action action = tfrl_algo_act(&algo, env, obs);
        tfrl_step_result step = tfrl_env_step(env, action);
        tfrl_transition transition = {
            .obs = obs,
            .action = action,
            .reward = step.reward,
            .next_obs = step.observation,
            .done = step.done,
            .policy_version = policy_version,
        };
        if (!mp_queue_push(actor->queue, &transition)) break;
        obs = step.observation;
        if (step.done) {
            obs = tfrl_env_reset(env, (uint64_t)(actor->cfg.seed + actor->actor_id + 1));
        }
    }
    tfrl_algo_destroy(&algo);
    tfrl_env_destroy(env);
    _exit(0);
}

static int run_mp_dqn(const tfrl_runner_config *cfg) {
    if (!cfg || !is_dqn_family(cfg->algo)) {
        fprintf(stderr, "mp mode only supports dqn/rainbow/qrdqn/iqn\n");
        return 1;
    }
    if (cfg->render || cfg->trace_out) {
        fprintf(stderr, "mp mode does not support render/trace; disable them.\n");
        return 1;
    }
    tfrl_env_config env_cfg = {.name = cfg->env_name, .seed = (uint64_t)(cfg->seed)};
    tfrl_env *env = tfrl_env_create(&env_cfg);
    if (!env) {
        fprintf(stderr, "env create failed\n");
        return 1;
    }
    const tfrl_env_spec *spec = tfrl_env_get_spec(env);
    int agent_count = tfrl_env_agent_count(env);
    if (agent_count != 1) {
        fprintf(stderr, "mp dqn requires single-agent env\n");
        tfrl_env_destroy(env);
        return 1;
    }

    int total_steps = cfg->steps > 0 ? cfg->steps : 1000;
    int actor_count = cfg->mp_actors > 0 ? cfg->mp_actors : 1;
    int queue_capacity = cfg->mp_queue > 0 ? cfg->mp_queue : 65536;
    const char *sync_path = cfg->mp_sync_path && cfg->mp_sync_path[0] ? cfg->mp_sync_path : "runs/mp_sync";
    char sync_marker[256];
    snprintf(sync_marker, sizeof(sync_marker), "%s.done", sync_path);
    int sync_every = cfg->mp_sync_every > 0 ? cfg->mp_sync_every : 1000;
    ensure_parent_dir(sync_path);

    char shm_name[64] = {0};
    int shm_fd = -1;
    tfrl_mp_queue *queue = mp_queue_create(queue_capacity, total_steps, shm_name, sizeof(shm_name), &shm_fd);
    if (!queue) {
        fprintf(stderr, "mp queue init failed\n");
        tfrl_env_destroy(env);
        return 1;
    }

    tfrl_algo_config algo_cfg;
    fill_algo_config(&algo_cfg, cfg, spec, cfg->seed, cfg->load_path);
    tfrl_algo_config_apply_defaults(&algo_cfg);
    tfrl_algo algo = tfrl_algo_create(&algo_cfg, spec);
    if (!algo.ctx) {
        mp_queue_destroy(queue, shm_name, shm_fd);
        tfrl_env_destroy(env);
        return 1;
    }

    pid_t *pids = (pid_t *)calloc((size_t)actor_count, sizeof(pid_t));
    if (!pids) {
        tfrl_algo_destroy(&algo);
        mp_queue_destroy(queue, shm_name, shm_fd);
        tfrl_env_destroy(env);
        return 1;
    }

    for (int i = 0; i < actor_count; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            tfrl_mp_actor actor = {
                .cfg = *cfg,
                .actor_id = i,
                .total_steps = total_steps,
                .queue = queue,
                .sync_path = sync_path,
                .sync_marker = sync_marker,
                .sync_every = sync_every,
            };
            mp_actor_main(&actor);
        }
        if (pid < 0) {
            fprintf(stderr, "fork failed\n");
            mp_queue_close(queue);
            for (int j = 0; j < i; j++) {
                if (pids[j] > 0) kill(pids[j], SIGTERM);
            }
            free(pids);
            tfrl_algo_destroy(&algo);
            mp_queue_destroy(queue, shm_name, shm_fd);
            tfrl_env_destroy(env);
            return 1;
        }
        pids[i] = pid;
    }

    int processed = 0;
    long long sync_version = 0;
    tfrl_transition tr;
    while (processed < total_steps) {
        if (!mp_queue_pop(queue, &tr)) break;
        algo.vtable->update(algo.ctx, &tr);
        processed++;
        if (sync_every > 0 && (processed % sync_every) == 0) {
            tfrl_algo_save(&algo, sync_path);
            sync_version++;
            write_marker_version(sync_marker, sync_version);
        }
    }

    mp_queue_close(queue);
    for (int i = 0; i < actor_count; i++) {
        int status = 0;
        waitpid(pids[i], &status, 0);
    }
    free(pids);
    tfrl_algo_destroy(&algo);
    mp_queue_destroy(queue, shm_name, shm_fd);
    tfrl_env_destroy(env);
    return 0;
}

static int run_mp_ppo(const tfrl_runner_config *cfg) {
    if (!cfg || !cfg->algo || strcmp(cfg->algo, "ppo") != 0) {
        fprintf(stderr, "mp mode only supports ppo here\n");
        return 1;
    }
    if (cfg->render || cfg->trace_out) {
        fprintf(stderr, "mp mode does not support render/trace; disable them.\n");
        return 1;
    }
    tfrl_env_config env_cfg = {.name = cfg->env_name, .seed = (uint64_t)(cfg->seed)};
    tfrl_env *env = tfrl_env_create(&env_cfg);
    if (!env) {
        fprintf(stderr, "env create failed\n");
        return 1;
    }
    const tfrl_env_spec *spec = tfrl_env_get_spec(env);
    int agent_count = tfrl_env_agent_count(env);
    if (agent_count != 1) {
        fprintf(stderr, "mp ppo requires single-agent env\n");
        tfrl_env_destroy(env);
        return 1;
    }

    int total_steps = cfg->steps > 0 ? cfg->steps : 1000;
    int actor_count = cfg->mp_actors > 0 ? cfg->mp_actors : 1;
    int queue_capacity = cfg->mp_queue > 0 ? cfg->mp_queue : 65536;
    const char *sync_path = cfg->mp_sync_path && cfg->mp_sync_path[0] ? cfg->mp_sync_path : "runs/mp_sync";
    char sync_marker[256];
    snprintf(sync_marker, sizeof(sync_marker), "%s.done", sync_path);
    ensure_parent_dir(sync_path);

    char shm_name[64] = {0};
    int shm_fd = -1;
    tfrl_mp_queue *queue = mp_queue_create(queue_capacity, total_steps, shm_name, sizeof(shm_name), &shm_fd);
    if (!queue) {
        fprintf(stderr, "mp queue init failed\n");
        tfrl_env_destroy(env);
        return 1;
    }

    tfrl_algo_config algo_cfg;
    fill_algo_config(&algo_cfg, cfg, spec, cfg->seed, cfg->load_path);
    tfrl_algo_config_apply_defaults(&algo_cfg);
    tfrl_algo algo = tfrl_algo_create(&algo_cfg, spec);
    if (!algo.ctx) {
        mp_queue_destroy(queue, shm_name, shm_fd);
        tfrl_env_destroy(env);
        return 1;
    }
    tfrl_algo_save(&algo, sync_path);
    write_marker_version(sync_marker, 0);

    pid_t *pids = (pid_t *)calloc((size_t)actor_count, sizeof(pid_t));
    if (!pids) {
        tfrl_algo_destroy(&algo);
        mp_queue_destroy(queue, shm_name, shm_fd);
        tfrl_env_destroy(env);
        return 1;
    }

    for (int i = 0; i < actor_count; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            tfrl_mp_actor actor = {
                .cfg = *cfg,
                .actor_id = i,
                .total_steps = total_steps,
                .queue = queue,
                .sync_path = sync_path,
                .sync_marker = sync_marker,
                .sync_every = 1,
            };
            mp_actor_main_ppo(&actor);
        }
        if (pid < 0) {
            fprintf(stderr, "fork failed\n");
            mp_queue_close(queue);
            for (int j = 0; j < i; j++) {
                if (pids[j] > 0) kill(pids[j], SIGTERM);
            }
            free(pids);
            tfrl_algo_destroy(&algo);
            mp_queue_destroy(queue, shm_name, shm_fd);
            tfrl_env_destroy(env);
            return 1;
        }
        pids[i] = pid;
    }

    int processed_total = 0;
    int processed_used = 0;
    int batch = cfg->steps_per_batch > 0 ? cfg->steps_per_batch : 64;
    int batch_count = 0;
    int policy_version = 0;
    tfrl_transition tr;
    while (processed_total < total_steps) {
        if (!mp_queue_pop(queue, &tr)) break;
        processed_total++;
        if (tr.policy_version != policy_version) {
            continue;
        }
        algo.vtable->update(algo.ctx, &tr);
        processed_used++;
        batch_count++;
        if (batch_count >= batch) {
            batch_count = 0;
            policy_version++;
            tfrl_algo_save(&algo, sync_path);
            write_marker_version(sync_marker, policy_version);
        }
    }

    mp_queue_close(queue);
    for (int i = 0; i < actor_count; i++) {
        int status = 0;
        waitpid(pids[i], &status, 0);
    }
    free(pids);
    tfrl_algo_destroy(&algo);
    mp_queue_destroy(queue, shm_name, shm_fd);
    tfrl_env_destroy(env);
    fprintf(stdout, "mp_ppo: total=%d used=%d version=%d\n", processed_total, processed_used, policy_version);
    return 0;
}

static int run_replay(const tfrl_runner_config *cfg) {
    tfrl_trace_reader *reader = tfrl_trace_reader_open(cfg->trace_in);
    if (!reader) {
        fprintf(stderr, "failed to open trace: %s\n", cfg->trace_in);
        return 1;
    }
    const char *meta = tfrl_trace_reader_meta(reader);
    if (cfg->dump_meta_json) {
        dump_meta_json(meta, stdout);
        tfrl_trace_reader_close(reader);
        return 0;
    }
    if (meta && meta[0] != '\0') {
        fprintf(stdout, "trace_meta: %s\n", meta);
    }
    tfrl_viewer_config vcfg = {.fps = cfg->render_fps, .title = "tinyfin-rl replay", .meta = meta};
    tfrl_viewer *viewer = tfrl_viewer_create(&vcfg);
    if (!viewer) {
        fprintf(stderr, "viewer not available (build without raylib?)\n");
        tfrl_trace_reader_close(reader);
        return 1;
    }

    size_t buf_cap = 1024 * 1024;
    void *buffer = calloc(1, buf_cap);
    if (!buffer) {
        tfrl_viewer_destroy(viewer);
        tfrl_trace_reader_close(reader);
        return 1;
    }

    size_t len = 0;
    while (tfrl_trace_reader_next(reader, buffer, buf_cap, &len)) {
        tfrl_viewer_draw(viewer, buffer, len);
        if (tfrl_viewer_should_close(viewer)) break;
    }
    free(buffer);
    tfrl_viewer_destroy(viewer);
    tfrl_trace_reader_close(reader);
    return 0;
}

int tfrl_runner_run(const tfrl_runner_config *cfg) {
    if (!cfg) return 1;
    configure_tinyfin(cfg);
    if (cfg->mode == TFRL_MODE_REPLAY) {
        if (!cfg->trace_in) {
            fprintf(stderr, "replay requires --trace-in\n");
            return 1;
        }
        return run_replay(cfg);
    }

    int seed = cfg->seed;
    if (cfg->deterministic && seed == 0) {
        seed = 1;
        fprintf(stdout, "deterministic mode: seed set to %d\n", seed);
    }
    seed_rng(seed);
    int env_count = cfg->envs > 0 ? cfg->envs : 1;
    if (cfg->actor_count > 0 && cfg->algo && strcmp(cfg->algo, "impala") == 0) {
        env_count = cfg->actor_count;
    }
    int render_idx = cfg->render_env;
    if (render_idx < 0) render_idx = 0;
    if (render_idx >= env_count) {
        fprintf(stderr, "render-env out of range (0..%d)\n", env_count - 1);
        return 1;
    }
    tfrl_env **envs = (tfrl_env **)calloc((size_t)env_count, sizeof(tfrl_env *));
    if (!envs) return 1;
    for (int i = 0; i < env_count; i++) {
        tfrl_env_config env_cfg = {.name = cfg->env_name, .seed = (uint64_t)(seed + i)};
        envs[i] = tfrl_env_create(&env_cfg);
        if (!envs[i]) {
            fprintf(stderr, "env creation failed\n");
            for (int j = 0; j < i; j++) {
                tfrl_env_destroy(envs[j]);
            }
            free(envs);
            return 1;
        }
    }
    const tfrl_env_spec *spec = tfrl_env_get_spec(envs[0]);
    int agent_count = tfrl_env_agent_count(envs[0]);
    if (agent_count <= 0) {
        fprintf(stderr, "env agent count invalid\n");
        for (int i = 0; i < env_count; i++) {
            tfrl_env_destroy(envs[i]);
        }
        free(envs);
        return 1;
    }
    for (int i = 1; i < env_count; i++) {
        if (tfrl_env_agent_count(envs[i]) != agent_count) {
            fprintf(stderr, "env agent count mismatch\n");
            for (int j = 0; j < env_count; j++) {
                tfrl_env_destroy(envs[j]);
            }
            free(envs);
            return 1;
        }
    }
    if (cfg->agents_expected > 0 && cfg->agents_expected != agent_count) {
        fprintf(stderr, "env agent count mismatch (expected %d got %d)\n", cfg->agents_expected, agent_count);
        for (int i = 0; i < env_count; i++) {
            tfrl_env_destroy(envs[i]);
        }
        free(envs);
        return 1;
    }
    tfrl_algo_config algo_cfg;
    fill_algo_config(&algo_cfg, cfg, spec, seed, cfg->load_path);
    tfrl_algo_config_apply_defaults(&algo_cfg);
    int policy_count = cfg->share_policy ? 1 : agent_count;
    tfrl_algo algos[policy_count];
    char load_paths[policy_count][256];
    for (int a = 0; a < policy_count; a++) {
        tfrl_algo_config cfg_agent = algo_cfg;
        if (cfg->load_path && policy_count > 1) {
            snprintf(load_paths[a], sizeof(load_paths[a]), "%s.agent%d", cfg->load_path, a);
            cfg_agent.load_path = load_paths[a];
        }
        algos[a] = tfrl_algo_create(&cfg_agent, spec);
        if (!algos[a].ctx) {
            fprintf(stderr, "algo creation failed\n");
            for (int k = 0; k < a; k++) {
                tfrl_algo_destroy(&algos[k]);
            }
            for (int i = 0; i < env_count; i++) {
                tfrl_env_destroy(envs[i]);
            }
            free(envs);
            return 1;
        }
    }

    tfrl_viewer *viewer = NULL;
    void *render_buf = NULL;
    size_t render_buf_len = 0;
    if (cfg->render) {
        tfrl_viewer_config vcfg = {.fps = cfg->render_fps, .title = "tinyfin-rl"};
        viewer = tfrl_viewer_create(&vcfg);
        if (!viewer) {
            fprintf(stderr, "viewer not available (build without raylib?)\n");
        } else {
            render_buf_len = tfrl_env_render_bytes_needed(envs[render_idx]);
            render_buf = calloc(1, render_buf_len);
        }
    }

    tfrl_trace_writer *writer = NULL;
    char trace_meta[768];
    char diag_meta[256];
    diag_meta[0] = '\0';
    if (policy_count > 0) {
        tfrl_algo_diagnostics(&algos[0], diag_meta, sizeof(diag_meta));
    }
    append_meta_kv(diag_meta, sizeof(diag_meta), "backend", backend_name());
    append_meta_kv(diag_meta, sizeof(diag_meta), "device",
                   tensor_get_default_device() == DEVICE_GPU ? "gpu" : "cpu");
    build_trace_meta(trace_meta, sizeof(trace_meta), &algo_cfg, agent_count, cfg->share_policy ? 1 : 0, diag_meta);
    if (cfg->trace_out) {
        writer = tfrl_trace_writer_open_with_meta(cfg->trace_out, trace_meta);
        if (!writer) {
            fprintf(stderr, "failed to open trace out: %s\n", cfg->trace_out);
        } else {
            if (!render_buf) {
                render_buf_len = tfrl_env_render_bytes_needed(envs[render_idx]);
                render_buf = calloc(1, render_buf_len);
            }
        }
    }

    int total_agents = env_count * agent_count;
    tfrl_obs *obs = (tfrl_obs *)calloc((size_t)total_agents, sizeof(tfrl_obs));
    tfrl_action *actions = (tfrl_action *)calloc((size_t)total_agents, sizeof(tfrl_action));
    tfrl_step_result *steps = (tfrl_step_result *)calloc((size_t)total_agents, sizeof(tfrl_step_result));
    double *episode_returns = (double *)calloc((size_t)total_agents, sizeof(double));
    int *episode_counts = (int *)calloc((size_t)total_agents, sizeof(int));
    tfrl_env **reset_envs = NULL;
    tfrl_obs *reset_obs = NULL;
    uint64_t *reset_seeds = NULL;
    int *reset_indices = NULL;
    if (agent_count == 1) {
        reset_envs = (tfrl_env **)calloc((size_t)env_count, sizeof(tfrl_env *));
        reset_obs = (tfrl_obs *)calloc((size_t)env_count, sizeof(tfrl_obs));
        reset_seeds = (uint64_t *)calloc((size_t)env_count, sizeof(uint64_t));
        reset_indices = (int *)calloc((size_t)env_count, sizeof(int));
    }
    if (!obs || !actions || !steps || !episode_returns || !episode_counts ||
        (agent_count == 1 && (!reset_envs || !reset_obs || !reset_seeds || !reset_indices))) {
        fprintf(stderr, "allocation failed\n");
        free(obs);
        free(actions);
        free(steps);
        free(episode_returns);
        free(episode_counts);
        free(reset_envs);
        free(reset_obs);
        free(reset_seeds);
        free(reset_indices);
        for (int i = 0; i < env_count; i++) {
            tfrl_env_destroy(envs[i]);
        }
        free(envs);
        return 1;
    }
    if (agent_count == 1) {
        tfrl_env_reset_batch(envs, env_count, (uint64_t)seed, obs);
    } else {
        for (int i = 0; i < env_count; i++) {
            int got = tfrl_env_reset_multi(envs[i], (uint64_t)(seed + i), &obs[i * agent_count], agent_count);
            if (got != agent_count) {
                fprintf(stderr, "env reset failed\n");
                for (int a = 0; a < agent_count; a++) {
                    tfrl_algo_destroy(&algos[a]);
                }
                for (int j = 0; j < env_count; j++) {
                    tfrl_env_destroy(envs[j]);
                }
                free(envs);
                free(obs);
                free(actions);
                free(steps);
                free(episode_returns);
                free(episode_counts);
                return 1;
            }
        }
    }

    if (cfg->mode == TFRL_MODE_TRAIN && cfg->algo && strcmp(cfg->algo, "a3c") == 0) {
        if (agent_count != 1) {
            fprintf(stderr, "a3c requires single-agent environments\n");
            return 1;
        }
        if (cfg->render || cfg->trace_out) {
            fprintf(stderr, "a3c async runner disables render/trace output\n");
        }
        int rc = run_a3c_async(cfg, envs, env_count, &algos[0]);
        if (writer) tfrl_trace_writer_close(writer);
        if (viewer) tfrl_viewer_destroy(viewer);
        free(render_buf);
        for (int a = 0; a < policy_count; a++) {
            tfrl_algo_destroy(&algos[a]);
        }
        for (int i = 0; i < env_count; i++) {
            tfrl_env_destroy(envs[i]);
        }
        free(envs);
        free(obs);
        free(actions);
        free(steps);
        free(episode_returns);
        free(episode_counts);
        return rc;
    }

    if (cfg->mode == TFRL_MODE_TRAIN && cfg->algo && strcmp(cfg->algo, "impala") == 0 && cfg->actor_count > 1) {
        if (agent_count != 1) {
            fprintf(stderr, "impala split requires single-agent environments\n");
            return 1;
        }
        if (cfg->render || cfg->trace_out) {
            fprintf(stderr, "impala split runner disables render/trace output\n");
        }
        int rc = run_impala_split(cfg, envs, env_count, &algos[0]);
        if (writer) tfrl_trace_writer_close(writer);
        if (viewer) tfrl_viewer_destroy(viewer);
        free(render_buf);
        for (int a = 0; a < policy_count; a++) {
            tfrl_algo_destroy(&algos[a]);
        }
        for (int i = 0; i < env_count; i++) {
            tfrl_env_destroy(envs[i]);
        }
        free(envs);
        free(obs);
        free(actions);
        free(steps);
        free(episode_returns);
        free(episode_counts);
        return rc;
    }

    if (cfg->mode == TFRL_MODE_TRAIN) {
        if (cfg->mp_actors > 0) {
            for (int i = 0; i < env_count; i++) {
                tfrl_env_destroy(envs[i]);
            }
            free(envs);
            if (cfg->algo && strcmp(cfg->algo, "ppo") == 0) {
                return run_mp_ppo(cfg);
            }
            return run_mp_dqn(cfg);
        }
        int total_steps = cfg->steps > 0 ? cfg->steps : 1000;
        int thread_count = cfg->threads > 0 ? cfg->threads : env_count;
        if (cfg->deterministic) thread_count = 1;
        int use_dashboard = dashboard_enabled();
        const int log_every = cfg->log_every;
        const int use_log = !use_dashboard && log_every > 0;
        int use_profile = cfg->profile || (cfg->profile_json && cfg->profile_json[0]) || use_dashboard;
        int dashboard_every = cfg->log_every > 0 ? cfg->log_every : 100;
        double start_time = use_profile ? now_seconds() : 0.0;
        double env_seconds = 0.0;
        double update_seconds = 0.0;
        double render_seconds = 0.0;
        double total_return_sum = 0.0;
        double last_return = 0.0;
        int total_episode_count = 0;
        if (use_dashboard) {
            fprintf(stdout, "\033[?25l");
            fflush(stdout);
        }
        tfrl_obs *obs_batch = NULL;
        tfrl_action *act_batch = NULL;
        if (agent_count > 1 && !cfg->share_policy) {
            obs_batch = (tfrl_obs *)calloc((size_t)env_count, sizeof(tfrl_obs));
            act_batch = (tfrl_action *)calloc((size_t)env_count, sizeof(tfrl_action));
            if (!obs_batch || !act_batch) {
                fprintf(stderr, "allocation failed\n");
                free(obs_batch);
                free(act_batch);
                for (int a = 0; a < agent_count; a++) {
                    tfrl_algo_destroy(&algos[a]);
                }
                for (int i = 0; i < env_count; i++) {
                    tfrl_env_destroy(envs[i]);
                }
                free(envs);
                free(obs);
                free(actions);
                free(steps);
                free(episode_returns);
                free(episode_counts);
                return 1;
            }
        }
        for (int step = 0; step < total_steps; step++) {
            if (agent_count == 1 || cfg->share_policy) {
                if (algos[0].vtable && algos[0].vtable->act_env) {
                    for (int i = 0; i < env_count; i++) {
                        actions[i] = tfrl_algo_act(&algos[0], envs[i], obs[i]);
                    }
                } else {
                    tfrl_algo_act_batch(&algos[0], obs, total_agents, actions);
                }
            } else {
                for (int a = 0; a < agent_count; a++) {
                    if (algos[a].vtable && algos[a].vtable->act_env) {
                        for (int i = 0; i < env_count; i++) {
                            int idx = i * agent_count + a;
                            actions[idx] = tfrl_algo_act(&algos[a], envs[i], obs[idx]);
                        }
                    } else {
                        for (int i = 0; i < env_count; i++) {
                            obs_batch[i] = obs[i * agent_count + a];
                        }
                        tfrl_algo_act_batch(&algos[a], obs_batch, env_count, act_batch);
                        for (int i = 0; i < env_count; i++) {
                            actions[i * agent_count + a] = act_batch[i];
                        }
                    }
                }
            }
            {
                double t0 = use_profile ? now_seconds() : 0.0;
                if (agent_count == 1) {
                    step_envs_threaded(envs, env_count, actions, steps, thread_count);
                } else {
                    for (int i = 0; i < env_count; i++) {
                        int got = tfrl_env_step_multi(envs[i], &actions[i * agent_count], agent_count,
                                                      &steps[i * agent_count], agent_count);
                        if (got != agent_count) {
                            fprintf(stderr, "env step failed\n");
                            step = total_steps;
                            break;
                        }
                    }
                }
                if (use_profile) env_seconds += now_seconds() - t0;
            }
            {
                double t0 = use_profile ? now_seconds() : 0.0;
                for (int i = 0; i < env_count; i++) {
                    for (int a = 0; a < agent_count; a++) {
                        int idx = i * agent_count + a;
                        tfrl_transition transition = {
                            .obs = obs[idx],
                            .action = actions[idx],
                            .reward = steps[idx].reward,
                            .next_obs = steps[idx].observation,
                            .done = steps[idx].done,
                        };
                        int algo_idx = cfg->share_policy ? 0 : a;
                        algos[algo_idx].vtable->update(algos[algo_idx].ctx, &transition);
                        obs[idx] = steps[idx].observation;
                        episode_returns[idx] += steps[idx].reward;
                    }
                }
                if (use_profile) update_seconds += now_seconds() - t0;
            }

            if ((viewer || writer) && should_render(step, cfg->render_every)) {
                double t0 = use_profile ? now_seconds() : 0.0;
                size_t written = tfrl_env_render_write(envs[render_idx], render_buf, render_buf_len);
                if (writer && written > 0) {
                    tfrl_trace_writer_write(writer, render_buf, written);
                }
                if (viewer && written > 0) {
                    tfrl_viewer_draw(viewer, render_buf, written);
                }
                if (use_profile) render_seconds += now_seconds() - t0;
            }

            int reset_count = 0;
            for (int i = 0; i < env_count; i++) {
                int env_done = 0;
                for (int a = 0; a < agent_count; a++) {
                    int idx = i * agent_count + a;
                    if (steps[idx].done) env_done = 1;
                }
                if (env_done) {
                    for (int a = 0; a < agent_count; a++) {
                        int idx = i * agent_count + a;
                        episode_counts[idx] += 1;
                        total_episode_count += 1;
                        total_return_sum += episode_returns[idx];
                        last_return = episode_returns[idx];
                        if (use_log && (episode_counts[idx] % log_every) == 0) {
                            fprintf(stdout, "env=%d agent=%d episode=%d return=%.4f\n", i, a, episode_counts[idx], episode_returns[idx]);
                        }
                        episode_returns[idx] = 0.0;
                    }
                    if (agent_count == 1) {
                        reset_envs[reset_count] = envs[i];
                        reset_seeds[reset_count] = (uint64_t)(cfg->seed + i);
                        reset_indices[reset_count] = i;
                        reset_count++;
                    } else {
                        int got = tfrl_env_reset_multi(envs[i], (uint64_t)(cfg->seed + i), &obs[i * agent_count], agent_count);
                        if (got != agent_count) {
                            fprintf(stderr, "env reset failed\n");
                            break;
                        }
                    }
                }
            }
            if (agent_count == 1 && reset_count > 0) {
                tfrl_env_reset_batch_seeds(reset_envs, reset_count, reset_seeds, reset_obs);
                for (int i = 0; i < reset_count; i++) {
                    int idx = reset_indices[i];
                    obs[idx] = reset_obs[i];
                }
            }
            if (viewer && tfrl_viewer_should_close(viewer)) {
                break;
            }
            if (use_dashboard && (step % dashboard_every) == 0) {
                double elapsed = now_seconds() - start_time;
                dashboard_render(cfg, step, total_steps, env_count, agent_count,
                                 total_episode_count, total_return_sum, last_return, elapsed,
                                 env_seconds, update_seconds, render_seconds);
            }
        }
        if (use_dashboard) {
            double elapsed = now_seconds() - start_time;
            dashboard_render(cfg, total_steps - 1, total_steps, env_count, agent_count,
                             total_episode_count, total_return_sum, last_return, elapsed,
                             env_seconds, update_seconds, render_seconds);
            fprintf(stdout, "\033[?25h");
            fflush(stdout);
        }
        if (use_profile) {
            double elapsed = now_seconds() - start_time;
            long long total_env_steps = (long long)total_steps * env_count * agent_count;
            double steps_per_sec = elapsed > 0.0 ? (double)total_steps / elapsed : 0.0;
            double samples_per_sec = elapsed > 0.0 ? (double)total_env_steps / elapsed : 0.0;
            double other = elapsed - env_seconds - update_seconds - render_seconds;
            if (other < 0.0) other = 0.0;
            if (cfg->profile || (cfg->profile_json && cfg->profile_json[0])) {
                fprintf(stdout,
                        "profile: steps=%d envs=%d agents=%d elapsed=%.2fs sps=%.1f samples=%.1f "
                        "env=%.2fs update=%.2fs render=%.2fs other=%.2fs\n",
                        total_steps, env_count, agent_count, elapsed, steps_per_sec, samples_per_sec,
                        env_seconds, update_seconds, render_seconds, other);
            }
            if (cfg->profile_json && cfg->profile_json[0]) {
                profile_write_json(cfg, cfg->profile_json, total_steps, env_count, agent_count,
                                   total_episode_count, total_return_sum, elapsed, env_seconds,
                                   update_seconds, render_seconds);
            }
        }
        if (env_count > 1 || agent_count > 1) {
            double sum = 0.0;
            int count = 0;
            for (int i = 0; i < total_agents; i++) {
                sum += episode_returns[i];
                count += episode_counts[i] > 0 ? episode_counts[i] : 0;
            }
            if (count > 0) {
                fprintf(stdout, "avg_return=%.4f episodes=%d\n", sum / (double)count, count);
            }
        }
        free(obs_batch);
        free(act_batch);
    } else if (cfg->mode == TFRL_MODE_EVAL) {
        int episodes = cfg->episodes > 0 ? cfg->episodes : 10;
        for (int ep = 0; ep < episodes; ep++) {
            if (tfrl_env_reset_multi(envs[0], (uint64_t)seed, obs, agent_count) != agent_count) {
                fprintf(stderr, "env reset failed\n");
                break;
            }
            int done = 0;
            int step = 0;
            while (!done) {
                for (int a = 0; a < agent_count; a++) {
                    int algo_idx = cfg->share_policy ? 0 : a;
                    actions[a] = tfrl_algo_act(&algos[algo_idx], envs[0], obs[a]);
                }
                if (tfrl_env_step_multi(envs[0], actions, agent_count, steps, agent_count) != agent_count) {
                    fprintf(stderr, "env step failed\n");
                    done = 1;
                } else {
                    for (int a = 0; a < agent_count; a++) {
                        obs[a] = steps[a].observation;
                        if (steps[a].done) done = 1;
                    }
                }

                if ((viewer || writer) && should_render(step, cfg->render_every)) {
                    size_t written = tfrl_env_render_write(envs[render_idx], render_buf, render_buf_len);
                    if (writer && written > 0) {
                        tfrl_trace_writer_write(writer, render_buf, written);
                    }
                    if (viewer && written > 0) {
                        tfrl_viewer_draw(viewer, render_buf, written);
                    }
                }
                step++;
                if (viewer && tfrl_viewer_should_close(viewer)) {
                    ep = episodes;
                    break;
                }
            }
        }
    }

    if (writer && policy_count > 0) {
        char diag_update[256];
        diag_update[0] = '\0';
        if (tfrl_algo_diagnostics(&algos[0], diag_update, sizeof(diag_update))) {
            char trace_meta_update[768];
            build_trace_meta(trace_meta_update, sizeof(trace_meta_update), &algo_cfg, agent_count, cfg->share_policy ? 1 : 0, diag_update);
            tfrl_trace_writer_update_meta(writer, trace_meta_update);
        }
    }
    if (writer) tfrl_trace_writer_close(writer);
    if (viewer) tfrl_viewer_destroy(viewer);
    free(render_buf);
    if (cfg->save_path) {
        if (policy_count == 1) {
            tfrl_algo_save(&algos[0], cfg->save_path);
        } else {
            for (int a = 0; a < policy_count; a++) {
                char path[256];
                snprintf(path, sizeof(path), "%s.agent%d", cfg->save_path, a);
                tfrl_algo_save(&algos[a], path);
            }
        }
    }
    for (int a = 0; a < policy_count; a++) {
        tfrl_algo_destroy(&algos[a]);
    }
    for (int i = 0; i < env_count; i++) {
        tfrl_env_destroy(envs[i]);
    }
    free(envs);
    free(obs);
    free(actions);
    free(steps);
    free(episode_returns);
    free(episode_counts);
    free(reset_envs);
    free(reset_obs);
    free(reset_seeds);
    free(reset_indices);
    return 0;
}
