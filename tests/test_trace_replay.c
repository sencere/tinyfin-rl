#include <stdio.h>
#include <string.h>

#include "core/env_api.h"
#include "core/trace.h"

static int record(const char *path, const char *name, unsigned long long seed) {
    tfrl_env *env = tfrl_env_create(&(tfrl_env_config){.name = name, .seed = seed});
    if (!env) return 0;
    tfrl_trace_writer *writer = tfrl_trace_writer_open_with_meta(path, name);
    size_t cap = tfrl_env_render_bytes_needed(env);
    unsigned char buffer[8192];
    int ok = writer != NULL && cap <= sizeof(buffer);
    if (ok) {
        tfrl_env_reset(env, seed);
        for (int i = 0; i < 12 && ok; i++) {
            tfrl_env_step(env, (tfrl_action){.index = i % 4});
            size_t written = tfrl_env_render_write(env, buffer, sizeof(buffer));
            ok = written > 0 && tfrl_trace_writer_write(writer, buffer, written);
        }
    }
    tfrl_trace_writer_close(writer);
    tfrl_env_destroy(env);
    return ok;
}

static int compare(const char *left_path, const char *right_path, const char *name) {
    tfrl_trace_reader *left = tfrl_trace_reader_open(left_path);
    tfrl_trace_reader *right = tfrl_trace_reader_open(right_path);
    unsigned char left_frame[8192], right_frame[8192];
    size_t left_len = 0, right_len = 0;
    int frames = 0;
    int ok = left && right && strcmp(tfrl_trace_reader_meta(left), name) == 0;
    while (ok) {
        int have_left = tfrl_trace_reader_next(left, left_frame, sizeof(left_frame), &left_len);
        int have_right = tfrl_trace_reader_next(right, right_frame, sizeof(right_frame), &right_len);
        if (!have_left || !have_right) {
            ok = have_left == have_right;
            break;
        }
        ok = left_len == right_len && memcmp(left_frame, right_frame, left_len) == 0;
        frames++;
    }
    tfrl_trace_reader_close(left);
    tfrl_trace_reader_close(right);
    return ok && frames == 12;
}

int main(void) {
    const char *names[] = {"lineworld", "maze_rooms", "coin_maze"};
    int ok = 1;
    for (int i = 0; i < 3; i++) {
        char a[64], b[64];
        snprintf(a, sizeof(a), "/tmp/tfrl_trace_a_%d.tft", i);
        snprintf(b, sizeof(b), "/tmp/tfrl_trace_b_%d.tft", i);
        ok = record(a, names[i], 42) && record(b, names[i], 42) && compare(a, b, names[i]) && ok;
        remove(a);
        remove(b);
    }
    if (!ok) fprintf(stderr, "test_trace_replay: deterministic trace comparison failed\n");
    return ok ? 0 : 1;
}
