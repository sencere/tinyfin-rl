#include "trace.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRACE_META_PREFIX "#META "

typedef struct {
    uint32_t len;
} trace_frame_header;

struct tfrl_trace_writer {
    FILE *f;
    char *path;
    char *meta;
    long meta_end_offset;
};

struct tfrl_trace_reader {
    FILE *f;
    char *meta;
};

static char *dup_cstr(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *out = (char *)calloc(n + 1, 1);
    if (!out) return NULL;
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static int write_meta_line(FILE *f, const char *meta, long *out_end_offset) {
    if (!f) return 0;
    const char *m = meta ? meta : "";
    if (fprintf(f, TRACE_META_PREFIX "%s\n", m) < 0) return 0;
    if (out_end_offset) {
        long pos = ftell(f);
        *out_end_offset = pos;
    }
    return 1;
}

tfrl_trace_writer *tfrl_trace_writer_open_with_meta(const char *path, const char *meta) {
    if (!path) return NULL;
    FILE *f = fopen(path, "wb");
    if (!f) return NULL;
    tfrl_trace_writer *w = (tfrl_trace_writer *)calloc(1, sizeof(*w));
    if (!w) {
        fclose(f);
        return NULL;
    }
    w->f = f;
    w->path = dup_cstr(path);
    w->meta = dup_cstr(meta ? meta : "");
    if (!write_meta_line(w->f, w->meta, &w->meta_end_offset)) {
        tfrl_trace_writer_close(w);
        return NULL;
    }
    return w;
}

int tfrl_trace_writer_write(tfrl_trace_writer *writer, const void *data, size_t len) {
    if (!writer || !writer->f || !data || len == 0) return 0;
    if (len > 0xFFFFFFFFu) return 0;
    trace_frame_header hdr = {.len = (uint32_t)len};
    if (fwrite(&hdr, sizeof(hdr), 1, writer->f) != 1) return 0;
    if (fwrite(data, 1, len, writer->f) != len) return 0;
    return 1;
}

int tfrl_trace_writer_update_meta(tfrl_trace_writer *writer, const char *meta) {
    if (!writer || !writer->path) return 0;
    const char *new_meta = meta ? meta : "";

    FILE *in = fopen(writer->path, "rb");
    if (!in) return 0;

    char tmp_path[512];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", writer->path);
    FILE *out = fopen(tmp_path, "wb");
    if (!out) {
        fclose(in);
        return 0;
    }

    long old_meta_end = writer->meta_end_offset;
    if (old_meta_end <= 0) {
        // Fallback: read the first line to find the end of meta.
        int c = 0;
        old_meta_end = 0;
        while ((c = fgetc(in)) != EOF) {
            old_meta_end++;
            if (c == '\n') break;
        }
    } else {
        if (fseek(in, old_meta_end, SEEK_SET) != 0) {
            fclose(in);
            fclose(out);
            remove(tmp_path);
            return 0;
        }
    }

    long new_meta_end = 0;
    if (!write_meta_line(out, new_meta, &new_meta_end)) {
        fclose(in);
        fclose(out);
        remove(tmp_path);
        return 0;
    }

    char buffer[64 * 1024];
    size_t nread = 0;
    while ((nread = fread(buffer, 1, sizeof(buffer), in)) > 0) {
        if (fwrite(buffer, 1, nread, out) != nread) {
            fclose(in);
            fclose(out);
            remove(tmp_path);
            return 0;
        }
    }

    fclose(in);
    fclose(out);

    if (rename(tmp_path, writer->path) != 0) {
        remove(tmp_path);
        return 0;
    }

    free(writer->meta);
    writer->meta = dup_cstr(new_meta);
    writer->meta_end_offset = new_meta_end;

    // Reopen for appending at end of file.
    if (writer->f) fclose(writer->f);
    writer->f = fopen(writer->path, "ab");
    return writer->f != NULL;
}

void tfrl_trace_writer_close(tfrl_trace_writer *writer) {
    if (!writer) return;
    if (writer->f) fclose(writer->f);
    free(writer->path);
    free(writer->meta);
    free(writer);
}

tfrl_trace_reader *tfrl_trace_reader_open(const char *path) {
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    tfrl_trace_reader *r = (tfrl_trace_reader *)calloc(1, sizeof(*r));
    if (!r) {
        fclose(f);
        return NULL;
    }
    r->f = f;

    char meta_line[2048];
    if (!fgets(meta_line, sizeof(meta_line), r->f)) {
        tfrl_trace_reader_close(r);
        return NULL;
    }
    const char *meta_start = meta_line;
    if (strncmp(meta_line, TRACE_META_PREFIX, strlen(TRACE_META_PREFIX)) == 0) {
        meta_start = meta_line + (int)strlen(TRACE_META_PREFIX);
    }
    size_t len = strlen(meta_start);
    while (len > 0 && (meta_start[len - 1] == '\n' || meta_start[len - 1] == '\r')) {
        len--;
    }
    r->meta = (char *)calloc(len + 1, 1);
    if (!r->meta) {
        tfrl_trace_reader_close(r);
        return NULL;
    }
    memcpy(r->meta, meta_start, len);
    r->meta[len] = '\0';
    return r;
}

const char *tfrl_trace_reader_meta(tfrl_trace_reader *reader) {
    return reader ? reader->meta : NULL;
}

int tfrl_trace_reader_next(tfrl_trace_reader *reader, void *buffer, size_t buffer_cap,
                           size_t *out_len) {
    if (!reader || !reader->f || !buffer || buffer_cap == 0) return 0;
    trace_frame_header hdr = {0};
    if (fread(&hdr, sizeof(hdr), 1, reader->f) != 1) return 0;
    size_t need = (size_t)hdr.len;
    if (need == 0 || need > buffer_cap) return 0;
    if (fread(buffer, 1, need, reader->f) != need) return 0;
    if (out_len) *out_len = need;
    return 1;
}

void tfrl_trace_reader_close(tfrl_trace_reader *reader) {
    if (!reader) return;
    if (reader->f) fclose(reader->f);
    free(reader->meta);
    free(reader);
}
