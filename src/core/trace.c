#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "render_snapshot.h"
#include "trace.h"

#define TFRL_TRACE_MAGIC "TFT1"
#define TFRL_TRACE_VERSION 1

struct tfrl_trace_writer {
    FILE *fp;
};

struct tfrl_trace_reader {
    FILE *fp;
};

static int write_header(FILE *fp) {
    char magic[4] = TFRL_TRACE_MAGIC;
    uint32_t version = TFRL_TRACE_VERSION;
    if (fwrite(magic, 1, sizeof(magic), fp) != sizeof(magic)) return 0;
    if (fwrite(&version, sizeof(version), 1, fp) != 1) return 0;
    return 1;
}

static int read_header(FILE *fp) {
    char magic[4];
    uint32_t version = 0;
    if (fread(magic, 1, sizeof(magic), fp) != sizeof(magic)) return 0;
    if (memcmp(magic, TFRL_TRACE_MAGIC, sizeof(magic)) != 0) return 0;
    if (fread(&version, sizeof(version), 1, fp) != 1) return 0;
    return version == TFRL_TRACE_VERSION;
}

tfrl_trace_writer *tfrl_trace_writer_open(const char *path) {
    if (!path) return NULL;
    FILE *fp = fopen(path, "wb");
    if (!fp) return NULL;
    if (!write_header(fp)) {
        fclose(fp);
        return NULL;
    }
    tfrl_trace_writer *writer = (tfrl_trace_writer *)calloc(1, sizeof(tfrl_trace_writer));
    if (!writer) {
        fclose(fp);
        return NULL;
    }
    writer->fp = fp;
    return writer;
}

void tfrl_trace_writer_close(tfrl_trace_writer *writer) {
    if (!writer) return;
    if (writer->fp) fclose(writer->fp);
    free(writer);
}

int tfrl_trace_writer_write(tfrl_trace_writer *writer, const void *buffer, size_t len) {
    if (!writer || !writer->fp || !buffer || len == 0) return 0;
    return fwrite(buffer, 1, len, writer->fp) == len;
}

tfrl_trace_reader *tfrl_trace_reader_open(const char *path) {
    if (!path) return NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (!read_header(fp)) {
        fclose(fp);
        return NULL;
    }
    tfrl_trace_reader *reader = (tfrl_trace_reader *)calloc(1, sizeof(tfrl_trace_reader));
    if (!reader) {
        fclose(fp);
        return NULL;
    }
    reader->fp = fp;
    return reader;
}

void tfrl_trace_reader_close(tfrl_trace_reader *reader) {
    if (!reader) return;
    if (reader->fp) fclose(reader->fp);
    free(reader);
}

int tfrl_trace_reader_next(tfrl_trace_reader *reader, void *buffer, size_t buffer_len, size_t *out_len) {
    if (!reader || !reader->fp || !buffer || buffer_len == 0) return 0;
    tfrl_render_snapshot_header header;
    if (fread(&header, sizeof(header), 1, reader->fp) != 1) return 0;
    size_t total = sizeof(header) + header.payload_bytes;
    if (buffer_len < total) return 0;
    memcpy(buffer, &header, sizeof(header));
    if (fread((char *)buffer + sizeof(header), 1, header.payload_bytes, reader->fp) != header.payload_bytes) return 0;
    if (out_len) *out_len = total;
    return 1;
}
