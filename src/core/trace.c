#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "render_snapshot.h"
#include "trace.h"

#define TFRL_TRACE_MAGIC "TFT1"
#define TFRL_TRACE_VERSION 2

struct tfrl_trace_writer {
    FILE *fp;
    long meta_offset;
    uint32_t meta_len;
};

struct tfrl_trace_reader {
    FILE *fp;
    char *meta;
};

static int write_header(FILE *fp, const char *meta, long *out_meta_offset, uint32_t *out_meta_len) {
    char magic[4] = TFRL_TRACE_MAGIC;
    uint32_t version = TFRL_TRACE_VERSION;
    uint32_t meta_len = meta ? (uint32_t)strlen(meta) : 0;
    if (out_meta_offset) *out_meta_offset = -1;
    if (out_meta_len) *out_meta_len = meta_len;
    if (fwrite(magic, 1, sizeof(magic), fp) != sizeof(magic)) return 0;
    if (fwrite(&version, sizeof(version), 1, fp) != 1) return 0;
    if (version >= 2) {
        if (fwrite(&meta_len, sizeof(meta_len), 1, fp) != 1) return 0;
        if (out_meta_offset) {
            long pos = ftell(fp);
            if (pos < 0) return 0;
            *out_meta_offset = pos;
        }
        if (meta_len > 0 && fwrite(meta, 1, meta_len, fp) != meta_len) return 0;
    }
    return 1;
}

static int read_header(FILE *fp, char **out_meta) {
    char magic[4];
    uint32_t version = 0;
    if (fread(magic, 1, sizeof(magic), fp) != sizeof(magic)) return 0;
    if (memcmp(magic, TFRL_TRACE_MAGIC, sizeof(magic)) != 0) return 0;
    if (fread(&version, sizeof(version), 1, fp) != 1) return 0;
    if (version == 1) {
        if (out_meta) *out_meta = NULL;
        return 1;
    }
    if (version == 2) {
        uint32_t meta_len = 0;
        if (fread(&meta_len, sizeof(meta_len), 1, fp) != 1) return 0;
        if (meta_len > 0 && out_meta) {
            char *meta = (char *)calloc((size_t)meta_len + 1, 1);
            if (!meta) return 0;
            if (fread(meta, 1, meta_len, fp) != meta_len) {
                free(meta);
                return 0;
            }
            meta[meta_len] = '\0';
            *out_meta = meta;
        } else if (meta_len > 0) {
            if (fseek(fp, (long)meta_len, SEEK_CUR) != 0) return 0;
        } else if (out_meta) {
            *out_meta = NULL;
        }
        return 1;
    }
    return 0;
}

tfrl_trace_writer *tfrl_trace_writer_open(const char *path) {
    return tfrl_trace_writer_open_with_meta(path, NULL);
}

tfrl_trace_writer *tfrl_trace_writer_open_with_meta(const char *path, const char *meta) {
    if (!path) return NULL;
    FILE *fp = fopen(path, "wb");
    if (!fp) return NULL;
    long meta_offset = -1;
    uint32_t meta_len = 0;
    if (!write_header(fp, meta, &meta_offset, &meta_len)) {
        fclose(fp);
        return NULL;
    }
    tfrl_trace_writer *writer = (tfrl_trace_writer *)calloc(1, sizeof(tfrl_trace_writer));
    if (!writer) {
        fclose(fp);
        return NULL;
    }
    writer->fp = fp;
    writer->meta_offset = meta_offset;
    writer->meta_len = meta_len;
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

int tfrl_trace_writer_update_meta(tfrl_trace_writer *writer, const char *meta) {
    if (!writer || !writer->fp || !meta) return 0;
    if (writer->meta_len == 0 || writer->meta_offset < 0) return 0;
    size_t meta_len = strlen(meta);
    if (meta_len != writer->meta_len) return 0;
    if (fseek(writer->fp, writer->meta_offset, SEEK_SET) != 0) return 0;
    if (fwrite(meta, 1, writer->meta_len, writer->fp) != writer->meta_len) return 0;
    fflush(writer->fp);
    return 1;
}

tfrl_trace_reader *tfrl_trace_reader_open(const char *path) {
    if (!path) return NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    char *meta = NULL;
    if (!read_header(fp, &meta)) {
        fclose(fp);
        return NULL;
    }
    tfrl_trace_reader *reader = (tfrl_trace_reader *)calloc(1, sizeof(tfrl_trace_reader));
    if (!reader) {
        free(meta);
        fclose(fp);
        return NULL;
    }
    reader->fp = fp;
    reader->meta = meta;
    return reader;
}

void tfrl_trace_reader_close(tfrl_trace_reader *reader) {
    if (!reader) return;
    if (reader->fp) fclose(reader->fp);
    free(reader->meta);
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

const char *tfrl_trace_reader_meta(const tfrl_trace_reader *reader) {
    return reader ? reader->meta : NULL;
}
