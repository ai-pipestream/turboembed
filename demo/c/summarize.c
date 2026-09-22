/* SPDX-License-Identifier: Apache-2.0
 *
 * Turbo C summarizer: stream a summary of stdin (or a file) from a
 * generative bundle through the push form of the generation API,
 * turbo_generate, which calls back once per chunk. The callback returns
 * TURBO_STREAM_STOP to cancel; here it stops when a --max-chars budget is
 * reached, which makes the cancellation path visible.
 *
 *   turbo-summarize [--provider-lib <so>] [--provider <id> --ordinal <n>] --bundle <dir>
 *                   [--max-new-tokens 160] [--max-chars 0] [file]
 */
#include "demo_args.h"
#include "turbo/turbo.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static turbo_text text_of(const char *s) {
    turbo_text t;
    t.ptr = s;
    t.len = (uint64_t)strlen(s);
    return t;
}

static int fail(const char *what, int32_t rc, const turbo_error *err) {
    fprintf(stderr, "%s failed: %s (%d)%s%s\n", what, turbo_status_name(rc), rc, err->message[0] ? ": " : "", err->message);
    return 1;
}

#define CHECK(what, expr)                                                                                          \
    do {                                                                                                           \
        int32_t rc_ = (expr);                                                                                      \
        if (rc_ != TURBO_OK) {                                                                                     \
            return fail(what, rc_, &err);                                                                          \
        }                                                                                                          \
    } while (0)

typedef struct {
    size_t chars;
    size_t max_chars;
    uint32_t chunks;
    uint32_t finish;
    uint32_t generated;
    uint32_t prompt_tokens;
    int stopped; /* the callback asked turbo_generate to stop */
} stream_state;

static uint32_t on_chunk(void *user_data, const turbo_generation_chunk *chunk) {
    stream_state *st = user_data;
    fwrite(chunk->text.ptr, 1, (size_t)chunk->text.len, stdout);
    fflush(stdout);
    st->chars += (size_t)chunk->text.len;
    st->generated = chunk->generated_tokens;
    /* turbo_types.h reports prompt_tokens on the first chunk; a provider
     * that honors that literally leaves it 0 on every later one. */
    if (st->chunks == 0) {
        st->prompt_tokens = chunk->prompt_tokens;
    }
    st->chunks += 1;
    if (chunk->done) {
        st->finish = chunk->finish_reason;
    }
    if (st->max_chars != 0 && st->chars >= st->max_chars && !chunk->done) {
        /* The push form cancels and releases the generation before it
         * returns; the callback is not called again, so the stop is
         * recorded here rather than read from a final chunk. */
        st->stopped = 1;
        return TURBO_STREAM_STOP;
    }
    return TURBO_STREAM_CONTINUE;
}

/* Read all of `f`. On failure returns NULL and sets *status: 1 for out of
 * memory, 2 for a read error. A short fread is not by itself EOF, so the
 * error indicator is checked before the bytes read so far are accepted. */
static char *read_all(FILE *f, int *status) {
    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    *status = 0;
    if (buf == NULL) { *status = 1; return NULL; }
    size_t n;
    while ((n = fread(buf + len, 1, cap - len - 1, f)) > 0) {
        len += n;
        if (cap - len < 1024) {
            char *grown = realloc(buf, cap *= 2);
            if (grown == NULL) { free(buf); *status = 1; return NULL; }
            buf = grown;
        }
    }
    if (ferror(f)) { free(buf); *status = 2; return NULL; }
    buf[len] = '\0';
    return buf;
}

int main(int argc, char **argv) {
    const char *provider_lib = NULL, *provider = NULL, *bundle = NULL, *file = NULL;
    uint32_t ordinal = 0, max_new = 160;
    size_t max_chars = 0;
    int have_ordinal = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--provider-lib") == 0) provider_lib = demo_value(argc, argv, &i);
        else if (strcmp(argv[i], "--provider") == 0) provider = demo_value(argc, argv, &i);
        else if (strcmp(argv[i], "--ordinal") == 0) { ordinal = demo_u32("--ordinal", demo_value(argc, argv, &i)); have_ordinal = 1; }
        else if (strcmp(argv[i], "--bundle") == 0) bundle = demo_value(argc, argv, &i);
        else if (strcmp(argv[i], "--max-new-tokens") == 0) max_new = demo_u32("--max-new-tokens", demo_value(argc, argv, &i));
        else if (strcmp(argv[i], "--max-chars") == 0) max_chars = (size_t)demo_u64("--max-chars", demo_value(argc, argv, &i), (unsigned long long)SIZE_MAX);
        else {
            demo_reject_unknown_flag(argv[i]);
            if (file != NULL) {
                fprintf(stderr, "this demo summarizes one file; %s is a second one\n", argv[i]);
                return 2;
            }
            file = argv[i];
        }
    }
    if (bundle == NULL || (provider == NULL) != (have_ordinal == 0)) {
        fprintf(stderr, "usage: %s [--provider-lib <so>] [--provider <id> --ordinal <n>] --bundle <dir> [--max-new-tokens n] [--max-chars n] [file]\n", argv[0]);
        return 2;
    }
    FILE *in = file ? fopen(file, "rb") : stdin;
    if (in == NULL) { perror(file); return 2; }
    int read_status = 0;
    char *document = read_all(in, &read_status);
    if (file) fclose(in);
    if (read_status == 1) { fprintf(stderr, "out of memory reading %s\n", file ? file : "stdin"); return 1; }
    if (read_status == 2) { fprintf(stderr, "read error on %s; the document is incomplete\n", file ? file : "stdin"); return 1; }
    if (document[0] == '\0') { fprintf(stderr, "no text to summarize\n"); free(document); return 2; }

    turbo_error err;
    memset(&err, 0, sizeof err);
    err.struct_size = (uint32_t)sizeof err;
    turbo_runtime_desc rd;
    memset(&rd, 0, sizeof rd);
    rd.struct_size = (uint32_t)sizeof rd;
    turbo_text lib_text;
    if (provider_lib) { lib_text = text_of(provider_lib); rd.n_provider_paths = 1; rd.provider_paths = &lib_text; }
    turbo_runtime *rt = NULL;
    CHECK("turbo_runtime_create", turbo_runtime_create(&rd, &rt, &err));
    uint32_t dev = 0;
    if (provider) {
        turbo_device_selector sel;
        memset(&sel, 0, sizeof sel);
        sel.struct_size = (uint32_t)sizeof sel;
        sel.policy = TURBO_SELECT_EXPLICIT;
        sel.provider_id = text_of(provider);
        sel.ordinal = ordinal;
        CHECK("turbo_runtime_select_device", turbo_runtime_select_device(rt, &sel, &dev, &err));
    } else {
        CHECK("turbo_runtime_select_device", turbo_runtime_select_device(rt, NULL, &dev, &err));
    }
    turbo_device_info di;
    memset(&di, 0, sizeof di);
    di.struct_size = (uint32_t)sizeof di;
    CHECK("turbo_runtime_device_info", turbo_runtime_device_info(rt, dev, &di, &err));
    turbo_context *ctx = NULL;
    CHECK("turbo_context_create", turbo_context_create(rt, dev, NULL, &ctx, &err));
    turbo_model *model = NULL;
    CHECK("turbo_model_load", turbo_model_load(ctx, text_of(bundle), NULL, &model, &err));
    turbo_model_info mi;
    memset(&mi, 0, sizeof mi);
    mi.struct_size = (uint32_t)sizeof mi;
    CHECK("turbo_model_get_info", turbo_model_get_info(model, &mi, &err));
    if (mi.kind != TURBO_MODEL_GENERATIVE) { fprintf(stderr, "%s is not a generative bundle (kind %u)\n", bundle, mi.kind); return 1; }
    fprintf(stderr, "model: %s on %s (%s:%u)\n", mi.model_id, di.name, di.provider_id, di.ordinal);

    turbo_generate_desc gd;
    memset(&gd, 0, sizeof gd);
    gd.struct_size = (uint32_t)sizeof gd;
    gd.max_new_tokens = max_new;
    turbo_message messages[2];
    messages[0].role = text_of("system");
    messages[0].content = text_of("You summarize text. Reply with a summary of at most three sentences and nothing else.");
    messages[1].role = text_of("user");
    messages[1].content = text_of(document);
    stream_state st;
    memset(&st, 0, sizeof st);
    st.max_chars = max_chars;
    CHECK("turbo_generate", turbo_generate(model, &gd, messages, 2, on_chunk, &st, &err));
    const char *why = st.stopped ? "stopped by the caller"
                    : st.finish == TURBO_FINISH_EOS ? "EOS" : st.finish == TURBO_FINISH_LENGTH ? "LENGTH"
                    : st.finish == TURBO_FINISH_STOP ? "STOP" : st.finish == TURBO_FINISH_CANCELLED ? "CANCELLED" : "NONE";
    fprintf(stderr, "\n[%u tokens, prompt %u tokens, finish %s]\n", st.generated, st.prompt_tokens, why);
    if (max_chars != 0 && st.chars >= max_chars && !st.stopped) {
        fprintf(stderr, "the character budget was reached but the stream was not stopped\n");
        return 1;
    }
    if (!st.stopped && st.finish == TURBO_FINISH_NONE) {
        fprintf(stderr, "the stream ended without a finish reason\n");
        return 1;
    }
    free(document);
    turbo_model_release(model);
    turbo_context_release(ctx);
    turbo_runtime_release(rt);
    return 0;
}
