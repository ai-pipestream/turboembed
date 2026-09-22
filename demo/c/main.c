/* SPDX-License-Identifier: Apache-2.0
 *
 * Turbo C demo: load a bundle on the best device, embed a few sentences,
 * and print their cosine similarities. Every call is checked; a failure
 * prints the status name and the library's message and exits non-zero.
 *
 *   turbo-demo-c [--provider-lib <so>] [--provider <id> --ordinal <n>] --bundle <dir> text...
 */
#include "turbo/turbo.h"

#include <math.h>
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

int main(int argc, char **argv) {
    const char *provider_lib = NULL, *provider = NULL, *bundle = NULL;
    uint32_t ordinal = 0;
    int have_ordinal = 0;
    const char *texts[64];
    uint32_t n_texts = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--provider-lib") == 0 && i + 1 < argc) {
            provider_lib = argv[++i];
        } else if (strcmp(argv[i], "--provider") == 0 && i + 1 < argc) {
            provider = argv[++i];
        } else if (strcmp(argv[i], "--ordinal") == 0 && i + 1 < argc) {
            ordinal = (uint32_t)strtoul(argv[++i], NULL, 10);
            have_ordinal = 1;
        } else if (strcmp(argv[i], "--bundle") == 0 && i + 1 < argc) {
            bundle = argv[++i];
        } else if (n_texts < 64) {
            texts[n_texts++] = argv[i];
        }
    }
    if (bundle == NULL || n_texts == 0) {
        fprintf(stderr, "usage: %s [--provider-lib <so>] [--provider <id> --ordinal <n>] --bundle <dir> text...\n", argv[0]);
        return 2;
    }
    if ((provider == NULL) != (have_ordinal == 0)) {
        fprintf(stderr, "--provider and --ordinal go together (explicit selection); omit both for AUTO\n");
        return 2;
    }

    turbo_error err;
    memset(&err, 0, sizeof err);
    err.struct_size = (uint32_t)sizeof err;

    /* Runtime, with an extra provider library when one is named. */
    turbo_runtime_desc rd;
    memset(&rd, 0, sizeof rd);
    rd.struct_size = (uint32_t)sizeof rd;
    turbo_text lib_text;
    if (provider_lib != NULL) {
        lib_text = text_of(provider_lib);
        rd.n_provider_paths = 1;
        rd.provider_paths = &lib_text;
    }
    turbo_runtime *rt = NULL;
    CHECK("turbo_runtime_create", turbo_runtime_create(&rd, &rt, &err));

    /* Device: AUTO never picks a CPU; an explicit provider/ordinal may. */
    uint32_t dev = 0;
    if (provider != NULL) {
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
    printf("device: %s (%s:%u, kind %u, runtime %s)\n", di.name, di.provider_id, di.ordinal, di.kind, di.runtime_version);

    turbo_context *ctx = NULL;
    CHECK("turbo_context_create", turbo_context_create(rt, dev, NULL, &ctx, &err));
    turbo_model *model = NULL;
    CHECK("turbo_model_load", turbo_model_load(ctx, text_of(bundle), NULL, &model, &err));
    turbo_model_info mi;
    memset(&mi, 0, sizeof mi);
    mi.struct_size = (uint32_t)sizeof mi;
    CHECK("turbo_model_get_info", turbo_model_get_info(model, &mi, &err));
    if (mi.kind != TURBO_MODEL_EMBEDDING) {
        fprintf(stderr, "bundle %s is not an embedding model (kind %u)\n", bundle, mi.kind);
        return 1;
    }
    printf("model: %s dim=%u max_seq=%u provider=%s fully_accelerated=%u\n", mi.model_id, mi.dim, mi.max_seq, mi.provider_id, mi.fully_accelerated);

    turbo_session_desc sd;
    memset(&sd, 0, sizeof sd);
    sd.struct_size = (uint32_t)sizeof sd;
    sd.max_batch = n_texts;
    sd.max_seq = mi.max_seq < 128 ? mi.max_seq : 128;
    turbo_session *session = NULL;
    CHECK("turbo_session_create", turbo_session_create(model, &sd, &session, &err));

    turbo_text views[64];
    for (uint32_t i = 0; i < n_texts; ++i) {
        views[i] = text_of(texts[i]);
    }
    CHECK("turbo_session_write_text", turbo_session_write_text(session, views, n_texts, NULL, &err));
    turbo_result *result = NULL;
    CHECK("turbo_session_run", turbo_session_run(session, NULL, &result, &err));
    turbo_result_info ri;
    memset(&ri, 0, sizeof ri);
    ri.struct_size = (uint32_t)sizeof ri;
    CHECK("turbo_result_get_info", turbo_result_get_info(result, &ri, &err));
    if (ri.dtype != TURBO_DTYPE_F32 || ri.batch != n_texts) {
        fprintf(stderr, "unexpected result: batch %u dtype %u\n", ri.batch, ri.dtype);
        return 1;
    }
    float *out = (float *)malloc((size_t)ri.bytes);
    if (out == NULL) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    uint64_t written = 0;
    CHECK("turbo_result_read", turbo_result_read(result, 0, out, ri.bytes, &written, &err));
    const uint32_t dim = ri.dim;
    printf("embeddings: %u x %u (placement %u)\n", ri.batch, dim, ri.placement);
    printf("cosine similarity:\n");
    for (uint32_t a = 0; a < n_texts; ++a) {
        for (uint32_t b = 0; b < n_texts; ++b) {
            double dot = 0, na = 0, nb = 0;
            for (uint32_t k = 0; k < dim; ++k) {
                dot += (double)out[a * dim + k] * out[b * dim + k];
                na += (double)out[a * dim + k] * out[a * dim + k];
                nb += (double)out[b * dim + k] * out[b * dim + k];
            }
            printf(" %6.3f", dot / (sqrt(na) * sqrt(nb)));
        }
        printf("  %s\n", texts[a]);
    }
    free(out);
    turbo_result_release(result);
    turbo_session_release(session);
    turbo_model_release(model);
    turbo_context_release(ctx);
    turbo_runtime_release(rt);
    return 0;
}
