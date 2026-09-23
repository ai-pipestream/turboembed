/* SPDX-License-Identifier: Apache-2.0
 *
 * C smoke test against libturbo through the public header only.
 * Exercises: runtime, device selection, capability query, context, model
 * load from the committed mock bundle, session write/run/read, the result
 * lease, option rejection with field index, and generation.
 *
 * Build and run: scripts/c-smoke.sh
 */
#include "turbo/turbo.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr)                                                                          \
    do {                                                                                     \
        int32_t rc_ = (expr);                                                                \
        if (rc_ != TURBO_OK) {                                                               \
            fprintf(stderr, "%s:%d: %s -> %d %s: %s\n", __FILE__, __LINE__, #expr, rc_,     \
                    turbo_status_name(rc_), err.message);                                    \
            return 1;                                                                        \
        }                                                                                    \
    } while (0)

#define EXPECT(expr, code)                                                                   \
    do {                                                                                     \
        int32_t rc_ = (expr);                                                                \
        if (rc_ != (code)) {                                                                 \
            fprintf(stderr, "%s:%d: %s -> %d %s (expected %s): %s\n", __FILE__, __LINE__,   \
                    #expr, rc_, turbo_status_name(rc_), #code, err.message);                 \
            return 1;                                                                        \
        }                                                                                    \
    } while (0)

static turbo_text T(const char *s) {
    turbo_text t;
    t.ptr = s;
    t.len = (uint64_t)strlen(s);
    return t;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <path to testdata/bundles/mock>\n", argv[0]);
        return 2;
    }
    char embed_bundle[4096], gen_bundle[4096], tok_bundle[4096];
    snprintf(embed_bundle, sizeof embed_bundle, "%s/embedding", argv[1]);
    snprintf(gen_bundle, sizeof gen_bundle, "%s/generative", argv[1]);
    snprintf(tok_bundle, sizeof tok_bundle, "%s/../minilm-tokenizer", argv[1]);

    turbo_error err;
    memset(&err, 0, sizeof err);
    err.struct_size = (uint32_t)sizeof err;

    if (turbo_abi_version() != TURBO_ABI_VERSION) {
        fprintf(stderr, "header ABI %u but library ABI %u\n", TURBO_ABI_VERSION, turbo_abi_version());
        return 1;
    }

    turbo_runtime *rt = NULL;
    CHECK(turbo_runtime_create(NULL, &rt, &err));

    uint32_t n_devices = 0;
    CHECK(turbo_runtime_device_count(rt, &n_devices, &err));
    if (n_devices < 2) {
        fprintf(stderr, "expected the mock provider's two devices, got %u\n", n_devices);
        return 1;
    }

    /* AUTO must not select a CPU. */
    uint32_t dev = 0;
    CHECK(turbo_runtime_select_device(rt, NULL, &dev, &err));
    turbo_device_info di;
    memset(&di, 0, sizeof di);
    di.struct_size = (uint32_t)sizeof di;
    CHECK(turbo_runtime_device_info(rt, dev, &di, &err));
    if (di.kind == TURBO_DEVICE_CPU) {
        fprintf(stderr, "AUTO selected a CPU device\n");
        return 1;
    }
    printf("device [%u] %s:%u kind=%u name=\"%s\" caps=0x%llx\n", dev, di.provider_id, di.ordinal, di.kind,
           di.name, (unsigned long long)di.caps);

    /* Explicit CPU selection works. */
    turbo_device_selector sel;
    memset(&sel, 0, sizeof sel);
    sel.struct_size = (uint32_t)sizeof sel;
    sel.policy = TURBO_SELECT_EXPLICIT;
    sel.provider_id = T("mock");
    sel.ordinal = 0;
    uint32_t cpu = 0;
    CHECK(turbo_runtime_select_device(rt, &sel, &cpu, &err));

    /* Capability matrix. */
    turbo_capability cap;
    memset(&cap, 0, sizeof cap);
    cap.struct_size = (uint32_t)sizeof cap;
    CHECK(turbo_runtime_capability(rt, dev, TURBO_TASK_EMBED, TURBO_MODALITY_TEXT, &cap, &err));
    if (cap.status != TURBO_CAP_SUPPORTED) {
        fprintf(stderr, "mock embed/text should be SUPPORTED, got %u\n", cap.status);
        return 1;
    }
    CHECK(turbo_runtime_capability(rt, dev, TURBO_TASK_EMBED, TURBO_MODALITY_AUDIO, &cap, &err));
    if (cap.status != TURBO_CAP_UNSUPPORTED) {
        fprintf(stderr, "mock embed/audio should be UNSUPPORTED, got %u\n", cap.status);
        return 1;
    }
    EXPECT(turbo_runtime_capability(rt, dev, 99, TURBO_MODALITY_TEXT, &cap, &err), TURBO_E_INVALID_ENUM);

    CHECK(turbo_can_run(rt, dev, T(embed_bundle), TURBO_TASK_EMBED, TURBO_MODALITY_TEXT, &err));

    turbo_context *ctx = NULL;
    CHECK(turbo_context_create(rt, dev, NULL, &ctx, &err));
    turbo_runtime_release(rt); /* the context keeps the runtime alive */

    turbo_model *model = NULL;
    CHECK(turbo_model_load(ctx, T(embed_bundle), NULL, &model, &err));
    turbo_model_info mi;
    memset(&mi, 0, sizeof mi);
    mi.struct_size = (uint32_t)sizeof mi;
    CHECK(turbo_model_get_info(model, &mi, &err));
    printf("model %s dim=%u max_seq=%u max_batch=%u provider=%s fully_accelerated=%u\n", mi.model_id, mi.dim,
           mi.max_seq, mi.max_batch, mi.provider_id, mi.fully_accelerated);
    if (mi.dim != 8 || mi.kind != TURBO_MODEL_EMBEDDING) {
        fprintf(stderr, "unexpected model info\n");
        return 1;
    }

    turbo_session *session = NULL;
    CHECK(turbo_session_create(model, NULL, &session, &err));
    turbo_model_release(model); /* the session keeps the model alive */
    turbo_context_release(ctx);

    turbo_text texts[2];
    texts[0] = T("hello world");
    texts[1] = T("héllo wörld ünïcode");
    CHECK(turbo_session_write_text(session, texts, 2, NULL, &err));

    turbo_result *result = NULL;
    CHECK(turbo_session_run(session, NULL, &result, &err));

    /* The lease blocks a second run. */
    turbo_result *second = NULL;
    EXPECT(turbo_session_run(session, NULL, &second, &err), TURBO_E_BUSY);

    turbo_result_info ri;
    memset(&ri, 0, sizeof ri);
    ri.struct_size = (uint32_t)sizeof ri;
    CHECK(turbo_result_get_info(result, &ri, &err));
    if (ri.batch != 2 || ri.dim != 8 || ri.dtype != TURBO_DTYPE_F32 || ri.bytes != 2 * 8 * 4) {
        fprintf(stderr, "unexpected result info batch=%u dim=%u dtype=%u bytes=%llu\n", ri.batch, ri.dim, ri.dtype,
                (unsigned long long)ri.bytes);
        return 1;
    }
    float out[16];
    uint64_t written = 0;
    CHECK(turbo_result_read(result, 0, out, sizeof out, &written, &err));
    if (written != sizeof out) {
        fprintf(stderr, "read %llu bytes, expected %zu\n", (unsigned long long)written, sizeof out);
        return 1;
    }
    for (int r = 0; r < 2; ++r) {
        double norm = 0;
        for (int i = 0; i < 8; ++i) norm += (double)out[r * 8 + i] * out[r * 8 + i];
        if (fabs(sqrt(norm) - 1.0) > 1e-5) {
            fprintf(stderr, "row %d is not L2-normalized (norm %f)\n", r, sqrt(norm));
            return 1;
        }
    }
    /* Too small a destination is a capacity error, not a truncation. */
    EXPECT(turbo_result_read(result, 0, out, 4, &written, &err), TURBO_E_CAPACITY);
    turbo_result_release(result);

    /* An un-advertised option is rejected and names its field. */
    turbo_embed_options opts;
    memset(&opts, 0, sizeof opts);
    opts.struct_size = (uint32_t)sizeof opts;
    opts.pooling = TURBO_POOLING_CLS;
    EXPECT(turbo_session_write_text(session, texts, 1, &opts, &err), TURBO_E_UNSUPPORTED_OPTION);
    if (err.field != 6) {
        fprintf(stderr, "expected field 6 (pooling), got %u\n", err.field);
        return 1;
    }
    /* An unknown enum value is rejected. */
    opts.pooling = 0;
    opts.truncate = 77;
    EXPECT(turbo_session_write_text(session, texts, 1, &opts, &err), TURBO_E_INVALID_ENUM);
    /* A struct_size larger than the library knows is rejected. */
    opts.truncate = 0;
    opts.struct_size = 4096;
    EXPECT(turbo_session_write_text(session, texts, 1, &opts, &err), TURBO_E_INVALID_STRUCT_SIZE);
    /* Invalid UTF-8 is rejected. */
    turbo_text bad;
    bad.ptr = "\xff\xfe";
    bad.len = 2;
    EXPECT(turbo_session_write_text(session, &bad, 1, NULL, &err), TURBO_E_INVALID_UTF8);
    /* NULL handle. */
    EXPECT(turbo_session_write_text(NULL, texts, 1, NULL, &err), TURBO_E_INVALID_HANDLE);

    turbo_session_stats st;
    memset(&st, 0, sizeof st);
    st.struct_size = (uint32_t)sizeof st;
    CHECK(turbo_session_get_stats(session, &st, &err));
    printf("runs=%llu provider_allocs=%llu\n", (unsigned long long)st.runs, (unsigned long long)st.provider_allocs);
    turbo_session_release(session);

    /* Generation through the pull iterator. */
    turbo_runtime *rt2 = NULL;
    CHECK(turbo_runtime_create(NULL, &rt2, &err));
    turbo_context *gctx = NULL;
    CHECK(turbo_context_create(rt2, dev, NULL, &gctx, &err));
    turbo_model *gmodel = NULL;
    CHECK(turbo_model_load(gctx, T(gen_bundle), NULL, &gmodel, &err));
    turbo_generate_desc gd;
    memset(&gd, 0, sizeof gd);
    gd.struct_size = (uint32_t)sizeof gd;
    gd.max_new_tokens = 5;
    turbo_generation *gen = NULL;
    CHECK(turbo_generation_create(gmodel, &gd, &gen, &err));
    turbo_message msg;
    msg.role = T("user");
    msg.content = T("say something");
    CHECK(turbo_generation_prompt(gen, &msg, 1, &err));
    turbo_generation_chunk chunk;
    memset(&chunk, 0, sizeof chunk);
    chunk.struct_size = (uint32_t)sizeof chunk;
    uint32_t steps = 0;
    do {
        CHECK(turbo_generation_step(gen, &chunk, &err));
        printf("step %u: %.*s(done=%u reason=%u)\n", steps, (int)chunk.text.len, chunk.text.ptr, chunk.done,
               chunk.finish_reason);
        steps++;
    } while (!chunk.done && steps < 100);
    if (chunk.finish_reason != TURBO_FINISH_LENGTH || chunk.generated_tokens != 5) {
        fprintf(stderr, "expected LENGTH after 5 tokens, got reason=%u generated=%u\n", chunk.finish_reason,
                chunk.generated_tokens);
        return 1;
    }
    EXPECT(turbo_generation_step(gen, &chunk, &err), TURBO_E_INVALID_STATE);
    turbo_generation_release(gen);
    turbo_model_release(gmodel);
    turbo_context_release(gctx);
    turbo_runtime_release(rt2);

    /* Push generation is declared but not implemented in this build. */
    EXPECT(turbo_generate(NULL, NULL, NULL, 0, NULL, NULL, &err), TURBO_E_INVALID_HANDLE);

    /* Tokenizer: reference ids for MiniLM, write-through padding, decode, count. */
    {
        turbo_runtime *rt3 = NULL;
        CHECK(turbo_runtime_create(NULL, &rt3, &err));
        turbo_tokenizer *tok = NULL;
        CHECK(turbo_tokenizer_create(rt3, T(tok_bundle), &tok, &err));
        turbo_tokenizer_info ti;
        memset(&ti, 0, sizeof ti);
        ti.struct_size = (uint32_t)sizeof ti;
        CHECK(turbo_tokenizer_get_info(tok, &ti, &err));
        if (ti.vocab_size != 30522 || ti.bos_id != 101 || ti.eos_id != 102 || ti.pad_id != 0) {
            fprintf(stderr, "unexpected tokenizer info vocab=%u bos=%d eos=%d pad=%d\n", ti.vocab_size, ti.bos_id,
                    ti.eos_id, ti.pad_id);
            return 1;
        }
        turbo_text two[2];
        two[0] = T("hello world");
        two[1] = T("hi");
        int32_t ids[16], mask[16];
        uint32_t lengths[2];
        turbo_encode_options eo;
        memset(&eo, 0, sizeof eo);
        eo.struct_size = (uint32_t)sizeof eo;
        eo.add_special_tokens = 1;
        eo.max_tokens = 8;
        CHECK(turbo_tokenizer_encode(tok, two, 2, &eo, ids, mask, NULL, 8, lengths, &err));
        if (lengths[0] != 4 || ids[0] != 101 || ids[1] != 7592 || ids[2] != 2088 || ids[3] != 102 || mask[4] != 0 ||
            lengths[1] != 3 || ids[8] != 101) {
            fprintf(stderr, "unexpected MiniLM token ids\n");
            return 1;
        }
        char text_out[64];
        uint64_t n_out = 0;
        int32_t two_ids[2] = {7592, 2088};
        CHECK(turbo_tokenizer_decode(tok, two_ids, 2, 1, text_out, sizeof text_out, &n_out, &err));
        if (n_out != 11 || memcmp(text_out, "hello world", 11) != 0) {
            fprintf(stderr, "unexpected decode\n");
            return 1;
        }
        EXPECT(turbo_tokenizer_decode(tok, two_ids, 2, 1, text_out, 3, &n_out, &err), TURBO_E_CAPACITY);
        uint32_t n_tok = 0;
        CHECK(turbo_tokenizer_count(tok, T("hello world"), 1, &n_tok, &err));
        if (n_tok != 4) {
            fprintf(stderr, "count %u != 4\n", n_tok);
            return 1;
        }
        /* Chunk plan over the same tokenizer. */
        turbo_chunk_desc cd;
        memset(&cd, 0, sizeof cd);
        cd.struct_size = (uint32_t)sizeof cd;
        cd.max_tokens = 8;
        cd.reserved_tokens = 2;
        const char *doc = "one two three four five six seven eight nine ten eleven twelve.\n\nsecond paragraph here";
        turbo_chunk_plan *plan = NULL;
        CHECK(turbo_chunk_plan_create(&cd, T(doc), tok, &plan, &err));
        uint32_t n_chunks = 0;
        CHECK(turbo_chunk_plan_count(plan, &n_chunks, &err));
        if (n_chunks < 3) {
            fprintf(stderr, "expected at least 3 chunks, got %u\n", n_chunks);
            return 1;
        }
        uint64_t prev_end = 0;
        for (uint32_t i = 0; i < n_chunks; ++i) {
            turbo_chunk c;
            CHECK(turbo_chunk_plan_get(plan, i, &c, &err));
            if (c.byte_start < prev_end || c.byte_end <= c.byte_start || c.byte_end > strlen(doc) || c.n_tokens > 6) {
                fprintf(stderr, "bad chunk %u: [%llu, %llu) tokens=%u\n", i, (unsigned long long)c.byte_start,
                        (unsigned long long)c.byte_end, c.n_tokens);
                return 1;
            }
            prev_end = c.byte_end;
        }
        turbo_chunk c;
        EXPECT(turbo_chunk_plan_get(plan, n_chunks, &c, &err), TURBO_E_INVALID_ARGUMENT);
        turbo_chunk_plan_release(plan);
        turbo_tokenizer_release(tok);
        turbo_runtime_release(rt3);
    }

    puts("c-smoke: OK");
    return 0;
}
