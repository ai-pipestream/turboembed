// SPDX-License-Identifier: Apache-2.0
//
// TurboEmbed C ABI edge suite against the plain stub dispatcher
// (no TURBOEMBED_ORT_CUDA / TURBOEMBED_GENAI):
//   lifecycle, fail-loud device policy, error paths, mock-embed
//   determinism, opts=NULL vs defaults, stream callbacks, list_models,
//   and per-engine last_error isolation.
// Mirrors the flags of `make turboembed-mock-arena-tests`. Mock never
// impersonates a catalog model; catalog aliases fail loud (NOT_IMPLEMENTED).

#include "turboembed.h"

#include <cstdio>
#include <cstring>

static int g_fails = 0;
static int g_passes = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_fails;                                                           \
        } else {                                                                 \
            ++g_passes;                                                          \
        }                                                                        \
    } while (0)

#define CHECK_ST(st) CHECK((st) == TURBOEMBED_OK)

static bool streq(const char *a, const char *b) {
    return a != nullptr && b != nullptr && std::strcmp(a, b) == 0;
}

static bool result_bytes_eq(
    const turboembed_embed_result *a,
    const turboembed_embed_result *b
) {
    if (a == nullptr || b == nullptr) {
        return false;
    }
    if (a->dim != b->dim || a->count != b->count || a->packed_len != b->packed_len) {
        return false;
    }
    const size_t bytes =
        static_cast<size_t>(a->count) * static_cast<size_t>(a->dim) * sizeof(float);
    if (bytes > 0 &&
        (a->values == nullptr || b->values == nullptr ||
         std::memcmp(a->values, b->values, bytes) != 0)) {
        return false;
    }
    if (a->packed_len > 0 &&
        (a->packed == nullptr || b->packed == nullptr ||
         std::memcmp(a->packed, b->packed, a->packed_len) != 0)) {
        return false;
    }
    return true;
}

static turboembed_embed_result *embed_texts(
    turboembed_engine *e,
    const char *alias,
    const turboembed_str *texts,
    size_t n,
    const turboembed_embed_options *opts
) {
    turboembed_embed_result *out = nullptr;
    if (turboembed_embed(e, alias, 0, texts, n, opts, &out) != TURBOEMBED_OK) {
        return nullptr;
    }
    return out;
}

static void test_lifecycle() {
    CHECK(turboembed_abi_version() == TURBOEMBED_ABI_VERSION);

    turboembed_engine *e = nullptr;
    CHECK_ST(turboembed_engine_create(TURBOEMBED_DEVICE_MOCK, nullptr, &e));
    CHECK(e != nullptr);
    // No error yet: empty but never NULL.
    CHECK(turboembed_last_error(e) != nullptr);
    CHECK(turboembed_last_error(e)[0] == '\0');
    // Successful create clears the thread-local create error.
    CHECK(turboembed_last_error(nullptr) != nullptr);
    CHECK(turboembed_last_error(nullptr)[0] == '\0');
    turboembed_engine_destroy(e);

    // Stub also accepts the explicit CPU devices.
    e = nullptr;
    CHECK_ST(turboembed_engine_create(TURBOEMBED_DEVICE_CPU, nullptr, &e));
    CHECK(e != nullptr);
    turboembed_engine_destroy(e);
    e = nullptr;
    CHECK_ST(turboembed_engine_create(TURBOEMBED_DEVICE_OPENVINO_CPU, nullptr, &e));
    CHECK(e != nullptr);
    turboembed_engine_destroy(e);

    // config_path is provider-only on the stub; mock create ignores it.
    e = nullptr;
    CHECK_ST(turboembed_engine_create(
        TURBOEMBED_DEVICE_MOCK, "definitely/missing/config.toml", &e
    ));
    CHECK(e != nullptr);
    turboembed_engine_destroy(e);

    // NULL out pointer is rejected and explained via last_error(NULL).
    CHECK(turboembed_engine_create(TURBOEMBED_DEVICE_MOCK, nullptr, nullptr) ==
          TURBOEMBED_ERR_INVALID_ARGUMENT);
    CHECK(turboembed_last_error(nullptr) != nullptr);
    CHECK(std::strstr(turboembed_last_error(nullptr), "out pointer is null") != nullptr);

    // destroy(NULL) is a documented no-op.
    turboembed_engine_destroy(nullptr);
}

static void test_fail_loud_devices() {
    // Plain stub has no GPU: accelerator/AUTO requests fail loud, never CPU.
    turboembed_engine *e = reinterpret_cast<turboembed_engine *>(0x1);
    CHECK(turboembed_engine_create(TURBOEMBED_DEVICE_CUDA, nullptr, &e) ==
          TURBOEMBED_ERR_UNAVAILABLE);
    CHECK(e == nullptr);
    CHECK(turboembed_last_error(nullptr) != nullptr);
    CHECK(std::strstr(turboembed_last_error(nullptr), "cuda") != nullptr);

    CHECK(turboembed_engine_create(TURBOEMBED_DEVICE_AUTO, nullptr, &e) ==
          TURBOEMBED_ERR_UNAVAILABLE);
    CHECK(e == nullptr);
    CHECK(std::strstr(turboembed_last_error(nullptr), "auto") != nullptr);

    CHECK(turboembed_engine_create(TURBOEMBED_DEVICE_METAL, nullptr, &e) ==
          TURBOEMBED_ERR_UNAVAILABLE);
    CHECK(e == nullptr);

    // Unknown enum value: UNSUPPORTED_DEVICE, not UNAVAILABLE.
    e = reinterpret_cast<turboembed_engine *>(0x1);
    CHECK(turboembed_engine_create(
              static_cast<turboembed_device>(42), nullptr, &e
          ) == TURBOEMBED_ERR_UNSUPPORTED_DEVICE);
    CHECK(e == nullptr);
    CHECK(std::strstr(turboembed_last_error(nullptr), "unknown device") != nullptr);
}

static void test_names() {
    static const char *kStatus[] = {
        "OK",
        "INVALID_ARGUMENT",
        "NOT_FOUND",
        "NOT_IMPLEMENTED",
        "UNAVAILABLE",
        "INTERNAL",
        "OUT_OF_MEMORY",
        "UNSUPPORTED_DEVICE",
    };
    for (int i = 0; i <= 7; ++i) {
        const char *name = turboembed_status_name(static_cast<turboembed_status>(i));
        CHECK(name != nullptr);
        CHECK(streq(name, kStatus[i]));
    }
    CHECK(streq(
        turboembed_status_name(static_cast<turboembed_status>(8)), "UNKNOWN"
    ));

    static const char *kDevice[] = {
        "auto",
        "cpu",
        "cuda",
        "tensorrt",
        "openvino-cpu",
        "openvino-gpu",
        "openvino-npu",
        "metal",
        "mock",
    };
    for (int i = 0; i <= 8; ++i) {
        const char *name = turboembed_device_name(static_cast<turboembed_device>(i));
        CHECK(name != nullptr);
        CHECK(streq(name, kDevice[i]));
    }
    CHECK(streq(
        turboembed_device_name(static_cast<turboembed_device>(42)), "unknown"
    ));
}

static void test_load_model_errors() {
    turboembed_engine *e = nullptr;
    CHECK_ST(turboembed_engine_create(TURBOEMBED_DEVICE_MOCK, nullptr, &e));

    // Unknown catalog alias: the stub's actual status is NOT_IMPLEMENTED,
    // and the mock never substitutes for a real model.
    CHECK(turboembed_load_model(e, "minilm", 0) == TURBOEMBED_ERR_NOT_IMPLEMENTED);
    CHECK(turboembed_last_error(e) != nullptr);
    CHECK(turboembed_last_error(e)[0] != '\0');

    // NULL / empty alias: INVALID_ARGUMENT.
    CHECK(turboembed_load_model(e, nullptr, 0) == TURBOEMBED_ERR_INVALID_ARGUMENT);
    CHECK(turboembed_load_model(e, "", 0) == TURBOEMBED_ERR_INVALID_ARGUMENT);

    // 'mock-embed' loads twice: idempotent, error cleared.
    CHECK_ST(turboembed_load_model(e, "mock-embed", 0));
    CHECK_ST(turboembed_load_model(e, "mock-embed", 0));
    CHECK(turboembed_last_error(e)[0] == '\0');
    // Short alias and explicit alias length both accepted.
    CHECK_ST(turboembed_load_model(e, "mock", 0));
    CHECK_ST(turboembed_load_model(e, "mock-embed", 10));

    turboembed_engine_destroy(e);
}

static void test_list_models() {
    turboembed_engine *e = nullptr;
    CHECK_ST(turboembed_engine_create(TURBOEMBED_DEVICE_MOCK, nullptr, &e));

    turboembed_model_info *infos = nullptr;
    size_t count = 0;
    CHECK_ST(turboembed_list_models(e, &infos, &count));
    CHECK(infos != nullptr);
    CHECK(count == 1);
    CHECK(infos[0].alias.ptr != nullptr);
    CHECK(infos[0].alias.len == 10);
    CHECK(std::memcmp(infos[0].alias.ptr, "mock-embed", 10) == 0);
    CHECK(infos[0].dim == 8);
    CHECK(infos[0].device == TURBOEMBED_DEVICE_MOCK);
    // Mock device preloads the mock alias at create.
    CHECK(infos[0].ready == 1);
    turboembed_model_list_free(infos, count);
    turboembed_model_list_free(nullptr, 0); // no-op

    // Null outputs are rejected.
    CHECK(turboembed_list_models(e, nullptr, &count) ==
          TURBOEMBED_ERR_INVALID_ARGUMENT);
    CHECK(turboembed_list_models(nullptr, &infos, &count) ==
          TURBOEMBED_ERR_INVALID_ARGUMENT);
    turboembed_engine_destroy(e);

    // Explicit CPU engine: mock alias starts unloaded, ready flips on load.
    e = nullptr;
    CHECK_ST(turboembed_engine_create(TURBOEMBED_DEVICE_CPU, nullptr, &e));
    infos = nullptr;
    count = 0;
    CHECK_ST(turboembed_list_models(e, &infos, &count));
    CHECK(count == 1);
    CHECK(infos[0].ready == 0);
    turboembed_model_list_free(infos, count);

    CHECK_ST(turboembed_load_model(e, "mock-embed", 0));
    CHECK_ST(turboembed_list_models(e, &infos, &count));
    CHECK(count == 1);
    CHECK(infos[0].ready == 1);
    turboembed_model_list_free(infos, count);

    // Mock alias also serves on explicit CPU.
    const char t[] = "cpu engine smoke";
    turboembed_str view = {t, sizeof(t) - 1};
    turboembed_embed_result *r = embed_texts(e, "mock-embed", &view, 1, nullptr);
    CHECK(r != nullptr);
    CHECK(r->dim == 8);
    CHECK(r->count == 1);
    turboembed_embed_result_free(r);
    turboembed_engine_destroy(e);
}

static void test_embed_edges() {
    turboembed_engine *e = nullptr;
    CHECK_ST(turboembed_engine_create(TURBOEMBED_DEVICE_MOCK, nullptr, &e));
    CHECK_ST(turboembed_load_model(e, "mock-embed", 0));

    const char a[] = "alpha";
    const char b[] = "beta gamma";
    const char c[] = "h\xc3\xa9llo w\xc3\xb6rld"; // UTF-8, non-ASCII
    turboembed_str batch[3] = {
        {a, sizeof(a) - 1},
        {b, sizeof(b) - 1},
        {c, sizeof(c) - 1},
    };

    // n_texts == 0: INVALID_ARGUMENT and *out is nullified.
    turboembed_embed_result *out = reinterpret_cast<turboembed_embed_result *>(0x1);
    CHECK(turboembed_embed(e, "mock-embed", 0, batch, 0, nullptr, &out) ==
          TURBOEMBED_ERR_INVALID_ARGUMENT);
    CHECK(out == nullptr);

    // Null out / null texts / null ptr with non-zero len: INVALID_ARGUMENT.
    CHECK(turboembed_embed(e, "mock-embed", 0, batch, 3, nullptr, nullptr) ==
          TURBOEMBED_ERR_INVALID_ARGUMENT);
    CHECK(turboembed_embed(e, "mock-embed", 0, nullptr, 1, nullptr, &out) ==
          TURBOEMBED_ERR_INVALID_ARGUMENT);
    turboembed_str bad_view = {nullptr, 3};
    CHECK(turboembed_embed(e, "mock-embed", 0, &bad_view, 1, nullptr, &out) ==
          TURBOEMBED_ERR_INVALID_ARGUMENT);

    // Unknown alias on embed: NOT_IMPLEMENTED, never the 8-d mock row.
    out = nullptr;
    CHECK(turboembed_embed(e, "minilm", 0, batch, 3, nullptr, &out) ==
          TURBOEMBED_ERR_NOT_IMPLEMENTED);
    CHECK(out == nullptr);

    // Non-default opts on the mock path: NOT_IMPLEMENTED.
    turboembed_embed_options mean_opts = {
        TURBOEMBED_POOLING_MEAN, -1, 0, TURBOEMBED_OUTPUT_TYPED
    };
    CHECK(turboembed_embed(e, "mock-embed", 0, batch, 3, &mean_opts, &out) ==
          TURBOEMBED_ERR_NOT_IMPLEMENTED);

    // opts == NULL vs explicit provider defaults: identical bytes.
    turboembed_embed_result *r_null = embed_texts(e, "mock-embed", batch, 3, nullptr);
    CHECK(r_null != nullptr);
    CHECK(r_null->dim == 8);
    CHECK(r_null->count == 3);
    const turboembed_embed_options default_opts = {
        TURBOEMBED_POOLING_DEFAULT, -1, 0, TURBOEMBED_OUTPUT_TYPED
    };
    turboembed_embed_result *r_def =
        embed_texts(e, "mock-embed", batch, 3, &default_opts);
    CHECK(r_def != nullptr);
    CHECK(result_bytes_eq(r_null, r_def));

    // Explicit alias length and the short alias agree with alias_len == 0.
    turboembed_embed_result *r_len = nullptr;
    CHECK_ST(turboembed_embed(e, "mock-embed", 10, batch, 3, nullptr, &r_len));
    CHECK(result_bytes_eq(r_null, r_len));
    turboembed_embed_result *r_short = embed_texts(e, "mock", batch, 3, nullptr);
    CHECK(result_bytes_eq(r_null, r_short));

    // Typed and packed views describe the same bytes.
    CHECK(r_null->packed != nullptr);
    CHECK(r_null->packed_len ==
          static_cast<size_t>(r_null->count) * r_null->dim * sizeof(float));
    CHECK(std::memcmp(
              r_null->packed,
              r_null->values,
              r_null->packed_len
          ) == 0);

    // Empty string (len 0) embeds fine; NULL ptr with len 0 is the same.
    turboembed_str empty_a = {nullptr, 0};
    turboembed_str empty_b = {"", 0};
    turboembed_embed_result *r_ea = embed_texts(e, "mock-embed", &empty_a, 1, nullptr);
    turboembed_embed_result *r_eb = embed_texts(e, "mock-embed", &empty_b, 1, nullptr);
    CHECK(r_ea != nullptr);
    CHECK(r_eb != nullptr);
    CHECK(r_ea->count == 1 && r_ea->dim == 8);
    CHECK(result_bytes_eq(r_ea, r_eb));

    // Embedded NUL is honored: {"a\0b",3} differs from {"a",1}.
    const char with_nul[3] = {'a', '\0', 'b'};
    turboembed_str nul_view = {with_nul, 3};
    turboembed_embed_result *r_nul = embed_texts(e, "mock-embed", &nul_view, 1, nullptr);
    const char just_a[] = "a";
    turboembed_str a_view = {just_a, 1};
    turboembed_embed_result *r_a = embed_texts(e, "mock-embed", &a_view, 1, nullptr);
    CHECK(r_nul != nullptr);
    CHECK(r_a != nullptr);
    CHECK(!result_bytes_eq(r_nul, r_a));

    // embed_one equals row 0 of the batch embed.
    turboembed_embed_result *r_one = nullptr;
    CHECK_ST(turboembed_embed_one(e, "mock-embed", 0, a, sizeof(a) - 1, nullptr, &r_one));
    CHECK(r_one != nullptr);
    CHECK(r_one->count == 1);
    CHECK(r_one->dim == r_null->dim);
    CHECK(std::memcmp(r_one->values, r_null->values, r_one->dim * sizeof(float)) == 0);
    // Row 1 of the batch equals embed_one("beta gamma").
    turboembed_embed_result *r_one_b = nullptr;
    CHECK_ST(turboembed_embed_one(e, "mock-embed", 0, b, sizeof(b) - 1, nullptr, &r_one_b));
    CHECK(r_one_b != nullptr);
    CHECK(std::memcmp(
              r_one_b->values,
              r_null->values + static_cast<size_t>(r_null->dim),
              r_null->dim * sizeof(float)
          ) == 0);

    turboembed_embed_result_free(nullptr); // no-op
    turboembed_buffer_free(nullptr);       // no-op
    turboembed_embed_result_free(r_null);
    turboembed_embed_result_free(r_def);
    turboembed_embed_result_free(r_len);
    turboembed_embed_result_free(r_short);
    turboembed_embed_result_free(r_ea);
    turboembed_embed_result_free(r_eb);
    turboembed_embed_result_free(r_nul);
    turboembed_embed_result_free(r_a);
    turboembed_embed_result_free(r_one);
    turboembed_embed_result_free(r_one_b);

    // Embed before load on explicit CPU: NOT_FOUND, explained.
    turboembed_engine *cpu = nullptr;
    CHECK_ST(turboembed_engine_create(TURBOEMBED_DEVICE_CPU, nullptr, &cpu));
    out = nullptr;
    CHECK(turboembed_embed(cpu, "mock-embed", 0, batch, 3, nullptr, &out) ==
          TURBOEMBED_ERR_NOT_FOUND);
    CHECK(out == nullptr);
    CHECK(std::strstr(turboembed_last_error(cpu), "not loaded") != nullptr);
    // ...and a failed embed does not poison the mock engine under test.
    CHECK_ST(turboembed_embed(e, "mock-embed", 0, batch, 3, nullptr, &out));
    turboembed_embed_result_free(out);

    turboembed_engine_destroy(cpu);
    turboembed_engine_destroy(e);
}

struct StreamCapture {
    uint32_t indices[8];
    uint32_t dims[8];
    int32_t finals[8];
    float rows[8][8];
    uint32_t calls;
    void *seen_user_data;
};

static void capture_cb(
    void *user_data,
    uint32_t index,
    const float *values,
    uint32_t dim,
    int32_t is_final
) {
    auto *cap = static_cast<StreamCapture *>(user_data);
    cap->seen_user_data = user_data;
    if (cap->calls < 8) {
        const uint32_t slot = cap->calls;
        cap->indices[slot] = index;
        cap->dims[slot] = dim;
        cap->finals[slot] = is_final;
        const uint32_t n = dim < 8 ? dim : 8;
        for (uint32_t i = 0; i < n; ++i) {
            cap->rows[slot][i] = values[i];
        }
    }
    ++cap->calls;
}

static void test_stream() {
    turboembed_engine *e = nullptr;
    CHECK_ST(turboembed_engine_create(TURBOEMBED_DEVICE_MOCK, nullptr, &e));

    const char a[] = "stream one";
    const char b[] = "stream two";
    const char c[] = "stream three";
    turboembed_str texts[3] = {
        {a, sizeof(a) - 1},
        {b, sizeof(b) - 1},
        {c, sizeof(c) - 1},
    };

    turboembed_embed_result *r_batch = embed_texts(e, "mock-embed", texts, 3, nullptr);
    CHECK(r_batch != nullptr);

    // NULL callback: identical to batch embed, out still populated.
    turboembed_embed_result *r_stream = nullptr;
    CHECK_ST(turboembed_embed_stream(
        e, "mock-embed", 0, texts, 3, nullptr, nullptr, nullptr, &r_stream
    ));
    CHECK(r_stream != nullptr);
    CHECK(result_bytes_eq(r_batch, r_stream));

    // Callback: rows arrive 0..n-1, final flag only on the last row,
    // per-row values match the batch result.
    StreamCapture cap = {};
    turboembed_embed_result *r_stream2 = nullptr;
    CHECK_ST(turboembed_embed_stream(
        e, "mock-embed", 0, texts, 3, nullptr, capture_cb, &cap, &r_stream2
    ));
    CHECK(cap.seen_user_data == &cap);
    CHECK(cap.calls == 3);
    CHECK(cap.indices[0] == 0 && cap.indices[1] == 1 && cap.indices[2] == 2);
    CHECK(cap.finals[0] == 0 && cap.finals[1] == 0 && cap.finals[2] == 1);
    for (uint32_t i = 0; i < 3; ++i) {
        CHECK(cap.dims[i] == r_batch->dim);
        CHECK(std::memcmp(
                  cap.rows[i],
                  r_batch->values + static_cast<size_t>(i) * r_batch->dim,
                  r_batch->dim * sizeof(float)
              ) == 0);
    }
    CHECK(result_bytes_eq(r_batch, r_stream2));

    // out == NULL with a callback: result freed internally, no crash.
    cap = StreamCapture{};
    CHECK_ST(turboembed_embed_stream(
        e, "mock-embed", 0, texts, 3, nullptr, capture_cb, &cap, nullptr
    ));
    CHECK(cap.calls == 3);

    // Error path: callback never fires, error status propagates.
    // NOTE: unlike turboembed_embed (which nullifies *out up front),
    // embed_stream returns before touching *out, so the caller's out
    // pointer is left as-is on error. Suspected product inconsistency
    // (stub.cpp embed_stream error path); asserted as current behavior.
    cap = StreamCapture{};
    turboembed_embed_result *r_err = reinterpret_cast<turboembed_embed_result *>(0x1);
    CHECK(turboembed_embed_stream(
              e, "mock-embed", 0, texts, 0, nullptr, capture_cb, &cap, &r_err
          ) == TURBOEMBED_ERR_INVALID_ARGUMENT);
    CHECK(cap.calls == 0);
    CHECK(r_err == reinterpret_cast<turboembed_embed_result *>(0x1));

    turboembed_embed_result_free(r_batch);
    turboembed_embed_result_free(r_stream);
    turboembed_embed_result_free(r_stream2);
    turboembed_engine_destroy(e);
}

static void test_error_isolation() {
    turboembed_engine *a = nullptr;
    turboembed_engine *b = nullptr;
    CHECK_ST(turboembed_engine_create(TURBOEMBED_DEVICE_MOCK, nullptr, &a));
    CHECK_ST(turboembed_engine_create(TURBOEMBED_DEVICE_MOCK, nullptr, &b));

    const char t[] = "isolation probe";
    turboembed_str view = {t, sizeof(t) - 1};

    // Failure on A records A's error only.
    turboembed_embed_result *out = nullptr;
    CHECK(turboembed_embed(a, "mock-embed", 0, &view, 0, nullptr, &out) ==
          TURBOEMBED_ERR_INVALID_ARGUMENT);
    CHECK(std::strstr(turboembed_last_error(a), "texts must not be empty") != nullptr);
    CHECK(turboembed_last_error(b) != nullptr);
    CHECK(turboembed_last_error(b)[0] == '\0');

    // Success on B clears B's error, leaves A's untouched.
    turboembed_embed_result *rb = embed_texts(b, "mock-embed", &view, 1, nullptr);
    CHECK(rb != nullptr);
    CHECK(turboembed_last_error(b)[0] == '\0');
    CHECK(std::strstr(turboembed_last_error(a), "texts must not be empty") != nullptr);

    // A different failure on B still leaves A's message intact.
    CHECK(turboembed_embed(b, "mock-embed", 0, &view, 0, nullptr, &out) ==
          TURBOEMBED_ERR_INVALID_ARGUMENT);
    CHECK(std::strstr(turboembed_last_error(b), "texts must not be empty") != nullptr);
    CHECK(std::strstr(turboembed_last_error(a), "texts must not be empty") != nullptr);

    // Success on A clears only A.
    turboembed_embed_result *ra = embed_texts(a, "mock-embed", &view, 1, nullptr);
    CHECK(ra != nullptr);
    CHECK(turboembed_last_error(a)[0] == '\0');
    CHECK(std::strstr(turboembed_last_error(b), "texts must not be empty") != nullptr);

    // Interleaved engines over the same text agree byte-for-byte.
    CHECK(result_bytes_eq(ra, rb));

    turboembed_embed_result_free(ra);
    turboembed_embed_result_free(rb);
    turboembed_engine_destroy(a);
    turboembed_engine_destroy(b);
}

static void test_free_order() {
    // result_free before engine_destroy; destroy after all frees is clean.
    turboembed_engine *e = nullptr;
    CHECK_ST(turboembed_engine_create(TURBOEMBED_DEVICE_MOCK, nullptr, &e));
    CHECK_ST(turboembed_load_model(e, "mock-embed", 0));

    const char t[] = "free order";
    turboembed_str view = {t, sizeof(t) - 1};
    turboembed_embed_result *r1 = embed_texts(e, "mock-embed", &view, 1, nullptr);
    turboembed_embed_result *r2 = embed_texts(e, "mock-embed", &view, 1, nullptr);
    CHECK(r1 != nullptr && r2 != nullptr);
    turboembed_embed_result_free(r1);
    turboembed_embed_result_free(r2);
    turboembed_engine_destroy(e);
}

static void test_register_provider() {
    static const turboembed_provider_vtbl kVtbl = {
        "edge-test-provider", nullptr, nullptr, nullptr
    };
    CHECK(turboembed_register_provider(&kVtbl) == TURBOEMBED_ERR_NOT_IMPLEMENTED);
    CHECK(turboembed_register_provider(nullptr) == TURBOEMBED_ERR_NOT_IMPLEMENTED);
    CHECK(turboembed_last_error(nullptr) != nullptr);
    CHECK(std::strstr(turboembed_last_error(nullptr), "register_provider") != nullptr);
}

int main() {
    test_lifecycle();
    test_fail_loud_devices();
    test_names();
    test_load_model_errors();
    test_list_models();
    test_embed_edges();
    test_stream();
    test_error_isolation();
    test_free_order();
    test_register_provider();
    std::fprintf(
        stderr, "abi_edge_tests: %d passed, %d failed\n", g_passes, g_fails
    );
    return g_fails == 0 ? 0 : 1;
}
