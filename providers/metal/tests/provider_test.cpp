// SPDX-License-Identifier: Apache-2.0
//
// Vtable-level tests for the Metal provider.
//
// These drive `turbo_provider_get` directly, which reaches the cases the
// core filters out before a provider sees them: a caller that declares a
// smaller `struct_size`, an option enumeration this build does not know, a
// session wider than the model, buffers in placements unified memory does
// not have, and the reranker head with raw and activated scores.
//
// The library under test is the one built next to this binary; set
// `TURBO_PROVIDER_LIB` to test another build. The embedding bundle comes
// from `TURBO_LIVE_BUNDLE` and the reranker bundle from
// `TURBO_LIVE_RERANK_BUNDLE` (both imported with `safetensors` and
// `hf_config` artifacts); tests that need one skip and say so when it is
// not set. Without a Metal device the device tests report the provider's
// own reason and the bundle tests skip.

#include "turbo/turbo_provider.h"
#include "turbo/turbo_types.h"

#include <dlfcn.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            ++g_failures;                                                                                              \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                              \
        }                                                                                                              \
    } while (0)

#define CHECK_EQ(got, want)                                                                                            \
    do {                                                                                                               \
        const long long g = static_cast<long long>(got);                                                               \
        const long long w = static_cast<long long>(want);                                                              \
        if (g != w) {                                                                                                  \
            ++g_failures;                                                                                              \
            std::printf("  FAIL %s:%d: %s is %lld, expected %lld\n", __FILE__, __LINE__, #got, g, w);                  \
        }                                                                                                              \
    } while (0)

struct Err {
    turbo_error e{};
    Err() { e.struct_size = sizeof(turbo_error); }
    turbo_error *p() { return &e; }
    const char *what() const { return e.message; }
};

bool ok(int32_t rc, Err &err, const char *what) {
    if (rc != TURBO_OK) {
        ++g_failures;
        std::printf("  FAIL %s: status %d field %u: %s\n", what, rc, err.e.field, err.what());
        return false;
    }
    return true;
}

turbo_text text_of(const std::string &s) { return turbo_text{s.data(), s.size()}; }

const char *env(const char *name) {
    const char *v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

const turbo_provider_vtbl *load_provider() {
    const char *path = env("TURBO_PROVIDER_LIB");
    if (path == nullptr) {
        path = TURBO_PROVIDER_LIB_PATH;
    }
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        std::printf("cannot dlopen %s: %s\n", path, dlerror());
        std::exit(2);
    }
    auto *get = reinterpret_cast<turbo_provider_get_fn>(dlsym(lib, "turbo_provider_get"));
    if (get == nullptr) {
        std::printf("%s exports no turbo_provider_get\n", path);
        std::exit(2);
    }
    const turbo_provider_vtbl *vt = get(TURBO_PROVIDER_ABI_VERSION);
    if (vt == nullptr) {
        std::printf("%s rejects provider ABI version %u\n", path, TURBO_PROVIDER_ABI_VERSION);
        std::exit(2);
    }
    std::printf("provider `%s` %s from %s\n", vt->id, vt->version, path);
    return vt;
}

const turbo_provider_vtbl *vt() {
    static const turbo_provider_vtbl *v = load_provider();
    return v;
}

uint32_t ordinal() {
    if (const char *v = env("TURBO_LIVE_ORDINAL")) {
        return static_cast<uint32_t>(std::atoi(v));
    }
    return 0;
}

/// Number of devices, or 0 with the provider's reason printed.
uint32_t device_count() {
    uint32_t n = 0;
    Err err;
    const int32_t rc = vt()->device_count(vt()->state, &n, err.p());
    if (rc != TURBO_OK) {
        std::printf("  device_count: status %d: %s\n", rc, err.what());
        return 0;
    }
    return n;
}

turbo_device_info device_info(uint32_t ordinal) {
    turbo_device_info info{};
    info.struct_size = sizeof(info);
    Err err;
    ok(vt()->device_info(vt()->state, ordinal, &info, err.p()), err, "device_info");
    return info;
}

turbo_capability capability(uint32_t ordinal, uint32_t task, uint32_t modality) {
    turbo_capability cap{};
    cap.struct_size = sizeof(cap);
    Err err;
    ok(vt()->capability(vt()->state, ordinal, task, modality, &cap, err.p()), err, "capability");
    return cap;
}

struct Fixture {
    void *ctx = nullptr;
    void *model = nullptr;
    void *session = nullptr;
    turbo_model_info info{};

    bool open(uint32_t max_batch, uint32_t max_seq, const char *bundle_var = "TURBO_LIVE_BUNDLE",
              const turbo_kv *opts = nullptr, uint32_t n_opts = 0) {
        const char *dir = env(bundle_var);
        if (dir == nullptr) {
            std::printf("  skipped: %s is not set\n", bundle_var);
            return false;
        }
        if (device_count() == 0) {
            std::printf("  skipped: no Metal device\n");
            return false;
        }
        Err err;
        if (!ok(vt()->context_create(vt()->state, ordinal(), nullptr, &ctx, err.p()), err, "context_create")) {
            return false;
        }
        turbo_model_desc md{};
        md.struct_size = sizeof(md);
        md.options = opts;
        md.n_options = n_opts;
        const std::string path(dir);
        if (!ok(vt()->model_load(ctx, text_of(path), &md, &model, err.p()), err, "model_load")) {
            return false;
        }
        info.struct_size = sizeof(info);
        if (!ok(vt()->model_info(model, &info, err.p()), err, "model_info")) {
            return false;
        }
        turbo_session_desc sd{};
        sd.struct_size = sizeof(sd);
        sd.max_batch = max_batch;
        sd.max_seq = max_seq;
        return ok(vt()->session_create(model, &sd, &session, err.p()), err, "session_create");
    }

    /// Embed `texts` and return the rows of the first output.
    std::vector<std::vector<float>> embed(const std::vector<std::string> &texts, const turbo_embed_options *opts) {
        std::vector<turbo_text> views;
        for (const auto &t : texts) {
            views.push_back(text_of(t));
        }
        Err err;
        std::vector<std::vector<float>> rows;
        if (!ok(vt()->session_write_text(session, views.data(), static_cast<uint32_t>(views.size()), opts, err.p()),
                err, "session_write_text")) {
            return rows;
        }
        turbo_provider_result r{};
        r.struct_size = sizeof(r);
        if (!ok(vt()->session_run(session, nullptr, &r, err.p()), err, "session_run")) {
            return rows;
        }
        CHECK_EQ(r.n_outputs, 1);
        const turbo_provider_output &o = r.outputs[0];
        CHECK_EQ(o.ndim, 2);
        CHECK_EQ(o.shape[0], texts.size());
        CHECK_EQ(o.buffer.desc.placement, TURBO_PLACE_SHARED);
        CHECK_EQ(o.buffer.desc.dtype, TURBO_DTYPE_F32);
        const size_t dim = static_cast<size_t>(o.shape[1]);
        const auto *p = static_cast<const float *>(o.buffer.host_ptr);
        CHECK(p != nullptr);
        for (size_t i = 0; i < texts.size(); ++i) {
            rows.emplace_back(p + i * dim, p + (i + 1) * dim);
        }
        return rows;
    }

    ~Fixture() {
        if (session != nullptr) {
            vt()->session_release(session);
        }
        if (model != nullptr) {
            vt()->model_release(model);
        }
        if (ctx != nullptr) {
            vt()->context_release(ctx);
        }
    }
};

float norm(const std::vector<float> &v) {
    double s = 0.0;
    for (float x : v) {
        s += static_cast<double>(x) * x;
    }
    return static_cast<float>(std::sqrt(s));
}

float cosine(const std::vector<float> &a, const std::vector<float> &b) {
    double dot = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
    }
    return static_cast<float>(dot / (norm(a) * norm(b)));
}

// ---------------------------------------------------------------------------
// Devices and capabilities
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Devices and capabilities
// ---------------------------------------------------------------------------

void the_device_is_an_apple_gpu_with_unified_memory() {
    const uint32_t n = device_count();
    if (n == 0) {
        std::printf("  skipped: no Metal device\n");
        return;
    }
    CHECK_EQ(n, 1);
    const turbo_device_info info = device_info(0);
    std::printf("  device `%s` vendor `%s` runtime `%s` driver `%s` caps %#llx memory %llu\n", info.name, info.vendor,
                info.runtime_version, info.driver_version, static_cast<unsigned long long>(info.caps),
                static_cast<unsigned long long>(info.memory_total));
    CHECK_EQ(info.kind, TURBO_DEVICE_IGPU);
    CHECK_EQ(info.ordinal, 0);
    CHECK_EQ(info.vendor_id, 0x106b);
    CHECK(std::string(info.vendor) == "Apple");
    CHECK(std::string(info.provider_id) == "metal");
    CHECK(std::string(info.name).find("(Metal)") != std::string::npos);
    CHECK(info.memory_total > 0);
    // Results are the GPU's own memory, and that memory is the host's too.
    CHECK((info.caps & TURBO_CAP_DEVICE_RESULT) != 0);
    CHECK((info.caps & TURBO_CAP_UNIFIED_MEMORY) != 0);
    CHECK((info.caps & TURBO_CAP_HOST_PTR_IMPORT) != 0);
    CHECK((info.caps & TURBO_CAP_DETERMINISTIC) != 0);
    CHECK((info.caps & TURBO_CAP_OPT_POOLING_OVERRIDE) != 0);
    CHECK((info.caps & TURBO_CAP_OPT_TOP_N) != 0);
    CHECK((info.caps & TURBO_CAP_OPT_RAW_SCORES) != 0);
    // One past the end is DEVICE_NOT_FOUND, not a crash.
    turbo_device_info past{};
    past.struct_size = sizeof(past);
    Err err;
    CHECK_EQ(vt()->device_info(vt()->state, 1, &past, err.p()), TURBO_E_DEVICE_NOT_FOUND);
}

void capability_offers_embed_and_rerank_in_fp32() {
    if (device_count() == 0) {
        std::printf("  skipped: no Metal device\n");
        return;
    }
    for (uint32_t task : {TURBO_TASK_EMBED, TURBO_TASK_RERANK}) {
        const turbo_capability c = capability(0, task, TURBO_MODALITY_TEXT);
        std::printf("  task %u x TEXT: status %u dtype %u reference %u notes `%s`\n", task, c.status, c.dtype,
                    c.reference_dtype, c.notes);
        CHECK_EQ(c.status, TURBO_CAP_EXPERIMENTAL);
        CHECK_EQ(c.dtype, TURBO_DTYPE_F32);
        CHECK_EQ(c.reference_dtype, TURBO_DTYPE_F32);
        CHECK_EQ(c.deterministic, 1);
        // FP32 is held to the suite's own 0.9995; the cell states no floor of its own.
        CHECK(c.cosine_floor == 0.0f);
    }
    for (uint32_t task : {TURBO_TASK_CLASSIFY, TURBO_TASK_TOKEN_CLASSIFY, TURBO_TASK_GENERATE, TURBO_TASK_RUN}) {
        CHECK_EQ(capability(0, task, TURBO_MODALITY_TEXT).status, TURBO_CAP_UNSUPPORTED);
    }
    CHECK_EQ(capability(0, TURBO_TASK_EMBED, TURBO_MODALITY_IMAGE).status, TURBO_CAP_UNSUPPORTED);
}

void can_run_checks_the_bundle_and_task() {
    const char *dir = env("TURBO_LIVE_BUNDLE");
    if (dir == nullptr || device_count() == 0) {
        std::printf("  skipped: TURBO_LIVE_BUNDLE is not set or no Metal device\n");
        return;
    }
    const std::string path(dir);
    Err err;
    ok(vt()->can_run(vt()->state, 0, text_of(path), TURBO_TASK_EMBED, TURBO_MODALITY_TEXT, err.p()), err, "can_run");
    Err e2;
    CHECK_EQ(vt()->can_run(vt()->state, 0, text_of(path), TURBO_TASK_RERANK, TURBO_MODALITY_TEXT, e2.p()),
             TURBO_E_UNSUPPORTED_TASK);
    Err e3;
    CHECK_EQ(vt()->can_run(vt()->state, 0, text_of(path), TURBO_TASK_CLASSIFY, TURBO_MODALITY_TEXT, e3.p()),
             TURBO_E_UNSUPPORTED_TASK);
    const std::string missing = "/nonexistent/bundle";
    Err e4;
    CHECK_EQ(vt()->can_run(vt()->state, 0, text_of(missing), TURBO_TASK_EMBED, TURBO_MODALITY_TEXT, e4.p()),
             TURBO_E_BUNDLE_NOT_FOUND);
    if (const char *rr = env("TURBO_LIVE_RERANK_BUNDLE")) {
        const std::string rpath(rr);
        Err e5;
        ok(vt()->can_run(vt()->state, 0, text_of(rpath), TURBO_TASK_RERANK, TURBO_MODALITY_TEXT, e5.p()), e5,
           "can_run (rerank)");
        Err e6;
        CHECK_EQ(vt()->can_run(vt()->state, 0, text_of(rpath), TURBO_TASK_EMBED, TURBO_MODALITY_TEXT, e6.p()),
                 TURBO_E_UNSUPPORTED_TASK);
    }
}

// ---------------------------------------------------------------------------
// Struct sizes and enumerations
// ---------------------------------------------------------------------------

void struct_size_rules_are_enforced() {
    if (device_count() == 0) {
        std::printf("  skipped: no Metal device\n");
        return;
    }
    // A smaller, older struct is filled up to its declared size.
    turbo_device_info small{};
    small.struct_size = 16;
    Err err;
    ok(vt()->device_info(vt()->state, 0, &small, err.p()), err, "device_info (small)");
    CHECK_EQ(small.struct_size, 16);
    CHECK_EQ(small.kind, TURBO_DEVICE_IGPU);
    CHECK_EQ(small.caps, 0); // beyond the declared size: untouched
    // A larger one is refused: the provider cannot fill fields it does not know.
    turbo_device_info big{};
    big.struct_size = sizeof(big) + 8;
    Err e2;
    CHECK_EQ(vt()->device_info(vt()->state, 0, &big, e2.p()), TURBO_E_INVALID_STRUCT_SIZE);
}

void unknown_option_enums_name_their_field() {
    Fixture f;
    if (!f.open(2, 64)) {
        return;
    }
    const std::string text = "hello";
    const turbo_text view = text_of(text);
    turbo_embed_options o{};
    o.struct_size = sizeof(o);
    o.truncate = 99;
    Err e1;
    CHECK_EQ(vt()->session_write_text(f.session, &view, 1, &o, e1.p()), TURBO_E_INVALID_ENUM);
    CHECK_EQ(e1.e.field, 2);
    o.truncate = TURBO_TRUNCATE_MODEL;
    o.prompt_role = 99;
    Err e2;
    CHECK_EQ(vt()->session_write_text(f.session, &view, 1, &o, e2.p()), TURBO_E_INVALID_ENUM);
    CHECK_EQ(e2.e.field, 4);
    o.prompt_role = TURBO_PROMPT_NONE;
    o.normalize = 99;
    Err e3;
    CHECK_EQ(vt()->session_write_text(f.session, &view, 1, &o, e3.p()), TURBO_E_INVALID_ENUM);
    CHECK_EQ(e3.e.field, 5);
    o.normalize = TURBO_NORMALIZE_MODEL;
    o.pooling = 99;
    Err e4;
    CHECK_EQ(vt()->session_write_text(f.session, &view, 1, &o, e4.p()), TURBO_E_INVALID_ENUM);
    CHECK_EQ(e4.e.field, 6);
    o.pooling = TURBO_POOLING_MODEL;
    o.output_dim = f.info.dim + 1;
    Err e5;
    CHECK_EQ(vt()->session_write_text(f.session, &view, 1, &o, e5.p()), TURBO_E_UNSUPPORTED_OPTION);
    CHECK_EQ(e5.e.field, 7);
    o.output_dim = 0;
    o.output_dtype = TURBO_OUTPUT_F16;
    Err e6;
    CHECK_EQ(vt()->session_write_text(f.session, &view, 1, &o, e6.p()), TURBO_E_UNSUPPORTED_OPTION);
    CHECK_EQ(e6.e.field, 8);
    o.output_dtype = 99;
    Err e7;
    CHECK_EQ(vt()->session_write_text(f.session, &view, 1, &o, e7.p()), TURBO_E_INVALID_ENUM);
    CHECK_EQ(e7.e.field, 8);
    o.output_dtype = TURBO_OUTPUT_F32;
    Err e8;
    ok(vt()->session_write_text(f.session, &view, 1, &o, e8.p()), e8, "session_write_text (output F32)");
    // A budget past the session is CAPACITY on field 3.
    o.max_tokens = 65;
    Err e9;
    CHECK_EQ(vt()->session_write_text(f.session, &view, 1, &o, e9.p()), TURBO_E_CAPACITY);
    CHECK_EQ(e9.e.field, 3);
}

// ---------------------------------------------------------------------------
// Models and sessions
// ---------------------------------------------------------------------------

void model_info_puts_everything_but_the_tokenizer_on_the_gpu() {
    Fixture f;
    if (!f.open(1, 32)) {
        return;
    }
    CHECK_EQ(f.info.task, TURBO_TASK_EMBED);
    CHECK_EQ(f.info.kind, TURBO_MODEL_EMBEDDING);
    CHECK_EQ(f.info.fully_accelerated, 0);
    CHECK_EQ(f.info.stage_placement[TURBO_STAGE_TOKENIZE], TURBO_STAGE_HOST);
    CHECK_EQ(f.info.stage_placement[TURBO_STAGE_ENCODE], TURBO_STAGE_DEVICE);
    CHECK_EQ(f.info.stage_placement[TURBO_STAGE_POOL], TURBO_STAGE_DEVICE);
    CHECK_EQ(f.info.stage_placement[TURBO_STAGE_NORMALIZE], TURBO_STAGE_DEVICE);
    CHECK_EQ(f.info.stage_placement[TURBO_STAGE_POSTPROCESS], TURBO_STAGE_UNUSED);
    CHECK_EQ(f.info.dtype_used, TURBO_DTYPE_F32);
    CHECK_EQ(f.info.pooling, TURBO_POOLING_MEAN);
    CHECK_EQ(f.info.normalize, TURBO_NORMALIZE_L2);
    CHECK(f.info.max_seq > 2);
    CHECK(f.info.dim > 0);
    CHECK(std::string(f.info.provider_id) == "metal");
    std::printf("  model `%s` dim %u max_seq %u max_batch %u\n", f.info.model_id, f.info.dim, f.info.max_seq,
                f.info.max_batch);
}

void session_wider_than_the_model_is_refused() {
    Fixture f;
    if (!f.open(1, 16)) {
        return;
    }
    turbo_session_desc sd{};
    sd.struct_size = sizeof(sd);
    sd.max_batch = 1;
    sd.max_seq = f.info.max_seq + 1;
    void *s = nullptr;
    Err err;
    CHECK_EQ(vt()->session_create(f.model, &sd, &s, err.p()), TURBO_E_CAPACITY);
    CHECK(s == nullptr);
    sd.max_seq = 16;
    sd.max_batch = f.info.max_batch + 1;
    Err e2;
    CHECK_EQ(vt()->session_create(f.model, &sd, &s, e2.p()), TURBO_E_CAPACITY);
    CHECK(s == nullptr);
}

void embeddings_are_unit_norm_and_bitwise_repeatable() {
    Fixture f;
    if (!f.open(2, 64)) {
        return;
    }
    const std::vector<std::string> texts = {"a brown dog runs through the grass", "the stock market closed higher"};
    const auto first = f.embed(texts, nullptr);
    const auto second = f.embed(texts, nullptr);
    if (first.size() != 2 || second.size() != 2) {
        return;
    }
    for (size_t i = 0; i < 2; ++i) {
        CHECK_EQ(first[i].size(), f.info.dim);
        CHECK(std::fabs(norm(first[i]) - 1.0f) < 1e-4f);
        CHECK(std::memcmp(first[i].data(), second[i].data(), first[i].size() * sizeof(float)) == 0);
    }
    const float c = cosine(first[0], first[1]);
    std::printf("  cosine(dog, market) = %.4f\n", static_cast<double>(c));
    CHECK(c < 0.9f);
    // Single-row runs equal the batch rows.
    const auto solo = f.embed({texts[1]}, nullptr);
    if (solo.size() == 1) {
        CHECK(cosine(solo[0], first[1]) > 0.9999f);
    }
    turbo_session_stats st{};
    st.struct_size = sizeof(st);
    Err err;
    ok(vt()->session_stats(f.session, &st, err.p()), err, "session_stats");
    CHECK_EQ(st.runs, 3);
    // Unified memory: nothing crosses a bus in either direction.
    CHECK_EQ(st.h2d_bytes, 0);
    CHECK_EQ(st.d2h_bytes, 0);
    CHECK(st.output_bytes > 0);
    CHECK(st.host_allocs == UINT64_MAX);
}

void normalize_pooling_and_output_dim_are_honored() {
    Fixture f;
    if (!f.open(1, 64)) {
        return;
    }
    const std::vector<std::string> texts = {"the quick brown fox"};
    turbo_embed_options o{};
    o.struct_size = sizeof(o);
    o.normalize = TURBO_NORMALIZE_NONE;
    const auto raw = f.embed(texts, &o);
    if (raw.size() == 1) {
        CHECK(std::fabs(norm(raw[0]) - 1.0f) > 1e-3f);
    }
    o.normalize = TURBO_NORMALIZE_MODEL;
    o.pooling = TURBO_POOLING_CLS;
    const auto cls = f.embed(texts, &o);
    o.pooling = TURBO_POOLING_MEAN;
    const auto mean = f.embed(texts, &o);
    o.pooling = TURBO_POOLING_LAST;
    const auto last = f.embed(texts, &o);
    if (cls.size() == 1 && mean.size() == 1 && last.size() == 1) {
        CHECK(cosine(cls[0], mean[0]) < 0.9999f);
        CHECK(cosine(last[0], mean[0]) < 0.9999f);
        CHECK(cosine(last[0], cls[0]) < 0.9999f);
    }
    // A truncated width is the head of the full vector, re-normalized.
    o.pooling = TURBO_POOLING_MODEL;
    o.normalize = TURBO_NORMALIZE_NONE;
    const auto full = f.embed(texts, &o);
    o.normalize = TURBO_NORMALIZE_MODEL;
    o.output_dim = 128;
    turbo_text view = text_of(texts[0]);
    Err err;
    ok(vt()->session_write_text(f.session, &view, 1, &o, err.p()), err, "session_write_text (output_dim)");
    turbo_provider_result r{};
    r.struct_size = sizeof(r);
    if (ok(vt()->session_run(f.session, nullptr, &r, err.p()), err, "session_run (output_dim)") && full.size() == 1) {
        CHECK_EQ(r.outputs[0].shape[1], 128);
        const auto *p = static_cast<const float *>(r.outputs[0].buffer.host_ptr);
        std::vector<float> v(p, p + 128);
        CHECK(std::fabs(norm(v) - 1.0f) < 1e-4f);
        std::vector<float> head(full[0].begin(), full[0].begin() + 128);
        CHECK(cosine(v, head) > 0.99999f);
    }
}

void truncation_none_over_budget_is_a_capacity_error() {
    Fixture f;
    if (!f.open(1, 8)) {
        return;
    }
    const std::string text = "one two three four five six seven eight nine ten";
    const turbo_text view = text_of(text);
    turbo_embed_options o{};
    o.struct_size = sizeof(o);
    o.truncate = TURBO_TRUNCATE_NONE;
    Err err;
    CHECK_EQ(vt()->session_write_text(f.session, &view, 1, &o, err.p()), TURBO_E_CAPACITY);
    o.truncate = TURBO_TRUNCATE_RIGHT;
    const auto right = f.embed({text}, &o);
    o.truncate = TURBO_TRUNCATE_LEFT;
    const auto left = f.embed({text}, &o);
    if (right.size() == 1 && left.size() == 1) {
        CHECK(std::memcmp(right[0].data(), left[0].data(), 4 * right[0].size()) != 0);
    }
    // The model's own budget is the session's when max_tokens is 0; a
    // smaller explicit budget changes the vector.
    o.truncate = TURBO_TRUNCATE_RIGHT;
    o.max_tokens = 4;
    const auto four = f.embed({text}, &o);
    if (four.size() == 1 && right.size() == 1) {
        CHECK(std::memcmp(four[0].data(), right[0].data(), 4 * four[0].size()) != 0);
    }
}

void token_rows_equal_text_rows() {
    Fixture f;
    if (!f.open(1, 8)) {
        return;
    }
    // [CLS] hello [SEP] in the BERT uncased vocabulary.
    int32_t ids[8] = {101, 7592, 102, 0, 0, 0, 0, 0};
    int32_t mask[8] = {1, 1, 1, 0, 0, 0, 0, 0};
    turbo_token_batch b{};
    b.struct_size = sizeof(b);
    b.batch = 1;
    b.seq = 8;
    b.ids = ids;
    b.mask = mask;
    Err e1;
    ok(vt()->session_write_tokens(f.session, &b, e1.p()), e1, "session_write_tokens");
    turbo_provider_result r{};
    r.struct_size = sizeof(r);
    Err e2;
    if (ok(vt()->session_run(f.session, nullptr, &r, e2.p()), e2, "session_run (tokens)")) {
        const auto *p = static_cast<const float *>(r.outputs[0].buffer.host_ptr);
        std::vector<float> v(p, p + f.info.dim);
        CHECK(std::fabs(norm(v) - 1.0f) < 1e-4f);
        const auto via_text = f.embed({"hello"}, nullptr);
        if (via_text.size() == 1) {
            CHECK(cosine(v, via_text[0]) > 0.9999f);
        }
    }
    // Token types are a real input to a BERT encoder: a second segment
    // changes the vector rather than being refused.
    int32_t types[8] = {0, 1, 1, 0, 0, 0, 0, 0};
    b.types = types;
    Err e3;
    ok(vt()->session_write_tokens(f.session, &b, e3.p()), e3, "session_write_tokens (types)");
    turbo_provider_result r2{};
    r2.struct_size = sizeof(r2);
    Err e4;
    if (ok(vt()->session_run(f.session, nullptr, &r2, e4.p()), e4, "session_run (types)")) {
        const auto *p = static_cast<const float *>(r2.outputs[0].buffer.host_ptr);
        std::vector<float> typed(p, p + f.info.dim);
        const auto plain = f.embed({"hello"}, nullptr);
        if (plain.size() == 1) {
            CHECK(cosine(typed, plain[0]) < 0.9999f);
        }
    }
    // A batch wider than the session is CAPACITY.
    b.batch = 2;
    Err e5;
    CHECK_EQ(vt()->session_write_tokens(f.session, &b, e5.p()), TURBO_E_CAPACITY);
}

void other_tasks_are_refused_on_an_embedding_session() {
    Fixture f;
    if (!f.open(1, 8)) {
        return;
    }
    const std::string q = "q";
    const turbo_text qv = text_of(q);
    Err e1;
    CHECK_EQ(vt()->session_write_pairs(f.session, &qv, &qv, 1, nullptr, e1.p()), TURBO_E_UNSUPPORTED_TASK);
    Err e2;
    CHECK_EQ(vt()->session_write_text_classify(f.session, &qv, 1, nullptr, e2.p()), TURBO_E_UNSUPPORTED_TASK);
    Err e3;
    turbo_provider_result r{};
    r.struct_size = sizeof(r);
    // Nothing written yet on a fresh session: INVALID_STATE, not a run.
    CHECK_EQ(vt()->session_run(f.session, nullptr, &r, e3.p()), TURBO_E_INVALID_STATE);
}

void model_options_are_checked() {
    const char *dir = env("TURBO_LIVE_BUNDLE");
    if (dir == nullptr || device_count() == 0) {
        std::printf("  skipped: TURBO_LIVE_BUNDLE is not set or no Metal device\n");
        return;
    }
    const std::string unknown = "threads", one = "1";
    turbo_kv kv{text_of(unknown), text_of(one)};
    Fixture f;
    Err err;
    ok(vt()->context_create(vt()->state, 0, nullptr, &f.ctx, err.p()), err, "context_create");
    turbo_model_desc md{};
    md.struct_size = sizeof(md);
    md.options = &kv;
    md.n_options = 1;
    const std::string path(dir);
    Err e2;
    CHECK_EQ(vt()->model_load(f.ctx, text_of(path), &md, &f.model, e2.p()), TURBO_E_INVALID_ARGUMENT);
    CHECK(f.model == nullptr);
    // A bundle without the safetensors artifact names what it has.
    const std::string missing = "/nonexistent/bundle";
    Err e3;
    CHECK_EQ(vt()->model_load(f.ctx, text_of(missing), nullptr, &f.model, e3.p()), TURBO_E_BUNDLE_NOT_FOUND);
}

// ---------------------------------------------------------------------------
// Buffers: shared (unified) and host
// ---------------------------------------------------------------------------

void buffers_are_shared_or_host() {
    if (device_count() == 0) {
        std::printf("  skipped: no Metal device\n");
        return;
    }
    void *ctx = nullptr;
    Err err;
    if (!ok(vt()->context_create(vt()->state, 0, nullptr, &ctx, err.p()), err, "context_create")) {
        return;
    }
    turbo_buffer_desc d{};
    d.struct_size = sizeof(d);
    d.placement = TURBO_PLACE_DEVICE;
    d.dtype = TURBO_DTYPE_F32;
    d.ndim = 1;
    d.shape[0] = 16;
    turbo_provider_buffer b{};
    b.struct_size = sizeof(b);
    Err e2;
    CHECK_EQ(vt()->buffer_alloc(ctx, &d, &b, e2.p()), TURBO_E_UNSUPPORTED_PLACEMENT);
    d.placement = TURBO_PLACE_PINNED;
    Err e2b;
    CHECK_EQ(vt()->buffer_alloc(ctx, &d, &b, e2b.p()), TURBO_E_UNSUPPORTED_PLACEMENT);
    // Shared: an MTLBuffer whose contents are host-visible.
    d.placement = TURBO_PLACE_SHARED;
    Err e3;
    if (ok(vt()->buffer_alloc(ctx, &d, &b, e3.p()), e3, "buffer_alloc (shared)")) {
        CHECK_EQ(b.desc.bytes, 64);
        CHECK_EQ(b.desc.placement, TURBO_PLACE_SHARED);
        CHECK(b.host_ptr != nullptr);
        float back[16];
        Err e4;
        ok(vt()->buffer_read(b.handle, back, sizeof(back), e4.p()), e4, "buffer_read");
        Err e5;
        CHECK_EQ(vt()->buffer_read(b.handle, back, 32, e5.p()), TURBO_E_CAPACITY);
        turbo_native_handle h{};
        h.struct_size = sizeof(h);
        Err e6;
        ok(vt()->buffer_export(b.handle, TURBO_HANDLE_MTL_BUFFER, &h, e6.p()), e6, "buffer_export (mtl)");
        CHECK_EQ(h.kind, TURBO_HANDLE_MTL_BUFFER);
        CHECK(h.handle != 0);
        turbo_native_handle hp{};
        hp.struct_size = sizeof(hp);
        Err e7;
        ok(vt()->buffer_export(b.handle, TURBO_HANDLE_HOST_PTR, &hp, e7.p()), e7, "buffer_export (host ptr)");
        CHECK(reinterpret_cast<void *>(hp.handle) == b.host_ptr);
        Err e8;
        CHECK_EQ(vt()->buffer_export(b.handle, TURBO_HANDLE_CUDA_PTR, &hp, e8.p()), TURBO_E_UNSUPPORTED);
        vt()->buffer_release(b.handle);
    }
    // Host: plain memory, exports its pointer only.
    d.placement = TURBO_PLACE_HOST;
    Err e9;
    if (ok(vt()->buffer_alloc(ctx, &d, &b, e9.p()), e9, "buffer_alloc (host)")) {
        CHECK_EQ(b.desc.placement, TURBO_PLACE_HOST);
        turbo_native_handle h{};
        h.struct_size = sizeof(h);
        Err e10;
        CHECK_EQ(vt()->buffer_export(b.handle, TURBO_HANDLE_MTL_BUFFER, &h, e10.p()), TURBO_E_UNSUPPORTED);
        Err e11;
        ok(vt()->buffer_export(b.handle, TURBO_HANDLE_HOST_PTR, &h, e11.p()), e11, "buffer_export (host)");
        CHECK(reinterpret_cast<void *>(h.handle) == b.host_ptr);
        vt()->buffer_release(b.handle);
    }
    // Import: caller memory is wrapped, not copied, and never freed.
    float mine[16] = {1.5f};
    turbo_native_handle in{};
    in.struct_size = sizeof(in);
    in.kind = TURBO_HANDLE_HOST_PTR;
    in.handle = reinterpret_cast<uint64_t>(mine);
    Err e12;
    if (ok(vt()->buffer_import(ctx, &d, &in, &b, e12.p()), e12, "buffer_import")) {
        CHECK(b.host_ptr == mine);
        float back[16];
        Err e13;
        ok(vt()->buffer_read(b.handle, back, sizeof(back), e13.p()), e13, "buffer_read (imported)");
        CHECK(back[0] == 1.5f);
        vt()->buffer_release(b.handle);
        CHECK(mine[0] == 1.5f);
    }
    in.kind = TURBO_HANDLE_MTL_BUFFER;
    Err e14;
    CHECK_EQ(vt()->buffer_import(ctx, &d, &in, &b, e14.p()), TURBO_E_UNSUPPORTED);
    vt()->context_release(ctx);
}

// ---------------------------------------------------------------------------
// Reranking
// ---------------------------------------------------------------------------

struct Scored {
    std::vector<float> scores;
    std::vector<int32_t> sorted;
};

/// The `contract.activation` string of the bundle named by `var` ("" when absent).
std::string bundle_activation(const char *var) {
    const char *dir = env(var);
    if (dir == nullptr) {
        return "";
    }
    std::FILE *fp = std::fopen((std::string(dir) + "/bundle.json").c_str(), "rb");
    if (fp == nullptr) {
        return "";
    }
    std::string text;
    char buf[4096];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), fp)) > 0) {
        text.append(buf, n);
    }
    std::fclose(fp);
    const std::string key = "\"activation\": \"";
    const size_t at = text.find(key);
    if (at == std::string::npos) {
        return "";
    }
    const size_t start = at + key.size();
    return text.substr(start, text.find('"', start) - start);
}

Scored rerank(Fixture &f, const std::string &query, const std::vector<std::string> &docs, const turbo_rerank_options *o) {
    Scored out;
    std::vector<turbo_text> views;
    for (const auto &d : docs) {
        views.push_back(text_of(d));
    }
    const turbo_text qv = text_of(query);
    Err err;
    if (!ok(vt()->session_write_pairs(f.session, &qv, views.data(), static_cast<uint32_t>(views.size()), o, err.p()), err,
            "session_write_pairs")) {
        return out;
    }
    turbo_provider_result r{};
    r.struct_size = sizeof(r);
    Err e2;
    if (!ok(vt()->session_run(f.session, nullptr, &r, e2.p()), e2, "session_run (pairs)")) {
        return out;
    }
    CHECK(r.n_outputs >= 1);
    CHECK_EQ(r.outputs[0].ndim, 1);
    CHECK_EQ(r.outputs[0].shape[0], docs.size());
    CHECK_EQ(r.outputs[0].buffer.desc.placement, TURBO_PLACE_SHARED);
    const auto *p = static_cast<const float *>(r.outputs[0].buffer.host_ptr);
    out.scores.assign(p, p + docs.size());
    if (r.n_outputs == 2) {
        CHECK(std::string(r.outputs[1].name.ptr, r.outputs[1].name.len) == "sorted");
        CHECK_EQ(r.outputs[1].buffer.desc.dtype, TURBO_DTYPE_I32);
        const auto *s = static_cast<const int32_t *>(r.outputs[1].buffer.host_ptr);
        out.sorted.assign(s, s + r.outputs[1].shape[0]);
    }
    return out;
}

void reranker_scores_sort_and_activate() {
    Fixture f;
    if (!f.open(4, 128, "TURBO_LIVE_RERANK_BUNDLE")) {
        return;
    }
    CHECK_EQ(f.info.task, TURBO_TASK_RERANK);
    CHECK_EQ(f.info.kind, TURBO_MODEL_RERANKER);
    CHECK_EQ(f.info.stage_placement[TURBO_STAGE_POSTPROCESS], TURBO_STAGE_DEVICE);
    const std::string query = "how do I bake sourdough bread";
    const std::vector<std::string> docs = {
        "The stock market closed higher on Tuesday.",
        "Mix flour, water and starter, then let the dough rise before baking in a hot oven.",
        "A sourdough loaf needs a long, slow fermentation and a very hot oven.",
        "Bicycle tire pressure depends on the rider's weight.",
    };
    turbo_rerank_options o{};
    o.struct_size = sizeof(o);
    o.return_sorted = 1;
    const Scored s = rerank(f, query, docs, &o);
    if (s.scores.size() != 4) {
        return;
    }
    // The bundle's contract decides whether the logits are activated: the
    // ms-marco cross-encoders declare Identity, so their scores are logits.
    const bool sigmoid = bundle_activation("TURBO_LIVE_RERANK_BUNDLE") == "sigmoid";
    const float midpoint = sigmoid ? 0.5f : 0.0f;
    for (size_t i = 0; i < 4; ++i) {
        std::printf("  score[%zu] = %.4f\n", i, static_cast<double>(s.scores[i]));
        if (sigmoid) {
            CHECK(s.scores[i] > 0.0f && s.scores[i] < 1.0f);
        }
    }
    // Both bread documents outrank both off-topic ones; the direct answer
    // is the best of all and sits above the midpoint.
    CHECK(s.scores[2] > midpoint);
    CHECK(s.scores[1] > s.scores[0] && s.scores[1] > s.scores[3]);
    CHECK(s.scores[2] > s.scores[0] && s.scores[2] > s.scores[3]);
    CHECK(s.scores[2] > s.scores[1]);
    CHECK_EQ(s.sorted.size(), 4);
    for (size_t i = 1; i < s.sorted.size(); ++i) {
        CHECK(s.scores[s.sorted[i - 1]] >= s.scores[s.sorted[i]]);
    }
    // Raw scores are the logits: the same order; the sigmoid of each is the
    // activated score, or they are the scores themselves under Identity.
    o.raw_scores = 1;
    const Scored raw = rerank(f, query, docs, &o);
    if (raw.scores.size() == 4) {
        bool outside = false;
        for (size_t i = 0; i < 4; ++i) {
            outside = outside || raw.scores[i] <= 0.0f || raw.scores[i] >= 1.0f;
            if (sigmoid) {
                const float sig = 1.0f / (1.0f + std::exp(-raw.scores[i]));
                CHECK(std::fabs(sig - s.scores[i]) < 1e-4f);
            } else {
                CHECK(raw.scores[i] == s.scores[i]);
            }
        }
        CHECK(outside);
        CHECK(raw.sorted == s.sorted);
    }
    // top_n limits the sorted output, not the scores.
    o.raw_scores = 0;
    o.top_n = 2;
    const Scored top = rerank(f, query, docs, &o);
    if (top.scores.size() == 4) {
        CHECK_EQ(top.sorted.size(), 2);
        CHECK(top.sorted[0] == s.sorted[0]);
    }
    // Repeatable bit for bit.
    const Scored again = rerank(f, query, docs, &o);
    if (again.scores.size() == 4) {
        CHECK(std::memcmp(again.scores.data(), top.scores.data(), 16) == 0);
    }
    // An embedding write on a reranker session is refused.
    const turbo_text qv = text_of(query);
    Err e;
    CHECK_EQ(vt()->session_write_text(f.session, &qv, 1, nullptr, e.p()), TURBO_E_UNSUPPORTED_TASK);
    // Left truncation of pairs is not offered and says so on field 2.
    o.truncate = TURBO_TRUNCATE_LEFT;
    Err e2;
    CHECK_EQ(vt()->session_write_pairs(f.session, &qv, &qv, 1, &o, e2.p()), TURBO_E_UNSUPPORTED_OPTION);
    CHECK_EQ(e2.e.field, 2);
}

struct Test {
    const char *name;
    void (*fn)();
};

const Test kTests[] = {
    {"the_device_is_an_apple_gpu_with_unified_memory", the_device_is_an_apple_gpu_with_unified_memory},
    {"capability_offers_embed_and_rerank_in_fp32", capability_offers_embed_and_rerank_in_fp32},
    {"can_run_checks_the_bundle_and_task", can_run_checks_the_bundle_and_task},
    {"struct_size_rules_are_enforced", struct_size_rules_are_enforced},
    {"unknown_option_enums_name_their_field", unknown_option_enums_name_their_field},
    {"model_info_puts_everything_but_the_tokenizer_on_the_gpu", model_info_puts_everything_but_the_tokenizer_on_the_gpu},
    {"session_wider_than_the_model_is_refused", session_wider_than_the_model_is_refused},
    {"embeddings_are_unit_norm_and_bitwise_repeatable", embeddings_are_unit_norm_and_bitwise_repeatable},
    {"normalize_pooling_and_output_dim_are_honored", normalize_pooling_and_output_dim_are_honored},
    {"truncation_none_over_budget_is_a_capacity_error", truncation_none_over_budget_is_a_capacity_error},
    {"token_rows_equal_text_rows", token_rows_equal_text_rows},
    {"other_tasks_are_refused_on_an_embedding_session", other_tasks_are_refused_on_an_embedding_session},
    {"model_options_are_checked", model_options_are_checked},
    {"buffers_are_shared_or_host", buffers_are_shared_or_host},
    {"reranker_scores_sort_and_activate", reranker_scores_sort_and_activate},
};

} // namespace

int main() {
    int n = 0;
    for (const Test &t : kTests) {
        std::printf("test %s\n", t.name);
        const int before = g_failures;
        t.fn();
        std::printf("  %s\n", g_failures == before ? "ok" : "FAILED");
        ++n;
    }
    std::printf("%d tests, %d failed checks\n", n, g_failures);
    return g_failures == 0 ? 0 : 1;
}
