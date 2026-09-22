// SPDX-License-Identifier: Apache-2.0
//
// Vtable-level tests for the Hailo provider.
//
// These drive `turbo_provider_get` directly, which reaches the cases the
// core filters out before a provider sees them: a caller that declares a
// smaller `struct_size`, an option enumeration this build does not know, a
// session wider than the HEF frame, and token batches with token types the
// HEF cannot honor.
//
// The library under test is the one built next to this binary; set
// `TURBO_PROVIDER_LIB` to test another build. The embedding bundle comes
// from `TURBO_LIVE_BUNDLE` as in the Rust live tests. A bundle variable
// that is not set is a configuration error, not a reason to pass: a case
// that needs it fails naming the variable. The device is
// `TURBO_LIVE_ORDINAL` or ordinal 0. Without a Hailo device there is
// nothing to run at all, so the device tests report the provider's own
// reason and the bundle tests print `not applicable` and return.

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

/// True when a Hailo device is present; prints the reason and returns false
/// when there is none, which is the one legitimate reason to run nothing.
bool has_device() {
    if (device_count() == 0) {
        std::printf("  not applicable: no Hailo device\n");
        return false;
    }
    return true;
}

/// The bundle directory `var` names, or a counted failure and nullptr. The
/// provider library and the device are there, so a bundle the run was not
/// given is a configuration error the run must report, not skip.
const char *require_bundle(const char *var) {
    const char *dir = env(var);
    if (dir == nullptr) {
        ++g_failures;
        std::printf("  FAIL %s is not set; point it at a bundle directory for this case\n", var);
    }
    return dir;
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

    bool open(uint32_t max_batch, uint32_t max_seq, const turbo_kv *opts = nullptr, uint32_t n_opts = 0) {
        if (!has_device()) {
            return false;
        }
        const char *dir = require_bundle("TURBO_LIVE_BUNDLE");
        if (dir == nullptr) {
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
        CHECK_EQ(o.buffer.desc.placement, TURBO_PLACE_HOST);
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

void devices_are_hailo_npus() {
    const uint32_t n = device_count();
    if (n == 0) {
        std::printf("  not applicable: no Hailo device\n");
        return;
    }
    for (uint32_t i = 0; i < n; ++i) {
        const turbo_device_info info = device_info(i);
        std::printf("  device %u `%s` vendor `%s` runtime `%s` driver `%s` caps %#llx\n", i, info.name, info.vendor,
                    info.runtime_version, info.driver_version, static_cast<unsigned long long>(info.caps));
        CHECK_EQ(info.kind, TURBO_DEVICE_NPU);
        CHECK_EQ(info.ordinal, i);
        CHECK_EQ(info.vendor_id, 0x1e60);
        CHECK(std::string(info.vendor) == "Hailo");
        CHECK(std::string(info.provider_id) == "hailo");
        CHECK(std::string(info.runtime_version).rfind("HailoRT ", 0) == 0);
        // Results live in host memory; the provider must not claim otherwise.
        CHECK((info.caps & TURBO_CAP_DEVICE_RESULT) == 0);
        CHECK((info.caps & TURBO_CAP_DETERMINISTIC) != 0);
        CHECK((info.caps & TURBO_CAP_OPT_POOLING_OVERRIDE) != 0);
    }
    // One past the end is DEVICE_NOT_FOUND, not a crash.
    turbo_device_info info{};
    info.struct_size = sizeof(info);
    Err err;
    CHECK_EQ(vt()->device_info(vt()->state, n, &info, err.p()), TURBO_E_DEVICE_NOT_FOUND);
}

void capability_states_the_quantized_floor() {
    if (!has_device()) {
        return;
    }
    const uint32_t d = ordinal();
    const turbo_capability embed = capability(d, TURBO_TASK_EMBED, TURBO_MODALITY_TEXT);
    std::printf("  EMBED x TEXT: status %u dtype %u reference %u cosine_floor %.3f notes `%s`\n", embed.status,
                embed.dtype, embed.reference_dtype, static_cast<double>(embed.cosine_floor), embed.notes);
    // On a Hailo-8 the cell is SUPPORTED and names its receipts (conformance,
    // precision, matched benchmark); any other architecture is EXPERIMENTAL.
    if (std::string(embed.notes).find("Hailo-8;") != std::string::npos) {
        CHECK_EQ(embed.status, TURBO_CAP_SUPPORTED);
        CHECK(std::string(embed.notes).find("compare-hailo-pi5ai1-embed") != std::string::npos);
    } else {
        CHECK_EQ(embed.status, TURBO_CAP_EXPERIMENTAL);
    }
    CHECK_EQ(embed.dtype, TURBO_DTYPE_I8);
    CHECK_EQ(embed.reference_dtype, TURBO_DTYPE_F32);
    // The cell reports one number, so the check is that number and not a
    // range: kCosineFloorVsF32 in providers/hailo/src/provider.cpp is 0.30,
    // the conservative floor the INT8 HEF is documented to state. The live
    // suite gates on it and on the receipt's floor
    // (testdata/reference_embeddings/quantized_floors.json: hailo/I8 0.45,
    // measured_min 0.472835) and requires the cell's number to be no higher
    // than the receipt's, which 0.30 is.
    CHECK(embed.cosine_floor == 0.30f);
    CHECK_EQ(embed.deterministic, 1);
    for (uint32_t task : {TURBO_TASK_RERANK, TURBO_TASK_CLASSIFY, TURBO_TASK_TOKEN_CLASSIFY, TURBO_TASK_GENERATE}) {
        CHECK_EQ(capability(d, task, TURBO_MODALITY_TEXT).status, TURBO_CAP_UNSUPPORTED);
    }
    CHECK_EQ(capability(d, TURBO_TASK_EMBED, TURBO_MODALITY_IMAGE).status, TURBO_CAP_UNSUPPORTED);
}

void can_run_checks_the_bundle_and_task() {
    if (!has_device()) {
        return;
    }
    const char *dir = require_bundle("TURBO_LIVE_BUNDLE");
    if (dir == nullptr) {
        return;
    }
    const std::string path(dir);
    Err err;
    ok(vt()->can_run(vt()->state, ordinal(), text_of(path), TURBO_TASK_EMBED, TURBO_MODALITY_TEXT, err.p()), err,
       "can_run");
    Err e2;
    CHECK_EQ(vt()->can_run(vt()->state, ordinal(), text_of(path), TURBO_TASK_RERANK, TURBO_MODALITY_TEXT, e2.p()),
             TURBO_E_UNSUPPORTED_TASK);
    const std::string missing = "/nonexistent/bundle";
    Err e3;
    CHECK_EQ(vt()->can_run(vt()->state, ordinal(), text_of(missing), TURBO_TASK_EMBED, TURBO_MODALITY_TEXT, e3.p()),
             TURBO_E_BUNDLE_NOT_FOUND);
}

// ---------------------------------------------------------------------------
// Struct sizes and enumerations
// ---------------------------------------------------------------------------

void struct_size_rules_are_enforced() {
    if (!has_device()) {
        return;
    }
    // A smaller, older struct is filled up to its declared size.
    turbo_device_info small{};
    small.struct_size = 16;
    Err err;
    ok(vt()->device_info(vt()->state, ordinal(), &small, err.p()), err, "device_info (small)");
    CHECK_EQ(small.struct_size, 16);
    CHECK_EQ(small.kind, TURBO_DEVICE_NPU);
    CHECK_EQ(small.caps, 0); // beyond the declared size: untouched
    // A larger one is refused: the provider cannot fill fields it does not know.
    turbo_device_info big{};
    big.struct_size = sizeof(big) + 8;
    Err e2;
    CHECK_EQ(vt()->device_info(vt()->state, ordinal(), &big, e2.p()), TURBO_E_INVALID_STRUCT_SIZE);
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
    o.pooling = 99;
    Err e2;
    CHECK_EQ(vt()->session_write_text(f.session, &view, 1, &o, e2.p()), TURBO_E_INVALID_ENUM);
    CHECK_EQ(e2.e.field, 6);
    o.pooling = TURBO_POOLING_MODEL;
    o.output_dim = f.info.dim + 1;
    Err e3;
    CHECK_EQ(vt()->session_write_text(f.session, &view, 1, &o, e3.p()), TURBO_E_UNSUPPORTED_OPTION);
    CHECK_EQ(e3.e.field, 7);
    o.output_dim = 0;
    o.output_dtype = TURBO_OUTPUT_F16;
    Err e4;
    CHECK_EQ(vt()->session_write_text(f.session, &view, 1, &o, e4.p()), TURBO_E_UNSUPPORTED_OPTION);
    CHECK_EQ(e4.e.field, 8);
    o.output_dtype = 99;
    Err e5;
    CHECK_EQ(vt()->session_write_text(f.session, &view, 1, &o, e5.p()), TURBO_E_INVALID_ENUM);
    CHECK_EQ(e5.e.field, 8);
    o.output_dtype = TURBO_OUTPUT_F32;
    Err e6;
    ok(vt()->session_write_text(f.session, &view, 1, &o, e6.p()), e6, "session_write_text (output F32)");
}

// ---------------------------------------------------------------------------
// Models and sessions
// ---------------------------------------------------------------------------

void model_info_reports_the_split_pipeline() {
    Fixture f;
    if (!f.open(1, 32)) {
        return;
    }
    CHECK_EQ(f.info.task, TURBO_TASK_EMBED);
    CHECK_EQ(f.info.kind, TURBO_MODEL_EMBEDDING);
    CHECK_EQ(f.info.fully_accelerated, 0);
    CHECK_EQ(f.info.stage_placement[TURBO_STAGE_TOKENIZE], TURBO_STAGE_HOST);
    CHECK_EQ(f.info.stage_placement[TURBO_STAGE_ENCODE], TURBO_STAGE_DEVICE);
    CHECK_EQ(f.info.stage_placement[TURBO_STAGE_POOL], TURBO_STAGE_HOST);
    CHECK_EQ(f.info.stage_placement[TURBO_STAGE_POSTPROCESS], TURBO_STAGE_UNUSED);
    CHECK_EQ(f.info.dtype_used, TURBO_DTYPE_F32);
    CHECK(f.info.max_seq > 2);
    CHECK(std::string(f.info.provider_id) == "hailo");
    std::printf("  model `%s` dim %u max_seq %u max_batch %u\n", f.info.model_id, f.info.dim, f.info.max_seq,
                f.info.max_batch);
}

void session_wider_than_the_frame_is_refused() {
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
}

void embeddings_are_unit_norm_and_bitwise_repeatable() {
    Fixture f;
    if (!f.open(2, 64)) {
        return;
    }
    const std::vector<std::string> texts = {"a brown dog runs through the grass", "the stock market closed higher"};
    const auto first = f.embed(texts, nullptr);
    const auto second = f.embed(texts, nullptr);
    CHECK_EQ(first.size(), 2);
    CHECK_EQ(second.size(), 2);
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
    // Unrelated sentences must stay far apart. The measurement on pi5ai1
    // (Hailo-8, HailoRT 4.23.0) is 0.3599, so the check holds it to about
    // twice that rather than to a cosine any two vectors would pass. The
    // threshold is well above an FP32 encoder's: the same pair measures
    // -0.0188 on the Metal provider, and INT8 activations pull unrelated
    // rows together.
    CHECK(c < 0.72f);
    // Single-row runs equal the batch rows.
    const auto solo = f.embed({texts[1]}, nullptr);
    CHECK_EQ(solo.size(), 1);
    if (solo.size() == 1) {
        CHECK(cosine(solo[0], first[1]) > 0.9999f);
    }
    turbo_session_stats st{};
    st.struct_size = sizeof(st);
    Err err;
    ok(vt()->session_stats(f.session, &st, err.p()), err, "session_stats");
    CHECK_EQ(st.runs, 3);
    CHECK(st.h2d_bytes > 0);
    CHECK(st.d2h_bytes > 0);
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
    CHECK_EQ(raw.size(), 1);
    if (raw.size() == 1) {
        CHECK(std::fabs(norm(raw[0]) - 1.0f) > 1e-3f);
    }
    o.normalize = TURBO_NORMALIZE_MODEL;
    o.pooling = TURBO_POOLING_CLS;
    const auto cls = f.embed(texts, &o);
    o.pooling = TURBO_POOLING_MEAN;
    const auto mean = f.embed(texts, &o);
    CHECK_EQ(cls.size(), 1);
    CHECK_EQ(mean.size(), 1);
    if (cls.size() == 1 && mean.size() == 1) {
        CHECK(cosine(cls[0], mean[0]) < 0.9999f);
    }
    o.pooling = TURBO_POOLING_MODEL;
    o.output_dim = 128;
    turbo_text view = text_of(texts[0]);
    Err err;
    ok(vt()->session_write_text(f.session, &view, 1, &o, err.p()), err, "session_write_text (output_dim)");
    turbo_provider_result r{};
    r.struct_size = sizeof(r);
    if (ok(vt()->session_run(f.session, nullptr, &r, err.p()), err, "session_run (output_dim)")) {
        CHECK_EQ(r.outputs[0].shape[1], 128);
        const auto *p = static_cast<const float *>(r.outputs[0].buffer.host_ptr);
        std::vector<float> v(p, p + 128);
        CHECK(std::fabs(norm(v) - 1.0f) < 1e-4f);
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
    CHECK_EQ(right.size(), 1);
    CHECK_EQ(left.size(), 1);
    if (right.size() != 1 || left.size() != 1) {
        return;
    }
    CHECK(std::memcmp(right[0].data(), left[0].data(), 4 * right[0].size()) != 0);
    // The positive form: the window that survived is exactly the head
    // (RIGHT) or the tail (LEFT) of the text. Ten single-token words into
    // this 8-column session leave six after [CLS] and [SEP], which is how
    // `live_truncation_policy_is_enforced` in
    // crates/turbo-conformance/tests/live_embed.rs derives the kept text.
    // An INT8 encoder is compared by cosine: the truncated row and the kept
    // text travel through different frames and need not be bitwise equal.
    const std::string kept_head = "one two three four five six";
    const std::string kept_tail = "five six seven eight nine ten";
    o.truncate = TURBO_TRUNCATE_MODEL;
    const auto head_alone = f.embed({kept_head}, &o);
    const auto tail_alone = f.embed({kept_tail}, &o);
    CHECK_EQ(head_alone.size(), 1);
    CHECK_EQ(tail_alone.size(), 1);
    if (head_alone.size() == 1 && tail_alone.size() == 1) {
        const float c_right = cosine(right[0], head_alone[0]);
        const float c_left = cosine(left[0], tail_alone[0]);
        std::printf("  truncation: right vs `%s` = %.6f, left vs `%s` = %.6f\n", kept_head.c_str(),
                    static_cast<double>(c_right), kept_tail.c_str(), static_cast<double>(c_left));
        // Measured 1.000000 both ways on pi5ai1: the kept window tokenizes
        // to the same frame as the text on its own, so even INT8 lands on
        // the same vector.
        CHECK(c_right > 0.9999f);
        CHECK(c_left > 0.9999f);
    }
}

void token_types_other_than_zero_are_refused() {
    Fixture f;
    if (!f.open(1, 8)) {
        return;
    }
    int32_t ids[8] = {101, 7592, 102, 0, 0, 0, 0, 0};
    int32_t mask[8] = {1, 1, 1, 0, 0, 0, 0, 0};
    int32_t types[8] = {0, 1, 0, 0, 0, 0, 0, 0};
    turbo_token_batch b{};
    b.struct_size = sizeof(b);
    b.batch = 1;
    b.seq = 8;
    b.ids = ids;
    b.mask = mask;
    b.types = types;
    Err err;
    CHECK_EQ(vt()->session_write_tokens(f.session, &b, err.p()), TURBO_E_UNSUPPORTED);
    b.types = nullptr;
    Err e2;
    ok(vt()->session_write_tokens(f.session, &b, e2.p()), e2, "session_write_tokens");
    turbo_provider_result r{};
    r.struct_size = sizeof(r);
    Err e3;
    // A run that fails here fails the case: `ok` counts it, and the checks
    // below are the point of the case, so the case cannot end quietly.
    const bool ran = ok(vt()->session_run(f.session, nullptr, &r, e3.p()), e3, "session_run (tokens)");
    if (ran) {
        const auto *p = static_cast<const float *>(r.outputs[0].buffer.host_ptr);
        std::vector<float> v(p, p + f.info.dim);
        CHECK(std::fabs(norm(v) - 1.0f) < 1e-4f);
        // The same tokens through write_text ([CLS] hello [SEP]) give the same vector.
        const auto via_text = f.embed({"hello"}, nullptr);
        CHECK_EQ(via_text.size(), 1);
        if (via_text.size() == 1) {
            CHECK(cosine(v, via_text[0]) > 0.9999f);
        }
    }
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
    if (!has_device()) {
        return;
    }
    const char *dir = require_bundle("TURBO_LIVE_BUNDLE");
    if (dir == nullptr) {
        return;
    }
    const std::string key = "front_end", bad = "bogus", unknown = "threads", one = "1";
    {
        turbo_kv kv{text_of(key), text_of(bad)};
        Fixture f;
        Err err;
        ok(vt()->context_create(vt()->state, ordinal(), nullptr, &f.ctx, err.p()), err, "context_create");
        turbo_model_desc md{};
        md.struct_size = sizeof(md);
        md.options = &kv;
        md.n_options = 1;
        const std::string path(dir);
        Err e2;
        CHECK_EQ(vt()->model_load(f.ctx, text_of(path), &md, &f.model, e2.p()), TURBO_E_INVALID_ARGUMENT);
        CHECK(f.model == nullptr);
        turbo_kv kv2{text_of(unknown), text_of(one)};
        md.options = &kv2;
        Err e3;
        CHECK_EQ(vt()->model_load(f.ctx, text_of(path), &md, &f.model, e3.p()), TURBO_E_INVALID_ARGUMENT);
    }
}

void buffers_are_host_only() {
    if (!has_device()) {
        return;
    }
    void *ctx = nullptr;
    Err err;
    if (!ok(vt()->context_create(vt()->state, ordinal(), nullptr, &ctx, err.p()), err, "context_create")) {
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
    d.placement = TURBO_PLACE_HOST;
    Err e3;
    if (ok(vt()->buffer_alloc(ctx, &d, &b, e3.p()), e3, "buffer_alloc (host)")) {
        CHECK_EQ(b.desc.bytes, 64);
        CHECK(b.host_ptr != nullptr);
        float back[16];
        Err e4;
        ok(vt()->buffer_read(b.handle, back, sizeof(back), e4.p()), e4, "buffer_read");
        Err e5;
        CHECK_EQ(vt()->buffer_read(b.handle, back, 32, e5.p()), TURBO_E_CAPACITY);
        turbo_native_handle h{};
        h.struct_size = sizeof(h);
        Err e6;
        ok(vt()->buffer_export(b.handle, TURBO_HANDLE_HOST_PTR, &h, e6.p()), e6, "buffer_export");
        CHECK(reinterpret_cast<void *>(h.handle) == b.host_ptr);
        vt()->buffer_release(b.handle);
    }
    vt()->context_release(ctx);
}

struct Test {
    const char *name;
    void (*fn)();
};

const Test kTests[] = {
    {"devices_are_hailo_npus", devices_are_hailo_npus},
    {"capability_states_the_quantized_floor", capability_states_the_quantized_floor},
    {"can_run_checks_the_bundle_and_task", can_run_checks_the_bundle_and_task},
    {"struct_size_rules_are_enforced", struct_size_rules_are_enforced},
    {"unknown_option_enums_name_their_field", unknown_option_enums_name_their_field},
    {"model_info_reports_the_split_pipeline", model_info_reports_the_split_pipeline},
    {"session_wider_than_the_frame_is_refused", session_wider_than_the_frame_is_refused},
    {"embeddings_are_unit_norm_and_bitwise_repeatable", embeddings_are_unit_norm_and_bitwise_repeatable},
    {"normalize_pooling_and_output_dim_are_honored", normalize_pooling_and_output_dim_are_honored},
    {"truncation_none_over_budget_is_a_capacity_error", truncation_none_over_budget_is_a_capacity_error},
    {"token_types_other_than_zero_are_refused", token_types_other_than_zero_are_refused},
    {"other_tasks_are_refused_on_an_embedding_session", other_tasks_are_refused_on_an_embedding_session},
    {"model_options_are_checked", model_options_are_checked},
    {"buffers_are_host_only", buffers_are_host_only},
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
