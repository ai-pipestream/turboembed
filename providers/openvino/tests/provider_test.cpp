// SPDX-License-Identifier: Apache-2.0
//
// Vtable-level tests for the OpenVINO provider.
//
// These drive `turbo_provider_get` directly, which is the only way to reach
// the cases the core filters out before a provider sees them: a caller that
// declares a smaller `struct_size`, an option enumeration this build does
// not know, a `top_n` above the row count, and the token-classification rows
// where truncation cuts a word in half.
//
// The library under test is the one built next to this binary; set
// `TURBO_PROVIDER_LIB` to test another build (for example a baseline, to see
// a test fail). Bundles come from the environment, as in the Rust live
// tests: `TURBO_LIVE_BUNDLE` (embedder), `TURBO_LIVE_RERANK_BUNDLE`,
// `TURBO_LIVE_NER_BUNDLE`. A test whose bundle is not named skips and says
// so; the device is `TURBO_LIVE_ORDINAL` or the provider's CPU device.

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

/// An error record the provider fills in; `check_size` accepts it whole.
struct Err {
    turbo_error e{};
    Err() { e.struct_size = sizeof(turbo_error); }
    turbo_error *p() { return &e; }
    const char *what() const { return e.message; }
};

/// Report a call that was expected to succeed.
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

turbo_device_info device_info(uint32_t ordinal) {
    turbo_device_info info{};
    info.struct_size = sizeof(info);
    Err err;
    ok(vt()->device_info(vt()->state, ordinal, &info, err.p()), err, "device_info");
    return info;
}

/// The device under test: `TURBO_LIVE_ORDINAL`, else the CPU device.
uint32_t ordinal() {
    if (const char *v = env("TURBO_LIVE_ORDINAL")) {
        return static_cast<uint32_t>(std::atoi(v));
    }
    uint32_t n = 0;
    Err err;
    ok(vt()->device_count(vt()->state, &n, err.p()), err, "device_count");
    for (uint32_t i = 0; i < n; ++i) {
        if (device_info(i).kind == TURBO_DEVICE_CPU) {
            return i;
        }
    }
    return 0;
}

/// A context, model, and session held together so a test body can return early.
struct Fixture {
    void *ctx = nullptr;
    void *model = nullptr;
    void *session = nullptr;
    std::vector<std::string> labels;

    bool open(const char *bundle_var, uint32_t max_batch, uint32_t max_seq) {
        const char *dir = env(bundle_var);
        if (dir == nullptr) {
            std::printf("  skipped: %s is not set\n", bundle_var);
            return false;
        }
        Err err;
        if (!ok(vt()->context_create(vt()->state, ordinal(), nullptr, &ctx, err.p()), err, "context_create")) {
            return false;
        }
        turbo_model_desc md{};
        md.struct_size = sizeof(md);
        const std::string path(dir);
        if (!ok(vt()->model_load(ctx, text_of(path), &md, &model, err.p()), err, "model_load")) {
            return false;
        }
        turbo_model_info info{};
        info.struct_size = sizeof(info);
        if (!ok(vt()->model_info(model, &info, err.p()), err, "model_info")) {
            return false;
        }
        for (uint32_t i = 0; i < info.n_labels; ++i) {
            turbo_text t{};
            ok(vt()->model_label(model, i, &t, err.p()), err, "model_label");
            labels.emplace_back(t.ptr, t.len);
        }
        turbo_session_desc sd{};
        sd.struct_size = sizeof(sd);
        sd.max_batch = max_batch;
        sd.max_seq = max_seq;
        return ok(vt()->session_create(model, &sd, &session, err.p()), err, "session_create");
    }

    ~Fixture() {
        vt()->session_release(session);
        vt()->model_release(model);
        vt()->context_release(ctx);
    }
};

// ---------------------------------------------------------------------------
// Devices and buffers
// ---------------------------------------------------------------------------

/// The CPU vendor read back from cpuid, independently of the provider.
std::string host_cpu_vendor() {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int regs[4] = {0, 0, 0, 0};
    __asm__ __volatile__("cpuid" : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3]) : "a"(0), "c"(0));
    char id[13];
    std::memcpy(id + 0, &regs[1], 4);
    std::memcpy(id + 4, &regs[3], 4);
    std::memcpy(id + 8, &regs[2], 4);
    id[12] = '\0';
    return std::string(id);
#else
    return std::string();
#endif
}

void cpu_device_reports_the_host_cpu_vendor() {
    uint32_t n = 0;
    Err err;
    ok(vt()->device_count(vt()->state, &n, err.p()), err, "device_count");
    bool saw_cpu = false;
    for (uint32_t i = 0; i < n; ++i) {
        const turbo_device_info info = device_info(i);
        if (info.kind != TURBO_DEVICE_CPU) {
            continue;
        }
        saw_cpu = true;
        std::printf("  cpu device `%s` vendor `%s` id %#x\n", info.name, info.vendor, info.vendor_id);
        const std::string cpuid = host_cpu_vendor();
        if (cpuid == "GenuineIntel") {
            CHECK(std::string(info.vendor) == "Intel");
            CHECK_EQ(info.vendor_id, 0x8086);
        } else if (cpuid == "AuthenticAMD") {
            CHECK(std::string(info.vendor) == "AMD");
            CHECK_EQ(info.vendor_id, 0x1022);
        } else {
            // An unknown CPU is reported as unknown, never as a guess.
            CHECK_EQ(info.vendor_id, 0);
        }
    }
    CHECK(saw_cpu);
}

void buffer_alloc_keeps_the_callers_struct_size() {
    Err err;
    void *ctx = nullptr;
    if (!ok(vt()->context_create(vt()->state, ordinal(), nullptr, &ctx, err.p()), err, "context_create")) {
        return;
    }
    turbo_buffer_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.placement = TURBO_PLACE_HOST;
    desc.dtype = TURBO_DTYPE_F32;
    desc.ndim = 1;
    desc.shape[0] = 16;

    // A caller that only understands the struct through `host_ptr` declares
    // that size; everything past it must stay untouched.
    const uint32_t declared = offsetof(turbo_provider_buffer, desc);
    std::vector<unsigned char> raw(sizeof(turbo_provider_buffer) + 64, 0xAB);
    auto *out = reinterpret_cast<turbo_provider_buffer *>(raw.data());
    out->struct_size = declared;
    if (ok(vt()->buffer_alloc(ctx, &desc, out, err.p()), err, "buffer_alloc")) {
        CHECK_EQ(out->struct_size, declared);
        CHECK(out->handle != nullptr);
        CHECK(out->host_ptr != nullptr);
        for (size_t i = declared; i < raw.size(); ++i) {
            CHECK_EQ(raw[i], 0xAB);
        }
        vt()->buffer_release(out->handle);
    }
    vt()->context_release(ctx);
}

void host_pointer_import_aliases_caller_memory() {
    Err err;
    void *ctx = nullptr;
    if (!ok(vt()->context_create(vt()->state, ordinal(), nullptr, &ctx, err.p()), err, "context_create")) {
        return;
    }
    const bool advertised = (device_info(ordinal()).caps & TURBO_CAP_HOST_PTR_IMPORT) != 0;
    CHECK(advertised);
    CHECK(vt()->buffer_import != nullptr);
    if (!advertised || vt()->buffer_import == nullptr) {
        vt()->context_release(ctx);
        return;
    }
    std::vector<float> mine{1.0f, 2.0f, 3.0f, 4.0f};
    turbo_buffer_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.placement = TURBO_PLACE_HOST;
    desc.dtype = TURBO_DTYPE_F32;
    desc.ndim = 1;
    desc.shape[0] = mine.size();
    desc.bytes = mine.size() * sizeof(float);
    turbo_native_handle h{};
    h.struct_size = sizeof(h);
    h.kind = TURBO_HANDLE_HOST_PTR;
    h.handle = reinterpret_cast<uint64_t>(mine.data());
    turbo_provider_buffer out{};
    out.struct_size = sizeof(out);
    if (ok(vt()->buffer_import(ctx, &desc, &h, &out, err.p()), err, "buffer_import")) {
        // The import wraps the pointer: no copy is taken, so a later write
        // through the caller's own vector is what a read returns.
        CHECK_EQ(reinterpret_cast<uint64_t>(out.host_ptr), h.handle);
        mine[2] = 42.0f;
        std::vector<float> back(mine.size(), 0.0f);
        ok(vt()->buffer_read(out.handle, back.data(), desc.bytes, err.p()), err, "buffer_read");
        CHECK(back[2] == 42.0f);
        turbo_native_handle exported{};
        exported.struct_size = sizeof(exported);
        ok(vt()->buffer_export(out.handle, TURBO_HANDLE_HOST_PTR, &exported, err.p()), err, "buffer_export");
        CHECK_EQ(exported.handle, h.handle);
        vt()->buffer_release(out.handle);
    }
    // The caller keeps its memory: releasing the handle must not have freed
    // or poisoned it.
    CHECK(mine[2] == 42.0f);

    // A device pointer is not something this provider can adopt.
    turbo_native_handle cl{};
    cl.struct_size = sizeof(cl);
    cl.kind = TURBO_HANDLE_CL_MEM;
    cl.handle = h.handle;
    turbo_provider_buffer out2{};
    out2.struct_size = sizeof(out2);
    Err err2;
    CHECK_EQ(vt()->buffer_import(ctx, &desc, &cl, &out2, err2.p()), TURBO_E_UNSUPPORTED);
    vt()->context_release(ctx);
}

// ---------------------------------------------------------------------------
// Sessions: struct_size, enumerations, top_n
// ---------------------------------------------------------------------------

void session_run_keeps_the_callers_struct_size() {
    Fixture f;
    if (!f.open("TURBO_LIVE_BUNDLE", 1, 32)) {
        return;
    }
    Err err;
    turbo_embed_options eo{};
    eo.struct_size = sizeof(eo);
    const std::string text = "struct size check";
    const turbo_text t = text_of(text);
    if (!ok(vt()->session_write_text(f.session, &t, 1, &eo, err.p()), err, "write_text")) {
        return;
    }
    // A caller built against a header without `spans` declares the size up
    // to `outputs`; the provider must not tell it the struct is bigger.
    const uint32_t declared = offsetof(turbo_provider_result, n_spans);
    std::vector<unsigned char> raw(sizeof(turbo_provider_result) + 64, 0xCD);
    auto *out = reinterpret_cast<turbo_provider_result *>(raw.data());
    out->struct_size = declared;
    if (ok(vt()->session_run(f.session, nullptr, out, err.p()), err, "session_run")) {
        CHECK_EQ(out->struct_size, declared);
        CHECK_EQ(out->n_outputs, 1);
        for (size_t i = declared; i < raw.size(); ++i) {
            CHECK_EQ(raw[i], 0xCD);
        }
    }
}

void unknown_embed_enums_are_invalid_enum() {
    Fixture f;
    if (!f.open("TURBO_LIVE_BUNDLE", 1, 32)) {
        return;
    }
    const std::string text = "unknown enumerations";
    const turbo_text t = text_of(text);
    {
        turbo_embed_options eo{};
        eo.struct_size = sizeof(eo);
        eo.truncate = 99; // no such TURBO_TRUNCATE_*
        Err err;
        CHECK_EQ(vt()->session_write_text(f.session, &t, 1, &eo, err.p()), TURBO_E_INVALID_ENUM);
        CHECK_EQ(err.e.field, 2);
    }
    {
        turbo_embed_options eo{};
        eo.struct_size = sizeof(eo);
        eo.prompt_role = 7; // no such TURBO_PROMPT_*
        Err err;
        CHECK_EQ(vt()->session_write_text(f.session, &t, 1, &eo, err.p()), TURBO_E_INVALID_ENUM);
        CHECK_EQ(err.e.field, 4);
    }
}

void unknown_classify_enums_are_invalid_enum() {
    Fixture f;
    if (!f.open("TURBO_LIVE_NER_BUNDLE", 1, 32)) {
        return;
    }
    const std::string text = "Ada Lovelace";
    const turbo_text t = text_of(text);
    {
        turbo_classify_options co{};
        co.struct_size = sizeof(co);
        co.truncate = 42;
        Err err;
        CHECK_EQ(vt()->session_write_text_classify(f.session, &t, 1, &co, err.p()), TURBO_E_INVALID_ENUM);
        CHECK_EQ(err.e.field, 2);
    }
    {
        turbo_classify_options co{};
        co.struct_size = sizeof(co);
        co.aggregation = 11;
        Err err;
        CHECK_EQ(vt()->session_write_text_classify(f.session, &t, 1, &co, err.p()), TURBO_E_INVALID_ENUM);
        CHECK_EQ(err.e.field, 4);
    }
    {
        // The per-token softmax is fused into the graph, so raw logits are
        // rejected naming raw_scores (field 5), not silently activated.
        turbo_classify_options co{};
        co.struct_size = sizeof(co);
        co.raw_scores = 1;
        Err err;
        CHECK_EQ(vt()->session_write_text_classify(f.session, &t, 1, &co, err.p()), TURBO_E_UNSUPPORTED_OPTION);
        CHECK_EQ(err.e.field, 5);
    }
}

void rerank_options_are_checked_and_top_n_is_clamped() {
    Fixture f;
    if (!f.open("TURBO_LIVE_RERANK_BUNDLE", 4, 64)) {
        return;
    }
    const std::string query = "How many people live in Berlin?";
    const std::string d0 = "Berlin has 3.5 million registered inhabitants.";
    const std::string d1 = "The Eiffel Tower stands in Paris.";
    const std::string d2 = "New York City is the most populous city in the United States.";
    const turbo_text q = text_of(query);
    const turbo_text docs[3] = {text_of(d0), text_of(d1), text_of(d2)};
    {
        turbo_rerank_options ro{};
        ro.struct_size = sizeof(ro);
        ro.truncate = 5;
        Err err;
        CHECK_EQ(vt()->session_write_pairs(f.session, &q, docs, 3, &ro, err.p()), TURBO_E_INVALID_ENUM);
        CHECK_EQ(err.e.field, 2);
    }
    {
        // LEFT would drop the [CLS] and the query; the packer has no such
        // policy, so it is refused naming truncate.
        turbo_rerank_options ro{};
        ro.struct_size = sizeof(ro);
        ro.truncate = TURBO_TRUNCATE_LEFT;
        Err err;
        CHECK_EQ(vt()->session_write_pairs(f.session, &q, docs, 3, &ro, err.p()), TURBO_E_UNSUPPORTED_OPTION);
        CHECK_EQ(err.e.field, 2);
    }
    {
        // top_n above the row count is clamped to the rows that were sorted;
        // publishing 9 indices out of a 3-row sort would be stale memory.
        turbo_rerank_options ro{};
        ro.struct_size = sizeof(ro);
        ro.top_n = 9;
        ro.return_sorted = 1;
        Err err;
        if (!ok(vt()->session_write_pairs(f.session, &q, docs, 3, &ro, err.p()), err, "write_pairs")) {
            return;
        }
        turbo_provider_result r{};
        r.struct_size = sizeof(r);
        if (ok(vt()->session_run(f.session, nullptr, &r, err.p()), err, "session_run")) {
            CHECK_EQ(r.n_outputs, 2);
            if (r.n_outputs == 2) {
                CHECK_EQ(r.outputs[1].ndim, 1);
                CHECK_EQ(r.outputs[1].shape[0], 3);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Token classification: rows, truncation, and spans
// ---------------------------------------------------------------------------

/// Entity name of a label ("B-LOC" -> "LOC").
std::string entity_of(const std::string &label) {
    if (label.size() > 2 && label[1] == '-') {
        return label.substr(2);
    }
    return label;
}

struct Found {
    std::string word;
    std::string entity;
    float score;
    uint64_t start, end;
};

/// Run one token-classification text and return the spans as text/entity.
std::vector<Found> classify(Fixture &f, const std::string &text, uint32_t truncate, uint32_t aggregation,
                            uint32_t max_tokens, int32_t *status) {
    std::vector<Found> out;
    turbo_classify_options co{};
    co.struct_size = sizeof(co);
    co.truncate = truncate;
    co.aggregation = aggregation;
    co.max_tokens = max_tokens;
    const turbo_text t = text_of(text);
    Err err;
    *status = vt()->session_write_text_classify(f.session, &t, 1, &co, err.p());
    if (*status != TURBO_OK) {
        std::printf("  write_text_classify: status %d field %u: %s\n", *status, err.e.field, err.what());
        return out;
    }
    turbo_provider_result r{};
    r.struct_size = sizeof(r);
    *status = vt()->session_run(f.session, nullptr, &r, err.p());
    if (*status != TURBO_OK) {
        std::printf("  session_run: status %d field %u: %s\n", *status, err.e.field, err.what());
        return out;
    }
    for (uint32_t i = 0; i < r.n_spans; ++i) {
        const turbo_span &s = r.spans[i];
        out.push_back(Found{text.substr(s.byte_start, s.byte_end - s.byte_start), entity_of(f.labels[s.label]), s.score,
                            s.byte_start, s.byte_end});
    }
    return out;
}

bool has(const std::vector<Found> &spans, const char *word, const char *entity) {
    for (const Found &s : spans) {
        if (s.word == word && s.entity == entity) {
            return true;
        }
    }
    return false;
}

void print(const char *what, const std::vector<Found> &spans) {
    std::printf("  %s:", what);
    for (const Found &s : spans) {
        std::printf(" [%s=%s %.6f]", s.word.c_str(), s.entity.c_str(), static_cast<double>(s.score));
    }
    std::printf("\n");
}

// One whitespace-free word that WordPiece splits into 49 sub-tokens, so a
// budget landing inside it cuts the word in half. (WordPiece maps a word
// longer than 100 characters to a single [UNK], which would not straddle
// anything, so this one stays at 96.)
const std::string kLongWord(96, 'a');
// 45 single-token words, enough that the cut lands deep inside the long
// word: every word after it then sits far from the column it would have had
// if the cut word had been admitted with its full sub-token count.
const std::string kFiller = [] {
    std::string s;
    for (int i = 0; i < 45; ++i) {
        s += (i == 0 ? "" : " ");
        s += "fog";
    }
    return s;
}();
// 10 tokens, three entities.
const std::string kSentence = "Ada Lovelace visited Berlin with colleagues from Microsoft.";

void left_truncation_maps_words_to_their_columns() {
    Fixture f;
    if (!f.open("TURBO_LIVE_NER_BUNDLE", 1, 64)) {
        return;
    }
    // 104 tokens against a 62-token content budget: the cut falls inside the
    // leading word, so the row starts mid-word and every following word sits
    // 42 columns below its sub-token index in the full text.
    const std::string text = kLongWord + " " + kFiller + " " + kSentence;
    int32_t status = TURBO_OK;
    const std::vector<Found> spans = classify(f, text, TURBO_TRUNCATE_LEFT, TURBO_AGGREGATE_SIMPLE, 0, &status);
    CHECK_EQ(status, TURBO_OK);
    print("left truncated", spans);
    CHECK(has(spans, "Ada Lovelace", "PER"));
    CHECK(has(spans, "Berlin", "LOC"));
    CHECK(has(spans, "Microsoft", "ORG"));
    // The half-kept word is dropped, not clipped: no span reaches into it.
    for (const Found &s : spans) {
        CHECK(s.start >= kLongWord.size());
    }

}

void right_truncation_drops_the_word_that_straddles_the_budget() {
    Fixture f;
    if (!f.open("TURBO_LIVE_NER_BUNDLE", 1, 64)) {
        return;
    }
    // The trailing word starts inside the budget and ends outside it. MAX
    // aggregation reads every sub-token of a word, so admitting that word
    // with its full sub-token count is a read past the row.
    const std::string text = kSentence + " " + kFiller + " " + kLongWord;
    int32_t status = TURBO_OK;
    const std::vector<Found> spans = classify(f, text, TURBO_TRUNCATE_RIGHT, TURBO_AGGREGATE_MAX, 0, &status);
    CHECK_EQ(status, TURBO_OK);
    print("right truncated", spans);
    CHECK(has(spans, "Ada Lovelace", "PER"));
    CHECK(has(spans, "Berlin", "LOC"));
    // The half-kept word is dropped, not clipped: no span reaches into it.
    for (const Found &s : spans) {
        CHECK(s.end <= text.size() - kLongWord.size());
    }
}

void span_score_is_the_mean_of_the_word_scores() {
    Fixture f;
    if (!f.open("TURBO_LIVE_NER_BUNDLE", 1, 64)) {
        return;
    }
    // Invented names: the model is confident enough to tag them but not
    // equally confident per word, so the mean and the minimum of a group
    // differ by more than float noise.
    const std::string text = "Xylo Quirkenbaum joined Frobnitz Dynamics in Lower Slobbovia .";
    int32_t status = TURBO_OK;
    const std::vector<Found> words = classify(f, text, TURBO_TRUNCATE_MODEL, TURBO_AGGREGATE_NONE, 0, &status);
    CHECK_EQ(status, TURBO_OK);
    const std::vector<Found> groups = classify(f, text, TURBO_TRUNCATE_MODEL, TURBO_AGGREGATE_SIMPLE, 0, &status);
    CHECK_EQ(status, TURBO_OK);
    print("per word", words);
    print("grouped", groups);
    int discriminating = 0;
    for (const Found &g : groups) {
        float sum = 0.0f;
        float lowest = 2.0f;
        int n = 0;
        for (const Found &w : words) {
            if (w.start >= g.start && w.end <= g.end) {
                sum += w.score;
                lowest = w.score < lowest ? w.score : lowest;
                ++n;
            }
        }
        if (n == 0) {
            continue;
        }
        const float mean = sum / static_cast<float>(n);
        CHECK(std::fabs(g.score - mean) < 1e-5f);
        if (n > 1 && std::fabs(mean - lowest) > 1e-3f) {
            ++discriminating; // this group would read differently under std::min
        }
    }
    // Without a group whose words disagree, the check above would pass for
    // the minimum too.
    CHECK(discriminating > 0);
}

struct Test {
    const char *name;
    void (*fn)();
};

const Test kTests[] = {
    {"cpu_device_reports_the_host_cpu_vendor", cpu_device_reports_the_host_cpu_vendor},
    {"buffer_alloc_keeps_the_callers_struct_size", buffer_alloc_keeps_the_callers_struct_size},
    {"host_pointer_import_aliases_caller_memory", host_pointer_import_aliases_caller_memory},
    {"session_run_keeps_the_callers_struct_size", session_run_keeps_the_callers_struct_size},
    {"unknown_embed_enums_are_invalid_enum", unknown_embed_enums_are_invalid_enum},
    {"unknown_classify_enums_are_invalid_enum", unknown_classify_enums_are_invalid_enum},
    {"rerank_options_are_checked_and_top_n_is_clamped", rerank_options_are_checked_and_top_n_is_clamped},
    {"left_truncation_maps_words_to_their_columns", left_truncation_maps_words_to_their_columns},
    {"right_truncation_drops_the_word_that_straddles_the_budget", right_truncation_drops_the_word_that_straddles_the_budget},
    {"span_score_is_the_mean_of_the_word_scores", span_score_is_the_mean_of_the_word_scores},
};

} // namespace

int main(int argc, char **argv) {
    const char *filter = argc > 1 ? argv[1] : nullptr;
    int run = 0;
    for (const Test &t : kTests) {
        if (filter != nullptr && std::strstr(t.name, filter) == nullptr) {
            continue;
        }
        std::printf("test %s\n", t.name);
        const int before = g_failures;
        t.fn();
        ++run;
        std::printf("  %s\n", g_failures == before ? "ok" : "FAILED");
    }
    std::printf("\n%d tests, %d failed checks\n", run, g_failures);
    return g_failures == 0 ? 0 : 1;
}
