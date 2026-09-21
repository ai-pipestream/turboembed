// SPDX-License-Identifier: Apache-2.0
//
// TurboRerank edge-suite: argument validation, determinism, truncation
// boundaries, engine-local error storage, row layout, and the reranker.hpp
// C++ layer (row_at / pack_pair_ids) allocation contract. CPU-only binary —
// mirrors the turborerank-tests-nocuda build (no CUDA / OV / Metal defines).
// Complements turborerank_tests.cpp; no case is duplicated.

#include "internal.hpp"
#include "reranker.hpp"
#include "turbo_buffer.h"
#include "turborerank.h"
#include "wordpiece.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

static int g_fails = 0;
static int g_passes = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_fails;                                                           \
        } else {                                                                 \
            ++g_passes;                                                           \
        }                                                                        \
    } while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_ST(st) CHECK((st) == TURBORERANK_OK)

static std::string workspace_root() {
    if (const char *e = std::getenv("INFERSTREAM_ROOT")) {
        return e;
    }
    if (const char *e = std::getenv("TURBORERANK_WORKSPACE_ROOT")) {
        return e;
    }
    return ".";
}

static std::string model_dir() {
    if (const char *e = std::getenv("TURBORERANK_MODEL_DIR")) {
        return e;
    }
    return workspace_root() + "/models/rerank/ms-marco-minilm-l6";
}

static bool file_exists(const std::string &p) {
    std::ifstream in(p);
    return static_cast<bool>(in);
}

static bool weights_present() {
    return file_exists(model_dir() + "/model.safetensors") &&
           file_exists(model_dir() + "/tokenizer.json");
}

static bool almost(float a, float b, float eps) {
    return std::fabs(a - b) <= eps;
}

/** Create a CPU engine with the catalog CE loaded, or return false (skipped). */
static bool load_cpu_engine(turborerank_engine **out) {
    if (!weights_present()) {
        std::fprintf(
            stderr,
            "SKIP live-engine edge case (weights missing; make fetch-rerankers)\n"
        );
        return false;
    }
    turborerank_engine *e = nullptr;
    const std::string dir = model_dir();
    CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_CPU, dir.c_str(), &e));
    if (e == nullptr) {
        return false;
    }
    const turborerank_status st = turborerank_load_model(e, "ms-marco-minilm-l6", 0);
    CHECK_ST(st);
    if (st != TURBORERANK_OK) {
        std::fprintf(stderr, "load error: %s\n", turborerank_last_error(e));
        turborerank_engine_destroy(e);
        return false;
    }
    *out = e;
    return true;
}

// score() with n_documents == 0 must fail loudly, not succeed vacuously.
static void test_score_rejects_zero_documents() {
    turborerank_engine *e = nullptr;
    if (!load_cpu_engine(&e)) {
        return;
    }
    const char *q = "What is the capital of Norway?";
    turborerank_str query{q, std::strlen(q)};
    float out = -1.0f;
    const turborerank_status st =
        turborerank_score(e, nullptr, 0, query, nullptr, 0, nullptr, &out);
    CHECK(st == TURBORERANK_ERR_INVALID_ARGUMENT);
    CHECK(std::strstr(turborerank_last_error(e), "documents") != nullptr);
    CHECK_EQ(out, -1.0f); // untouched on the error path
    turborerank_engine_destroy(e);
}

// Both sides empty is rejected at pack_text; an empty *query* side alone is
// a valid pair and must score.
static void test_score_empty_query_and_doc() {
    turborerank_engine *e = nullptr;
    if (!load_cpu_engine(&e)) {
        return;
    }
    turborerank_str empty_q{nullptr, 0};
    turborerank_str empty_d{nullptr, 0};
    float out = 0;
    const turborerank_status both =
        turborerank_score(e, nullptr, 0, empty_q, &empty_d, 1, nullptr, &out);
    CHECK(both == TURBORERANK_ERR_INVALID_ARGUMENT);
    CHECK(std::strstr(turborerank_last_error(e), "empty") != nullptr);

    // pack_text direct: same rejection, error is engine-local.
    const turborerank_status pack =
        turborerank_pack_text(e, e->work, 0, empty_q, empty_d,
                              TURBORERANK_TRUNC_LONGEST_FIRST, 0);
    CHECK(pack == TURBORERANK_ERR_INVALID_ARGUMENT);

    // Empty query with a real document is a legal pair (scores in (0,1)).
    const char *doc = "Oslo is the capital and most populous city of Norway.";
    turborerank_str d{doc, std::strlen(doc)};
    turborerank_score_options opts{};
    opts.truncation = TURBORERANK_TRUNC_LONGEST_FIRST;
    opts.activation = TURBORERANK_ACT_SIGMOID;
    opts.max_length = 512;
    const turborerank_status one =
        turborerank_score(e, nullptr, 0, empty_q, &d, 1, &opts, &out);
    CHECK_ST(one);
    CHECK(out > 0.0f && out < 1.0f);

    // Non-NULL pointer with zero length is the same empty view.
    const char *blank = "";
    turborerank_str blank_q{blank, 0};
    const turborerank_status two =
        turborerank_score(e, nullptr, 0, blank_q, &d, 1, &opts, &out);
    CHECK_ST(two);
    CHECK(out > 0.0f && out < 1.0f);
    turborerank_engine_destroy(e);
}

// Two consecutive score() runs must be bitwise identical on CPU.
static void test_score_is_bitwise_deterministic() {
    turborerank_engine *e = nullptr;
    if (!load_cpu_engine(&e)) {
        return;
    }
    const char *q = "How do I reset my password?";
    turborerank_str query{q, std::strlen(q)};
    const char *texts[4] = {
        "Open settings, then Security, then choose Reset password.",
        "The recipe calls for two cups of flour and one egg.",
        "Password reset: click your profile picture, then Account, then Reset.",
        "Aston Villa won the European Cup in 1982.",
    };
    turborerank_str docs[4];
    for (int i = 0; i < 4; ++i) {
        docs[i] = {texts[i], std::strlen(texts[i])};
    }
    turborerank_score_options opts{};
    opts.truncation = TURBORERANK_TRUNC_LONGEST_FIRST;
    opts.activation = TURBORERANK_ACT_IDENTITY;
    opts.max_length = 512;
    float a[4] = {0, 0, 0, 0};
    float b[4] = {0, 0, 0, 0};
    CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 4, &opts, a));
    CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 4, &opts, b));
    CHECK_EQ(std::memcmp(a, b, sizeof(a)), 0);

    // Same through the explicit pack + forward path.
    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 4, 64, &buf));
    float c[4] = {0, 0, 0, 0};
    float d2[4] = {0, 0, 0, 0};
    for (int round = 0; round < 2; ++round) {
        for (uint32_t i = 0; i < 4; ++i) {
            CHECK_ST(turborerank_pack_text(
                e, buf, i, query, docs[i], TURBORERANK_TRUNC_LONGEST_FIRST, 64
            ));
        }
        float *dst = round == 0 ? c : d2;
        CHECK_ST(turborerank_forward(e, buf, 4, TURBORERANK_ACT_IDENTITY, dst));
    }
    CHECK_EQ(std::memcmp(c, d2, sizeof(c)), 0);
    CHECK_EQ(std::memcmp(a, c, sizeof(a)), 0); // convenience == explicit path
    turborerank_buffer_free(buf);
    turborerank_engine_destroy(e);
}

// Default opts (NULL) mean sigmoid; every output must lie strictly in (0,1)
// and match the header's sigmoid() reference.
static void test_sigmoid_scores_bounded() {
    turborerank_engine *e = nullptr;
    if (!load_cpu_engine(&e)) {
        return;
    }
    const char *q = "What year did the Apollo 11 mission land on the Moon?";
    turborerank_str query{q, std::strlen(q)};
    const char *texts[3] = {
        "Apollo 11 landed on the Moon on July 20, 1969.",
        "The Great Barrier Reef is the world's largest coral reef system.",
        "Neil Armstrong and Buzz Aldrin walked on the lunar surface.",
    };
    turborerank_str docs[3];
    for (int i = 0; i < 3; ++i) {
        docs[i] = {texts[i], std::strlen(texts[i])};
    }
    float sig[3] = {0, 0, 0};
    CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 3, nullptr, sig));
    for (int i = 0; i < 3; ++i) {
        CHECK(sig[i] > 0.0f && sig[i] < 1.0f);
    }
    turborerank_score_options opts{};
    opts.truncation = TURBORERANK_TRUNC_LONGEST_FIRST;
    opts.activation = TURBORERANK_ACT_IDENTITY;
    opts.max_length = 512;
    float logits[3] = {0, 0, 0};
    CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 3, &opts, logits));
    for (int i = 0; i < 3; ++i) {
        CHECK(almost(sig[i], turborerank::sigmoid(logits[i]), 1e-6f));
    }
    turborerank_engine_destroy(e);
}

// Sigmoid is strictly increasing on moderate logits: the rank order of the
// identity logits must be preserved exactly by the sigmoid outputs.
static void test_identity_ordering_preserved_by_sigmoid() {
    turborerank_engine *e = nullptr;
    if (!load_cpu_engine(&e)) {
        return;
    }
    const char *q = "Which planet is known as the Red Planet?";
    turborerank_str query{q, std::strlen(q)};
    const char *texts[5] = {
        "Mars is called the Red Planet because of its iron oxide dust.",
        "Jupiter is the largest planet in the Solar System.",
        "Mars hosts Olympus Mons, the tallest volcano known.",
        "Photosynthesis converts sunlight into chemical energy.",
        "Venus is the hottest planet in the Solar System.",
    };
    turborerank_str docs[5];
    for (int i = 0; i < 5; ++i) {
        docs[i] = {texts[i], std::strlen(texts[i])};
    }
    turborerank_score_options opts{};
    opts.truncation = TURBORERANK_TRUNC_LONGEST_FIRST;
    opts.max_length = 512;
    float logits[5] = {0, 0, 0, 0, 0};
    opts.activation = TURBORERANK_ACT_IDENTITY;
    CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 5, &opts, logits));
    float sig[5] = {0, 0, 0, 0, 0};
    opts.activation = TURBORERANK_ACT_SIGMOID;
    CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 5, &opts, sig));
    for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) {
            if (logits[i] == logits[j]) {
                CHECK_EQ(sig[i], sig[j]);
            } else if (logits[i] < logits[j]) {
                // Strict once both logits are clear of float saturation.
                if (logits[i] > -14.0f && logits[j] < 14.0f) {
                    CHECK(sig[i] < sig[j]);
                } else {
                    CHECK(sig[i] <= sig[j]);
                }
            } else {
                if (logits[j] > -14.0f && logits[i] < 14.0f) {
                    CHECK(sig[i] > sig[j]);
                } else {
                    CHECK(sig[i] >= sig[j]);
                }
            }
        }
    }
    turborerank_engine_destroy(e);
}

// Longest-first with the QUERY as the longer side: truncation starts from
// the query counts, keeps prefixes, and still lands exactly on budget.
static void test_truncation_longest_first_query_heavy() {
    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 1, 16, &buf));
    const int32_t q[7] = {1, 2, 3, 4, 5, 6, 7};
    const int32_t d[5] = {8, 9, 10, 11, 12};
    CHECK_ST(turborerank_pack_ids(
        buf, 0, q, 7, d, 5, TURBORERANK_TRUNC_LONGEST_FIRST, 8
    ));
    // budget = 8 - 3 = 5 → nq=2, nd=3 (alternating from the longer side).
    const int32_t want_ids[8] = {101, 1, 2, 102, 8, 9, 10, 102};
    const int32_t want_types[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    CHECK_EQ(std::memcmp(buf->input_ids, want_ids, sizeof(want_ids)), 0);
    CHECK_EQ(std::memcmp(buf->token_type_ids, want_types, sizeof(want_types)), 0);
    for (int i = 0; i < 8; ++i) {
        CHECK_EQ(buf->attention_mask[i], 1);
        CHECK_EQ(buf->position_ids[i], i);
    }
    CHECK_EQ(buf->input_ids[8], 0);
    CHECK_EQ(buf->attention_mask[8], 0);

    // max_length = 3 → specials only, both sides fully truncated away.
    CHECK_ST(turborerank_pack_ids(
        buf, 0, q, 7, d, 5, TURBORERANK_TRUNC_LONGEST_FIRST, 3
    ));
    CHECK_EQ(buf->input_ids[0], 101);
    CHECK_EQ(buf->input_ids[1], 102);
    CHECK_EQ(buf->input_ids[2], 102);
    CHECK_EQ(buf->attention_mask[3], 0);
    turborerank_buffer_free(buf);
}

// Over-long TEXT pairs through the engine vocab: TRUNC_ERROR fails loud,
// LONGEST_FIRST / QUERY_PRIORITY pack exactly max_length tokens, and
// query-priority preserves the query prefix of the untruncated pack.
static void test_truncation_modes_overlong_text_pair() {
    turborerank_engine *e = nullptr;
    if (!load_cpu_engine(&e)) {
        return;
    }
    const char *q = "alpha beta gamma delta epsilon";
    const char *doc =
        "zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma";
    turborerank_str query{q, std::strlen(q)};
    turborerank_str docv{doc, std::strlen(doc)};

    turborerank_buffer *wide = nullptr;
    turborerank_buffer *narrow = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 2, 512, &wide));
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 2, 16, &narrow));

    // Untruncated reference pack: [CLS] q… [SEP] d… [SEP] at max_length 512.
    CHECK_ST(turborerank_pack_text(
        e, wide, 0, query, docv, TURBORERANK_TRUNC_LONGEST_FIRST, 512
    ));
    // Locate the query/doc boundary in the reference row.
    uint32_t ref_nq = 0;
    while (ref_nq < 512 && wide->input_ids[1 + ref_nq] != 102) {
        ++ref_nq;
    }
    CHECK(ref_nq < 512); // found the [SEP] after the query

    const turborerank_status err = turborerank_pack_text(
        e, narrow, 0, query, docv, TURBORERANK_TRUNC_ERROR, 8
    );
    CHECK(err == TURBORERANK_ERR_INVALID_ARGUMENT);
    CHECK(std::strstr(turborerank_last_error(e), "max_length") != nullptr);

    CHECK_ST(turborerank_pack_text(
        e, narrow, 0, query, docv, TURBORERANK_TRUNC_LONGEST_FIRST, 8
    ));
    CHECK_EQ(narrow->input_ids[0], 101);
    CHECK_EQ(narrow->input_ids[7], 102);
    uint32_t lf_ones = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        lf_ones += static_cast<uint32_t>(narrow->attention_mask[i] == 1);
    }
    CHECK_EQ(lf_ones, 8u);

    CHECK_ST(turborerank_pack_text(
        e, narrow, 1, query, docv, TURBORERANK_TRUNC_QUERY_PRIORITY, 8
    ));
    const size_t off = static_cast<size_t>(1) * narrow->row_stride;
    uint32_t qp_ones = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        qp_ones += static_cast<uint32_t>(narrow->attention_mask[off + i] == 1);
    }
    CHECK_EQ(qp_ones, 8u);
    CHECK_EQ(narrow->input_ids[off + 0], 101);
    CHECK_EQ(narrow->input_ids[off + 7], 102);
    // Query priority keeps the leading query tokens of the reference pack.
    // budget = 5; whatever the vocab produced, the first 5 ids must match.
    for (uint32_t i = 0; i < 5 && i < ref_nq; ++i) {
        CHECK_EQ(narrow->input_ids[off + 1 + i], wide->input_ids[1 + i]);
        CHECK_EQ(narrow->token_type_ids[off + 1 + i], 0);
    }
    // The query segment of the query-priority row carries type id 0 and the
    // tail after the second [SEP] is padding.
    CHECK_EQ(narrow->attention_mask[off + 7], 1);
    CHECK_EQ(narrow->attention_mask[off + 8], 0);

    turborerank_buffer_free(wide);
    turborerank_buffer_free(narrow);
    turborerank_engine_destroy(e);
}

// max_length at seq-1 / seq / seq+1 against a fixed-capacity buffer: seq-1
// truncates by one, seq and seq+1 (clamped to capacity) are identical rows.
static void test_max_length_boundary_seq_minus_plus() {
    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 4, 16, &buf));
    const int32_t q[5] = {1, 2, 3, 4, 5};
    const int32_t d[10] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    CHECK_ST(turborerank_pack_ids(
        buf, 0, q, 5, d, 10, TURBORERANK_TRUNC_LONGEST_FIRST, 15
    ));
    CHECK_ST(turborerank_pack_ids(
        buf, 1, q, 5, d, 10, TURBORERANK_TRUNC_LONGEST_FIRST, 16
    ));
    CHECK_ST(turborerank_pack_ids(
        buf, 2, q, 5, d, 10, TURBORERANK_TRUNC_LONGEST_FIRST, 17
    ));
    CHECK_ST(turborerank_pack_ids(
        buf, 3, q, 5, d, 10, TURBORERANK_TRUNC_LONGEST_FIRST, 0
    ));

    uint32_t ones0 = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        ones0 += static_cast<uint32_t>(buf->attention_mask[i] == 1);
    }
    CHECK_EQ(ones0, 15u); // budget 12 → packed 15
    for (uint32_t r = 1; r <= 3; ++r) {
        const size_t off = static_cast<size_t>(r) * buf->row_stride;
        uint32_t ones = 0;
        for (uint32_t i = 0; i < 16; ++i) {
            ones += static_cast<uint32_t>(buf->attention_mask[off + i] == 1);
        }
        CHECK_EQ(ones, 16u); // budget 13 → packed 16 (full row)
    }
    // seq+1 clamps to the 16-wide capacity → row identical to the seq case;
    // max_length 0 means "default", clamped to the same capacity.
    for (uint32_t r = 2; r <= 3; ++r) {
        const size_t off = static_cast<size_t>(r) * buf->row_stride;
        const size_t base = static_cast<size_t>(1) * buf->row_stride;
        CHECK_EQ(std::memcmp(buf->input_ids + off, buf->input_ids + base,
                             16 * sizeof(int32_t)),
                 0);
        CHECK_EQ(std::memcmp(buf->token_type_ids + off, buf->token_type_ids + base,
                             16 * sizeof(int32_t)),
                 0);
        CHECK_EQ(std::memcmp(buf->attention_mask + off, buf->attention_mask + base,
                             16 * sizeof(int32_t)),
                 0);
        CHECK_EQ(std::memcmp(buf->position_ids + off, buf->position_ids + base,
                             16 * sizeof(int32_t)),
                 0);
    }
    // Row 0 (seq-1) must actually differ from the full row.
    CHECK_EQ(std::memcmp(buf->input_ids, buf->input_ids + buf->row_stride,
                         16 * sizeof(int32_t)) == 0,
             false);
    turborerank_buffer_free(buf);
}

// forward() rejects n_rows == 0 and n_rows > batch with a clear engine error.
static void test_forward_rejects_zero_and_over_batch_rows() {
    turborerank_engine *e = nullptr;
    if (!load_cpu_engine(&e)) {
        return;
    }
    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 2, 16, &buf));
    const int32_t q[1] = {7592};
    const int32_t d[1] = {2088};
    CHECK_ST(turborerank_pack_ids(
        buf, 0, q, 1, d, 1, TURBORERANK_TRUNC_LONGEST_FIRST, 16
    ));
    float score = 0;
    const turborerank_status zero =
        turborerank_forward(e, buf, 0, TURBORERANK_ACT_SIGMOID, &score);
    CHECK(zero == TURBORERANK_ERR_INVALID_ARGUMENT);
    CHECK(std::strstr(turborerank_last_error(e), "n_rows") != nullptr);
    const turborerank_status over =
        turborerank_forward(e, buf, 3, TURBORERANK_ACT_SIGMOID, &score);
    CHECK(over == TURBORERANK_ERR_INVALID_ARGUMENT);
    // The valid call right after still works. Note: a successful forward
    // does not clear last_error (only a successful load_model does), so the
    // previous "n_rows" message persists — assert current behavior.
    CHECK_ST(turborerank_forward(e, buf, 1, TURBORERANK_ACT_SIGMOID, &score));
    CHECK(score > 0.0f && score < 1.0f);
    CHECK(std::strstr(turborerank_last_error(e), "n_rows") != nullptr);
    turborerank_buffer_free(buf);
    turborerank_engine_destroy(e);
}

// Errors live on the engine that produced them: a failed load on engine A
// leaves engine B (and the thread-local create error) untouched, and a
// success clears the engine's own message.
static void test_last_error_is_engine_local() {
    turborerank_engine *a = nullptr;
    turborerank_engine *b = nullptr;
    CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_CPU, nullptr, &a));
    CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_CPU, nullptr, &b));
    CHECK(a != nullptr && b != nullptr);

    const turborerank_status bad =
        turborerank_load_model(a, "definitely-missing-ce-edge", 0);
    CHECK(bad == TURBORERANK_ERR_UNAVAILABLE);
    const char *msg_a = turborerank_last_error(a);
    CHECK(msg_a != nullptr);
    CHECK(std::strlen(msg_a) > 0);
    CHECK(std::strstr(msg_a, "definitely-missing-ce-edge") != nullptr);

    // Engine B never failed: its message must be empty and must not leak A's.
    const char *msg_b = turborerank_last_error(b);
    CHECK(msg_b != nullptr);
    CHECK_EQ(std::strlen(msg_b), 0u);
    // The thread-local create error is a separate store: A's load failure
    // must not appear there either.
    const char *msg_tls = turborerank_last_error(nullptr);
    CHECK(msg_tls != nullptr);
    CHECK(std::strstr(msg_tls, "definitely-missing-ce-edge") == nullptr);

    // A fresh failed create sets only the create error, not any engine's.
    turborerank_engine *trt = nullptr;
    CHECK(turborerank_engine_create(TURBORERANK_DEVICE_TENSORRT, nullptr, &trt) ==
          TURBORERANK_ERR_UNAVAILABLE);
    CHECK(trt == nullptr);
    const char *msg_create = turborerank_last_error(nullptr);
    CHECK(std::strlen(msg_create) > 0);
    CHECK(std::strstr(msg_create, "Refusing") != nullptr);
    // Engine A still carries its own load error, untouched by the create
    // path (the missing-weights text itself says "Refusing a mock score",
    // so match on the alias, not on the word "Refusing").
    CHECK_EQ(std::strlen(turborerank_last_error(a)) > 0, true);
    CHECK(std::strstr(turborerank_last_error(a), "definitely-missing-ce-edge") !=
          nullptr);
    CHECK(std::strstr(turborerank_last_error(a), "TENSORRT") == nullptr);
    CHECK_EQ(std::strlen(turborerank_last_error(b)), 0u);

    // Success clears B's message; a later failure on B leaves A untouched.
    if (weights_present()) {
        CHECK_ST(turborerank_load_model(b, "ms-marco-minilm-l6", 0));
        CHECK_EQ(std::strlen(turborerank_last_error(b)), 0u);
        const turborerank_status bad_b =
            turborerank_load_model(b, "other-missing-ce-edge", 0);
        CHECK(bad_b == TURBORERANK_ERR_UNAVAILABLE);
        CHECK(std::strstr(turborerank_last_error(b), "other-missing-ce-edge") !=
              nullptr);
        CHECK(std::strstr(turborerank_last_error(a), "other-missing-ce-edge") ==
              nullptr);
        CHECK(std::strstr(turborerank_last_error(a), "definitely-missing-ce-edge") !=
              nullptr);
    }
    turborerank_engine_destroy(a);
    turborerank_engine_destroy(b);
}

// Buffer geometry: row_stride == seq, base pointers 64-byte aligned, and a
// seq that is a multiple of 16 i32 keeps every row 64-byte aligned. Odd seq
// documents the row indexing arithmetic. batch == 0 and seq < 3 are
// rejected by buffer_alloc.
static void test_buffer_row_layout_and_alignment() {
    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 4, 16, &buf));
    CHECK_EQ(buf->batch, 4u);
    CHECK_EQ(buf->seq, 16u);
    CHECK_EQ(buf->row_stride, 16u);
    auto aligned = [](const void *p) {
        return (reinterpret_cast<uintptr_t>(p) % 64u) == 0;
    };
    for (uint32_t r = 0; r < 4; ++r) {
        const size_t off = static_cast<size_t>(r) * buf->row_stride;
        CHECK(aligned(buf->input_ids + off));
        CHECK(aligned(buf->attention_mask + off));
        CHECK(aligned(buf->token_type_ids + off));
        CHECK(aligned(buf->position_ids + off));
    }
    // Write the last element of the last row; row 0 must be unaffected.
    // (Arena rents are not zeroed — pack does that — so pin row 0 first.)
    buf->input_ids[0] = 5;
    buf->input_ids[3 * 16 + 15] = 4242;
    CHECK_EQ(buf->input_ids[0], 5);
    CHECK_EQ(buf->input_ids[3 * 16 + 15], 4242);
    turborerank_buffer_free(buf);

    // Odd seq: base stays 64-byte aligned; rows step by seq * sizeof(i32).
    buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 3, 13, &buf));
    CHECK_EQ(buf->row_stride, 13u);
    CHECK(aligned(buf->input_ids));
    CHECK_EQ(buf->input_ids + 13, buf->input_ids + 1 * buf->row_stride);
    CHECK_EQ(buf->attention_mask + 26, buf->attention_mask + 2 * buf->row_stride);
    buf->input_ids[0] = 7;
    buf->input_ids[2 * 13 + 12] = 1717;
    CHECK_EQ(buf->input_ids[2 * 13 + 12], 1717);
    CHECK_EQ(buf->input_ids[0], 7);
    turborerank_buffer_free(buf);

    // Argument validation edges.
    buf = nullptr;
    CHECK(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 0, 16, &buf) ==
          TURBORERANK_ERR_INVALID_ARGUMENT);
    CHECK(buf == nullptr);
    CHECK(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 1, 2, &buf) ==
          TURBORERANK_ERR_INVALID_ARGUMENT);
    CHECK(buf == nullptr);
    CHECK(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 1, 0, &buf) ==
          TURBORERANK_ERR_INVALID_ARGUMENT);
    CHECK(buf == nullptr);
}

// reranker.hpp layer: a burst of pack_pair_ids + row_at calls must not move
// the process alloc counter, and the packed layout is exactly
// [CLS] q [SEP] d [SEP] with HF-style mask/types/positions.
static void test_pack_pair_ids_burst_no_allocs() {
    int32_t ids_a[32], mask_a[32], types_a[32], pos_a[32];
    int32_t ids_b[32], mask_b[32], types_b[32], pos_b[32];
    const int32_t q[3] = {10, 11, 12};
    const int32_t d[3] = {20, 21, 22};

    turborerank::alloc_counter_reset();
    for (int i = 0; i < 200; ++i) {
        const bool use_a = (i % 2) == 0;
        int32_t *ids = use_a ? ids_a : ids_b;
        int32_t *mask = use_a ? mask_a : mask_b;
        int32_t *types = use_a ? types_a : types_b;
        int32_t *pos = use_a ? pos_a : pos_b;
        turborerank::Status st = turborerank::Status::Ok;
        const size_t nq = 1 + static_cast<size_t>(i % 3);
        const size_t nd = static_cast<size_t>(i % 3);
        const uint32_t packed = turborerank::pack_pair_ids(
            ids, mask, types, pos, 32, q, nq, d, nd,
            turborerank::Truncation::LongestFirst, 16, &st
        );
        CHECK(st == turborerank::Status::Ok);
        CHECK_EQ(packed, turborerank::kSpecials + nq + nd);
    }
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);

    // Exact layout for q={10,11}, d={20,21,22} at seq_capacity 32.
    turborerank::Status st = turborerank::Status::Ok;
    const uint32_t packed = turborerank::pack_pair_ids(
        ids_a, mask_a, types_a, pos_a, 32, q, 2, d, 3,
        turborerank::Truncation::LongestFirst, 16, &st
    );
    CHECK(st == turborerank::Status::Ok);
    CHECK_EQ(packed, 8u);
    const int32_t want_ids[8] = {101, 10, 11, 102, 20, 21, 22, 102};
    const int32_t want_types[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    CHECK_EQ(std::memcmp(ids_a, want_ids, sizeof(want_ids)), 0);
    CHECK_EQ(std::memcmp(types_a, want_types, sizeof(want_types)), 0);
    for (int i = 0; i < 8; ++i) {
        CHECK_EQ(mask_a[i], 1);
        CHECK_EQ(pos_a[i], i);
    }
    // Padding: ids/types/mask zeroed, positions continue the arange
    // (HF assigns arange over the full used capacity).
    for (uint32_t p = 8; p < 32; ++p) {
        CHECK_EQ(ids_a[p], 0);
        CHECK_EQ(mask_a[p], 0);
        CHECK_EQ(types_a[p], 0);
        CHECK_EQ(pos_a[p], static_cast<int32_t>(p));
    }

    // An empty pair is legal at this layer: [CLS] [SEP] [SEP].
    st = turborerank::Status::Ok;
    const uint32_t empty = turborerank::pack_pair_ids(
        ids_b, mask_b, types_b, pos_b, 32, nullptr, 0, nullptr, 0,
        turborerank::Truncation::LongestFirst, 16, &st
    );
    CHECK(st == turborerank::Status::Ok);
    CHECK_EQ(empty, turborerank::kSpecials);
    CHECK_EQ(ids_b[0], 101);
    CHECK_EQ(ids_b[1], 102);
    CHECK_EQ(ids_b[2], 102);
    CHECK_EQ(mask_b[2], 1);

    // Error edges: null destination, capacity < specials, max_length < 3.
    st = turborerank::Status::Ok;
    CHECK_EQ(turborerank::pack_pair_ids(
                 nullptr, mask_b, types_b, pos_b, 32, q, 2, d, 3,
                 turborerank::Truncation::LongestFirst, 16, &st),
             0u);
    CHECK(st == turborerank::Status::InvalidArgument);
    st = turborerank::Status::Ok;
    CHECK_EQ(turborerank::pack_pair_ids(
                 ids_b, mask_b, types_b, pos_b, 2, q, 2, d, 3,
                 turborerank::Truncation::LongestFirst, 16, &st),
             0u);
    CHECK(st == turborerank::Status::InvalidArgument);
    st = turborerank::Status::Ok;
    CHECK_EQ(turborerank::pack_pair_ids(
                 ids_b, mask_b, types_b, pos_b, 32, q, 2, d, 3,
                 turborerank::Truncation::LongestFirst, 2, &st),
             0u);
    CHECK(st == turborerank::Status::InvalidArgument);
    // TRUNC_ERROR on an over-long pair.
    st = turborerank::Status::Ok;
    CHECK_EQ(turborerank::pack_pair_ids(
                 ids_b, mask_b, types_b, pos_b, 32, q, 3, d, 3,
                 turborerank::Truncation::Error, 6, &st),
             0u);
    CHECK(st == turborerank::Status::InvalidArgument);
    // status may be NULL: the return value alone reports success.
    const uint32_t no_status = turborerank::pack_pair_ids(
        ids_b, mask_b, types_b, pos_b, 32, q, 2, d, 3,
        turborerank::Truncation::LongestFirst, 16, nullptr
    );
    CHECK_EQ(no_status, 8u);
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
}

// row_at (reranker.hpp): pure pointer arithmetic, in-range rows view the
// batch row, out-of-range rows return the span unchanged.
static void test_row_at_span_math() {
    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 3, 16, &buf));
    turborerank::TokenSpan span{
        buf->input_ids, buf->attention_mask, buf->token_type_ids,
        buf->position_ids, buf->batch, buf->seq, buf->row_stride
    };
    CHECK_EQ(span.batch, 3u);
    turborerank::alloc_counter_reset();
    const turborerank::TokenSpan row1 = turborerank::row_at(span, 1);
    CHECK(row1.input_ids == buf->input_ids + 16);
    CHECK(row1.attention_mask == buf->attention_mask + 16);
    CHECK(row1.token_type_ids == buf->token_type_ids + 16);
    CHECK(row1.position_ids == buf->position_ids + 16);
    CHECK_EQ(row1.batch, 1u);
    CHECK_EQ(row1.row_stride, 16u);
    const turborerank::TokenSpan row2 = turborerank::row_at(span, 2);
    CHECK(row2.input_ids == buf->input_ids + 32);
    // row == batch (out of range): span comes back unchanged.
    const turborerank::TokenSpan past = turborerank::row_at(span, 3);
    CHECK(past.input_ids == buf->input_ids);
    CHECK(past.batch == 3u);
    // Writing through the row view lands in the parent buffer's row.
    row1.input_ids[0] = 777;
    CHECK_EQ(buf->input_ids[16], 777);
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    turborerank_buffer_free(buf);
}

// Real-model relevance sanity on fresh pairs (no golden numbers): for two
// unrelated query/doc pairs the on-topic document must outrank the off-topic
// one by a clear margin under both activations.
static void test_real_model_relevance_sanity() {
    turborerank_engine *e = nullptr;
    if (!load_cpu_engine(&e)) {
        return;
    }
    struct Case {
        const char *query;
        const char *relevant;
        const char *irrelevant;
    };
    const Case cases[2] = {
        {
            "What is the chemical symbol for gold?",
            "Gold has the chemical symbol Au and atomic number 79.",
            "The French Revolution began in 1789 in Paris.",
        },
        {
            "Who wrote the play Romeo and Juliet?",
            "William Shakespeare wrote Romeo and Juliet in the mid-1590s.",
            "The Amazon rainforest produces a large share of the world's oxygen.",
        },
    };
    for (const Case &c : cases) {
        turborerank_str query{c.query, std::strlen(c.query)};
        turborerank_str docs[2] = {
            {c.relevant, std::strlen(c.relevant)},
            {c.irrelevant, std::strlen(c.irrelevant)},
        };
        turborerank_score_options opts{};
        opts.truncation = TURBORERANK_TRUNC_LONGEST_FIRST;
        opts.max_length = 512;
        float logits[2] = {0, 0};
        opts.activation = TURBORERANK_ACT_IDENTITY;
        CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 2, &opts, logits));
        CHECK(logits[0] > logits[1]);
        CHECK(logits[0] - logits[1] > 2.0f); // clear separation, not noise
        float sig[2] = {0, 0};
        opts.activation = TURBORERANK_ACT_SIGMOID;
        CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 2, &opts, sig));
        CHECK(sig[0] > 0.5f); // on-topic reads as relevant under sigmoid
        CHECK(sig[0] > sig[1]);
        CHECK(sig[1] > 0.0f && sig[1] < 1.0f);
    }
    turborerank_engine_destroy(e);
}

int main() {
    test_score_rejects_zero_documents();
    test_score_empty_query_and_doc();
    test_score_is_bitwise_deterministic();
    test_sigmoid_scores_bounded();
    test_identity_ordering_preserved_by_sigmoid();
    test_truncation_longest_first_query_heavy();
    test_truncation_modes_overlong_text_pair();
    test_max_length_boundary_seq_minus_plus();
    test_forward_rejects_zero_and_over_batch_rows();
    test_last_error_is_engine_local();
    test_buffer_row_layout_and_alignment();
    test_pack_pair_ids_burst_no_allocs();
    test_row_at_span_math();
    test_real_model_relevance_sanity();

    std::fprintf(
        stderr,
        "turborerank_edge_tests: %d passed, %d failed\n",
        g_passes,
        g_fails
    );
    return g_fails == 0 ? 0 : 1;
}
