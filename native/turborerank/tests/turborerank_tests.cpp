// SPDX-License-Identifier: Apache-2.0
//
// Close-to-metal TurboRerank tests. No stub scores.

#include "cuda_api.hpp"
#include "internal.hpp"
#include "reranker.hpp"
#include "turborerank.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

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
           file_exists(model_dir() + "/vocab.txt");
}

static void test_abi_names() {
    CHECK_EQ(turborerank_abi_version(), 1u);
    CHECK(std::strcmp(turborerank_status_name(TURBORERANK_ERR_NOT_IMPLEMENTED),
                      "NOT_IMPLEMENTED") == 0);
    CHECK(std::strcmp(turborerank_device_name(TURBORERANK_DEVICE_CPU), "CPU") == 0);
}

static void test_buffer_alignment_and_write_via_pointer() {
    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 2, 16, &buf));
    CHECK(buf != nullptr);
    CHECK(buf->row_stride == 16);
    CHECK(buf->batch == 2);
    auto aligned = [](const void *p) {
        return (reinterpret_cast<uintptr_t>(p) % 64u) == 0;
    };
    CHECK(aligned(buf->input_ids));
    CHECK(aligned(buf->attention_mask));
    CHECK(aligned(buf->token_type_ids));
    CHECK(aligned(buf->position_ids));

    // Caller writes tokens directly — no helper required.
    buf->input_ids[0] = 101;
    buf->input_ids[1] = 7592;
    buf->input_ids[2] = 102;
    CHECK_EQ(buf->input_ids[0], 101);
    CHECK_EQ(buf->input_ids[1], 7592);

    turborerank::alloc_counter_reset();
    // Re-writing through the same pointer must not allocate.
    buf->input_ids[3] = 2088;
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    turborerank_buffer_free(buf);
}

static bool cuda_live() {
    std::string why;
    return turborerank::impl::cuda_device_present(&why);
}

static void test_cuda_buffer_policy() {
    turborerank_buffer *buf = nullptr;
    const turborerank_status st =
        turborerank_buffer_alloc(TURBORERANK_DEVICE_CUDA, 2, 16, &buf);
    if (!cuda_live()) {
        CHECK(st == TURBORERANK_ERR_NOT_IMPLEMENTED ||
              st == TURBORERANK_ERR_UNAVAILABLE);
        CHECK(buf == nullptr);
        const char *msg = turborerank_last_error(nullptr);
        CHECK(msg != nullptr);
        CHECK(std::strstr(msg, "Refusing CPU") != nullptr ||
              std::strstr(msg, "refusing CPU") != nullptr ||
              std::strstr(msg, "without CUDA") != nullptr);
        return;
    }
    CHECK_ST(st);
    CHECK(buf != nullptr);
    CHECK(buf->device == TURBORERANK_DEVICE_CUDA);
    auto aligned = [](const void *p) {
        return (reinterpret_cast<uintptr_t>(p) % 64u) == 0;
    };
    CHECK(aligned(buf->input_ids));
    CHECK(aligned(buf->attention_mask));
    CHECK(aligned(buf->token_type_ids));
    CHECK(aligned(buf->position_ids));
    // Caller writes tokens into pinned host memory.
    buf->input_ids[0] = 101;
    buf->input_ids[1] = 7592;
    buf->input_ids[2] = 102;
    CHECK_EQ(buf->input_ids[0], 101);
    CHECK_EQ(buf->input_ids[1], 7592);
    turborerank::alloc_counter_reset();
    buf->input_ids[3] = 2088;
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    turborerank_buffer_free(buf);

    turborerank_buffer *auto_buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_AUTO, 1, 8, &auto_buf));
    CHECK(auto_buf->device == TURBORERANK_DEVICE_CUDA);
    CHECK(aligned(auto_buf->input_ids));
    turborerank_buffer_free(auto_buf);
}

static void test_pack_cls_sep_and_types() {
    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 1, 16, &buf));
    const int32_t q[] = {10, 11};
    const int32_t d[] = {20, 21, 22};
    CHECK_ST(turborerank_pack_ids(
        buf, 0, q, 2, d, 3, TURBORERANK_TRUNC_LONGEST_FIRST, 16
    ));
    // [CLS] 10 11 [SEP] 20 21 22 [SEP]
    CHECK_EQ(buf->input_ids[0], 101);
    CHECK_EQ(buf->input_ids[1], 10);
    CHECK_EQ(buf->input_ids[2], 11);
    CHECK_EQ(buf->input_ids[3], 102);
    CHECK_EQ(buf->input_ids[4], 20);
    CHECK_EQ(buf->input_ids[5], 21);
    CHECK_EQ(buf->input_ids[6], 22);
    CHECK_EQ(buf->input_ids[7], 102);
    CHECK_EQ(buf->input_ids[8], 0);
    for (int i = 0; i < 8; ++i) {
        CHECK_EQ(buf->attention_mask[i], 1);
    }
    CHECK_EQ(buf->attention_mask[8], 0);
    CHECK_EQ(buf->token_type_ids[0], 0);
    CHECK_EQ(buf->token_type_ids[3], 0);
    CHECK_EQ(buf->token_type_ids[4], 1);
    CHECK_EQ(buf->token_type_ids[7], 1);
    CHECK_EQ(buf->position_ids[0], 0);
    CHECK_EQ(buf->position_ids[7], 7);
    turborerank_buffer_free(buf);
}

static void test_pack_empty_sides() {
    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 2, 8, &buf));
    const int32_t q[] = {5};
    CHECK_ST(turborerank_pack_ids(
        buf, 0, q, 1, nullptr, 0, TURBORERANK_TRUNC_LONGEST_FIRST, 8
    ));
    CHECK_EQ(buf->input_ids[0], 101);
    CHECK_EQ(buf->input_ids[1], 5);
    CHECK_EQ(buf->input_ids[2], 102);
    CHECK_EQ(buf->input_ids[3], 102);
    CHECK_EQ(buf->token_type_ids[3], 1);

    const int32_t d[] = {9, 8};
    CHECK_ST(turborerank_pack_ids(
        buf, 1, nullptr, 0, d, 2, TURBORERANK_TRUNC_LONGEST_FIRST, 8
    ));
    CHECK_EQ(buf->input_ids[8 + 0], 101);
    CHECK_EQ(buf->input_ids[8 + 1], 102);
    CHECK_EQ(buf->input_ids[8 + 2], 9);
    CHECK_EQ(buf->input_ids[8 + 3], 8);
    CHECK_EQ(buf->input_ids[8 + 4], 102);
    turborerank_buffer_free(buf);
}

static void test_pack_truncation() {
    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 3, 16, &buf));
    const int32_t q[] = {1, 2, 3, 4, 5};          // 5
    const int32_t d[] = {6, 7, 8, 9, 10, 11, 12}; // 7
    // max_length = 8 → budget 5. Longest-first drops from doc first (7>5)
    // then alternates until nq+nd=5.
    CHECK_ST(turborerank_pack_ids(
        buf, 0, q, 5, d, 7, TURBORERANK_TRUNC_LONGEST_FIRST, 8
    ));
    uint32_t ones = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        ones += static_cast<uint32_t>(buf->attention_mask[i] == 1);
    }
    CHECK_EQ(ones, 8u);
    CHECK_EQ(buf->input_ids[0], 101);
    CHECK_EQ(buf->input_ids[7], 102);

    // Query-priority: keep all 5 query tokens, doc gets 0.
    CHECK_ST(turborerank_pack_ids(
        buf, 1, q, 5, d, 7, TURBORERANK_TRUNC_QUERY_PRIORITY, 8
    ));
    CHECK_EQ(buf->input_ids[16 + 0], 101);
    CHECK_EQ(buf->input_ids[16 + 1], 1);
    CHECK_EQ(buf->input_ids[16 + 5], 5);
    CHECK_EQ(buf->input_ids[16 + 6], 102);
    CHECK_EQ(buf->input_ids[16 + 7], 102);

    // ERROR mode
    const turborerank_status st = turborerank_pack_ids(
        buf, 2, q, 5, d, 7, TURBORERANK_TRUNC_ERROR, 8
    );
    CHECK(st == TURBORERANK_ERR_INVALID_ARGUMENT);

    // max_length == 3 → only specials
    CHECK_ST(turborerank_pack_ids(
        buf, 2, q, 5, d, 7, TURBORERANK_TRUNC_LONGEST_FIRST, 3
    ));
    CHECK_EQ(buf->input_ids[32 + 0], 101);
    CHECK_EQ(buf->input_ids[32 + 1], 102);
    CHECK_EQ(buf->input_ids[32 + 2], 102);

    const turborerank_status bad =
        turborerank_pack_ids(buf, 2, q, 5, d, 7, TURBORERANK_TRUNC_LONGEST_FIRST, 2);
    CHECK(bad == TURBORERANK_ERR_INVALID_ARGUMENT);
    turborerank_buffer_free(buf);
}

static void test_device_create_policy() {
    turborerank_engine *e = nullptr;
    if (cuda_live()) {
        CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_CUDA, nullptr, &e));
        CHECK(e != nullptr);
        CHECK(e->device == TURBORERANK_DEVICE_CUDA);
        turborerank_engine_destroy(e);
        e = nullptr;
        CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_AUTO, nullptr, &e));
        CHECK(e != nullptr);
        CHECK(e->device == TURBORERANK_DEVICE_CUDA);
        turborerank_engine_destroy(e);
        e = nullptr;
    } else {
        CHECK(turborerank_engine_create(TURBORERANK_DEVICE_CUDA, nullptr, &e) ==
              TURBORERANK_ERR_UNAVAILABLE);
        CHECK(e == nullptr);
        const char *msg = turborerank_last_error(nullptr);
        CHECK(std::strstr(msg, "Refusing CPU") != nullptr);

        CHECK(turborerank_engine_create(TURBORERANK_DEVICE_AUTO, nullptr, &e) ==
              TURBORERANK_ERR_UNAVAILABLE);
        CHECK(std::strstr(turborerank_last_error(nullptr), "Refusing CPU") != nullptr);
    }

    CHECK(turborerank_engine_create(TURBORERANK_DEVICE_METAL, nullptr, &e) ==
          TURBORERANK_ERR_UNAVAILABLE);

    CHECK(turborerank_engine_create(TURBORERANK_DEVICE_TENSORRT, nullptr, &e) ==
          TURBORERANK_ERR_UNAVAILABLE);
    CHECK(std::strstr(turborerank_last_error(nullptr), "Refusing") != nullptr);

    CHECK(turborerank_engine_create(TURBORERANK_DEVICE_OPENVINO_CPU, nullptr, &e) ==
          TURBORERANK_ERR_NOT_IMPLEMENTED);

    CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_CPU, nullptr, &e));
    CHECK(e != nullptr);
    turborerank_engine_destroy(e);

    CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_MOCK, nullptr, &e));
    const turborerank_status load =
        turborerank_load_model(e, "ms-marco-minilm-l6", 0);
    CHECK(load == TURBORERANK_ERR_NOT_IMPLEMENTED);
    CHECK(std::strstr(turborerank_last_error(e), "mock") != nullptr ||
          std::strstr(turborerank_last_error(e), "MOCK") != nullptr);
    turborerank_engine_destroy(e);
}

#ifndef TURBORERANK_CUDA
static void test_create_without_cuda_fails() {
    CHECK(!turborerank::impl::cuda_compiled());
    turborerank_engine *e = nullptr;
    CHECK(turborerank_engine_create(TURBORERANK_DEVICE_CUDA, nullptr, &e) ==
          TURBORERANK_ERR_UNAVAILABLE);
    CHECK(e == nullptr);
    const char *msg = turborerank_last_error(nullptr);
    CHECK(std::strstr(msg, "without CUDA") != nullptr ||
          std::strstr(msg, "Refusing CPU") != nullptr);
    turborerank_buffer *buf = nullptr;
    CHECK(turborerank_buffer_alloc(TURBORERANK_DEVICE_CUDA, 1, 16, &buf) ==
          TURBORERANK_ERR_UNAVAILABLE);
    CHECK(buf == nullptr);
}
#endif

static void test_missing_weights_fails_loud() {
    turborerank_engine *e = nullptr;
    CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_CPU, "/no/such/turborerank", &e));
    const turborerank_status st =
        turborerank_load_model(e, "definitely-missing-ce", 0);
    CHECK(st == TURBORERANK_ERR_UNAVAILABLE);
    const char *msg = turborerank_last_error(e);
    CHECK(std::strstr(msg, "missing") != nullptr || std::strstr(msg, "not found") != nullptr);
    CHECK(std::strstr(msg, "mock") != nullptr || std::strstr(msg, "Refusing") != nullptr ||
          std::strstr(msg, "expected") != nullptr);
    turborerank_engine_destroy(e);
}

static void test_wordpiece_fixture() {
    const std::string dir = workspace_root() + "/testdata/turborerank/tiny-vocab";
    if (!file_exists(dir + "/vocab.txt")) {
        std::fprintf(stderr, "SKIP wordpiece fixture (no testdata)\n");
        return;
    }
    turborerank::impl::Vocab v;
    std::string err;
    CHECK(turborerank::impl::load_vocab_txt((dir + "/vocab.txt").c_str(), &v, &err));
    int32_t ids[16];
    const char hello[] = "hello world";
    const size_t n = turborerank::impl::tokenize_wordpiece(
        v, hello, sizeof(hello) - 1, ids, 16, &err
    );
    CHECK(n == 2);
    CHECK_EQ(ids[0], 4); // hello
    CHECK_EQ(ids[1], 5); // world

    const char cafe[] = "CAF\u00C9"; // CAFÉ — lower+strip → cafe if in vocab as cafe? tiny vocab has cafe
    // Our fixture may only have hello/world. Tokenize unaffable-style ##.
    const char ing[] = "helloing";
    const size_t n2 = turborerank::impl::tokenize_wordpiece(
        v, ing, sizeof(ing) - 1, ids, 16, &err
    );
    CHECK(n2 == 2);
    CHECK_EQ(ids[0], 4);
    CHECK_EQ(ids[1], 6); // ##ing
}

static bool almost(float a, float b, float eps) {
    return std::fabs(a - b) <= eps;
}

static void test_real_model_scores() {
    if (!weights_present()) {
        std::fprintf(
            stderr,
            "SKIP live MiniLM CE scores (fetch with make fetch-rerankers)\n"
        );
        return;
    }
    turborerank_engine *e = nullptr;
    const std::string dir = model_dir();
    CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_CPU, dir.c_str(), &e));
    const turborerank_status load = turborerank_load_model(e, "ms-marco-minilm-l6", 0);
    CHECK_ST(load);
    if (load != TURBORERANK_OK) {
        std::fprintf(stderr, "load error: %s\n", turborerank_last_error(e));
        turborerank_engine_destroy(e);
        return;
    }

    const char *q = "How many people live in Berlin?";
    const char *rel =
        "Berlin has a population of 3,520,031 registered inhabitants in an "
        "area of 891.82 square kilometers.";
    const char *irrel = "New York City is famous for its pizza and bagels.";
    const char *mid = "Berlin is well known for its museums.";

    turborerank_str query{q, std::strlen(q)};
    turborerank_str docs[3] = {
        {rel, std::strlen(rel)},
        {mid, std::strlen(mid)},
        {irrel, std::strlen(irrel)},
    };
    turborerank_score_options opts{};
    opts.truncation = TURBORERANK_TRUNC_LONGEST_FIRST;
    opts.activation = TURBORERANK_ACT_IDENTITY;
    opts.max_length = 512;
    float logits[3] = {0, 0, 0};
    CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 3, &opts, logits));

    // Must not be constant / FNV-flat.
    CHECK(logits[0] != logits[1] || logits[1] != logits[2]);
    CHECK(logits[0] > logits[1]);
    CHECK(logits[1] > logits[2]);
    const float span = logits[0] - logits[2];
    CHECK(span > 2.0f); // real CE separates these by several logit units

    opts.activation = TURBORERANK_ACT_SIGMOID;
    float sig[3] = {0, 0, 0};
    CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 3, &opts, sig));
    CHECK(almost(sig[0], turborerank::sigmoid(logits[0]), 1e-5f));
    CHECK(almost(sig[1], turborerank::sigmoid(logits[1]), 1e-5f));
    CHECK(sig[0] > sig[1] && sig[1] > sig[2]);

    // Batch vs one-by-one
    float one[3];
    for (int i = 0; i < 3; ++i) {
        opts.activation = TURBORERANK_ACT_IDENTITY;
        CHECK_ST(turborerank_score(e, nullptr, 0, query, &docs[i], 1, &opts, &one[i]));
        CHECK(almost(one[i], logits[i], 1e-5f));
    }

    // No heap growth on forward.
    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 1, 64, &buf));
    CHECK_ST(turborerank_pack_text(
        e, buf, 0, query, docs[0], TURBORERANK_TRUNC_LONGEST_FIRST, 64
    ));
    turborerank::alloc_counter_reset();
    float s = 0;
    CHECK_ST(turborerank_forward(e, buf, 1, TURBORERANK_ACT_IDENTITY, &s));
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    CHECK(s > 0.0f);
    turborerank_buffer_free(buf);

    // Frozen HF / ST golden vector (testdata/reference_rerank).
    CHECK(almost(logits[0], 8.84585285f, 2e-3f));
    CHECK(almost(logits[1], -4.32007599f, 2e-3f));
    CHECK(almost(logits[2], -11.27389431f, 2e-3f));
    // Constant/FNV mock cannot hit these three numbers together.
    CHECK(logits[0] > 5.0f && logits[2] < -5.0f);

    turborerank_engine_destroy(e);
}

static void test_cuda_real_model_scores() {
    if (!cuda_live()) {
        std::fprintf(stderr, "SKIP CUDA MiniLM CE (no device / not compiled)\n");
        return;
    }
    if (!weights_present()) {
        std::fprintf(stderr, "SKIP CUDA MiniLM CE scores (make fetch-rerankers)\n");
        return;
    }
    turborerank_engine *e = nullptr;
    const std::string dir = model_dir();
    CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_CUDA, dir.c_str(), &e));
    CHECK(e->device == TURBORERANK_DEVICE_CUDA);
    const turborerank_status load = turborerank_load_model(e, "ms-marco-minilm-l6", 0);
    CHECK_ST(load);
    if (load != TURBORERANK_OK) {
        std::fprintf(stderr, "CUDA load error: %s\n", turborerank_last_error(e));
        turborerank_engine_destroy(e);
        return;
    }
    CHECK(e->cuda.enabled);
    CHECK_EQ(e->cuda.n_layers, 6u);

    const char *q = "How many people live in Berlin?";
    const char *rel =
        "Berlin has a population of 3,520,031 registered inhabitants in an "
        "area of 891.82 square kilometers.";
    const char *irrel = "New York City is famous for its pizza and bagels.";
    const char *mid = "Berlin is well known for its museums.";
    turborerank_str query{q, std::strlen(q)};
    turborerank_str docs[3] = {
        {rel, std::strlen(rel)},
        {mid, std::strlen(mid)},
        {irrel, std::strlen(irrel)},
    };
    turborerank_score_options opts{};
    opts.truncation = TURBORERANK_TRUNC_LONGEST_FIRST;
    opts.activation = TURBORERANK_ACT_IDENTITY;
    opts.max_length = 512;
    float logits[3] = {0, 0, 0};
    CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 3, &opts, logits));
    CHECK(logits[0] != logits[1] || logits[1] != logits[2]);
    CHECK(logits[0] > logits[1]);
    CHECK(logits[1] > logits[2]);
    CHECK(logits[0] - logits[2] > 2.0f);
    CHECK(almost(logits[0], 8.84585285f, 2e-3f));
    CHECK(almost(logits[1], -4.32007599f, 2e-3f));
    CHECK(almost(logits[2], -11.27389431f, 2e-3f));

    opts.activation = TURBORERANK_ACT_SIGMOID;
    float sig[3] = {0, 0, 0};
    CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 3, &opts, sig));
    CHECK(almost(sig[0], turborerank::sigmoid(logits[0]), 1e-5f));
    CHECK(sig[0] > sig[1] && sig[1] > sig[2]);

    float one[3];
    for (int i = 0; i < 3; ++i) {
        opts.activation = TURBORERANK_ACT_IDENTITY;
        CHECK_ST(turborerank_score(e, nullptr, 0, query, &docs[i], 1, &opts, &one[i]));
        CHECK(almost(one[i], logits[i], 1e-5f));
    }

    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CUDA, 1, 64, &buf));
    CHECK(buf->device == TURBORERANK_DEVICE_CUDA);
    CHECK_ST(turborerank_pack_text(
        e, buf, 0, query, docs[0], TURBORERANK_TRUNC_LONGEST_FIRST, 64
    ));
    turborerank::alloc_counter_reset();
    float s = 0;
    CHECK_ST(turborerank_forward(e, buf, 1, TURBORERANK_ACT_IDENTITY, &s));
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    CHECK(almost(s, logits[0], 2e-3f));
    turborerank_buffer_free(buf);

    // AUTO → CUDA on this host.
    turborerank_engine *auto_e = nullptr;
    CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_AUTO, dir.c_str(), &auto_e));
    CHECK(auto_e->device == TURBORERANK_DEVICE_CUDA);
    CHECK_ST(turborerank_load_model(auto_e, "ms-marco-minilm-l6", 0));
    opts.activation = TURBORERANK_ACT_IDENTITY;
    float auto_logit = 0;
    CHECK_ST(turborerank_score(auto_e, nullptr, 0, query, &docs[0], 1, &opts, &auto_logit));
    CHECK(almost(auto_logit, logits[0], 2e-3f));
    turborerank_engine_destroy(auto_e);

    std::fprintf(
        stderr,
        "CUDA Berlin logits: %.8f %.8f %.8f (abs err %.3e %.3e %.3e)\n",
        logits[0],
        logits[1],
        logits[2],
        std::fabs(logits[0] - 8.84585285f),
        std::fabs(logits[1] + 4.32007599f),
        std::fabs(logits[2] + 11.27389431f)
    );
    turborerank_engine_destroy(e);
}

int main() {
    test_abi_names();
    test_buffer_alignment_and_write_via_pointer();
    test_cuda_buffer_policy();
    test_pack_cls_sep_and_types();
    test_pack_empty_sides();
    test_pack_truncation();
    test_device_create_policy();
#ifndef TURBORERANK_CUDA
    test_create_without_cuda_fails();
#endif
    test_missing_weights_fails_loud();
    test_wordpiece_fixture();
    test_real_model_scores();
    test_cuda_real_model_scores();

    std::fprintf(
        stderr,
        "turborerank_tests: %d passed, %d failed\n",
        g_passes,
        g_fails
    );
    return g_fails == 0 ? 0 : 1;
}
