// SPDX-License-Identifier: Apache-2.0
//
// Close-to-metal TurboRerank tests. No stub scores.

#include "cuda_api.hpp"
#include "internal.hpp"
#include "metal_api.hpp"
#include "ov_api.hpp"
#include "reranker.hpp"
#include "turbo_buffer.h"
#include "turborerank.h"
#include "wordpiece.h"

#ifdef TURBORERANK_CUDA
#include <cuda_runtime.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
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

static std::string ov_ir_dir() {
    if (const char *e = std::getenv("TURBORERANK_OV_MODEL_DIR")) {
        return e;
    }
    return workspace_root() + "/models/ov-rerank/ms-marco-minilm-l6";
}

static bool ov_ir_present() {
    return (file_exists(ov_ir_dir() + "/openvino_model.xml") ||
            file_exists(ov_ir_dir() + "/model.xml")) &&
           file_exists(ov_ir_dir() + "/vocab.txt");
}

static bool ov_gpu_live() {
    return turborerank::impl::ov_gpu_present(nullptr);
}

static bool ov_cpu_live() {
    return turborerank::impl::ov_cpu_present(nullptr);
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

static bool metal_live() {
    std::string why;
    return turborerank::impl::metal_device_present(&why);
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

static void test_cuda_gemm_stack_is_cublaslt() {
    const char *backend = turborerank::impl::cuda_gemm_backend();
    CHECK(backend != nullptr);
#ifdef TURBORERANK_CUDA
    CHECK(std::strcmp(backend, "cublasLtMatmul") == 0);
    const std::string src_path =
        workspace_root() + "/native/turborerank/src/bert_cuda.cu";
    std::ifstream in(src_path);
    CHECK(static_cast<bool>(in));
    const std::string src(
        (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()
    );
    CHECK(src.find("cublasLtMatmul(") != std::string::npos);
    CHECK(src.find("cublasLtMatmulAlgoGetHeuristic") != std::string::npos);
    CHECK(src.find("linear_nt_kernel") == std::string::npos);
    CHECK(src.find("refusing hand-rolled") != std::string::npos);
#else
    CHECK(std::strcmp(backend, "unavailable") == 0);
#endif
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

        if (ov_gpu_live()) {
            CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_AUTO, nullptr, &e));
            CHECK(e != nullptr);
            CHECK(e->device == TURBORERANK_DEVICE_OPENVINO_GPU);
            turborerank_engine_destroy(e);
            e = nullptr;
        } else if (metal_live()) {
            CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_AUTO, nullptr, &e));
            CHECK(e != nullptr);
            CHECK(e->device == TURBORERANK_DEVICE_METAL);
            turborerank_engine_destroy(e);
            e = nullptr;
        } else {
            CHECK(turborerank_engine_create(TURBORERANK_DEVICE_AUTO, nullptr, &e) ==
                  TURBORERANK_ERR_UNAVAILABLE);
            CHECK(std::strstr(turborerank_last_error(nullptr), "Refusing CPU") != nullptr);
        }
    }

    if (metal_live()) {
        CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_METAL, nullptr, &e));
        CHECK(e != nullptr);
        CHECK(e->device == TURBORERANK_DEVICE_METAL);
        turborerank_engine_destroy(e);
        e = nullptr;
    } else {
        CHECK(turborerank_engine_create(TURBORERANK_DEVICE_METAL, nullptr, &e) ==
              TURBORERANK_ERR_UNAVAILABLE);
        CHECK(std::strstr(turborerank_last_error(nullptr), "Refusing") != nullptr);
    }

    CHECK(turborerank_engine_create(TURBORERANK_DEVICE_TENSORRT, nullptr, &e) ==
          TURBORERANK_ERR_UNAVAILABLE);
    CHECK(std::strstr(turborerank_last_error(nullptr), "Refusing") != nullptr);

    if (ov_cpu_live()) {
        CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_OPENVINO_CPU, nullptr, &e));
        CHECK(e != nullptr);
        CHECK(e->device == TURBORERANK_DEVICE_OPENVINO_CPU);
        turborerank_engine_destroy(e);
        e = nullptr;
    } else if (turborerank::impl::ov_compiled()) {
        CHECK(turborerank_engine_create(TURBORERANK_DEVICE_OPENVINO_CPU, nullptr, &e) ==
              TURBORERANK_ERR_UNAVAILABLE);
    } else {
        CHECK(turborerank_engine_create(TURBORERANK_DEVICE_OPENVINO_CPU, nullptr, &e) ==
              TURBORERANK_ERR_NOT_IMPLEMENTED);
    }

    if (ov_gpu_live()) {
        CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_OPENVINO_GPU, nullptr, &e));
        CHECK(e != nullptr);
        CHECK(e->device == TURBORERANK_DEVICE_OPENVINO_GPU);
        turborerank_engine_destroy(e);
        e = nullptr;
    } else {
        CHECK(turborerank_engine_create(TURBORERANK_DEVICE_OPENVINO_GPU, nullptr, &e) ==
              TURBORERANK_ERR_UNAVAILABLE);
        CHECK(std::strstr(turborerank_last_error(nullptr), "Refusing") != nullptr ||
              std::strstr(turborerank_last_error(nullptr), "refus") != nullptr);
    }

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

    // Scratch + work buffer are arena-owned. Reintroducing private
    // malloc for those slots fails this owns() check.
    CHECK(e->arena != nullptr);
    CHECK(turbo_buffer_arena_owns(e->arena, e->scratch.x));
    CHECK(turbo_buffer_arena_owns(e->arena, e->work->input_ids));
    CHECK(turbo_buffer_arena_owns(e->arena, e->work->attention_mask));
    CHECK(turbo_buffer_arena_owns(e->arena, e->work->token_type_ids));
    CHECK(turbo_buffer_arena_owns(e->arena, e->work->position_ids));
    CHECK(e->vocab.img != nullptr);
    CHECK(wordpiece_vocab_is_loaded(e->vocab.img));

    // High-level score reuses the load-time work buffer + scratch.
    // Pack writes WordPiece ids directly into the rented row.
    turborerank::alloc_counter_reset();
    wordpiece_hot_alloc_counter_reset();
    opts.activation = TURBORERANK_ACT_IDENTITY;
    float score_again[3] = {0, 0, 0};
    CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 3, &opts, score_again));
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    CHECK_EQ(wordpiece_hot_alloc_counter(), 0u);
    CHECK(almost(score_again[0], logits[0], 1e-5f));
    CHECK_EQ(e->work->input_ids[0], 101); // [CLS] write-through

    // No heap growth on forward. Pack writes into the rented row.
    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 1, 64, &buf));
    wordpiece_hot_alloc_counter_reset();
    CHECK_ST(turborerank_pack_text(
        e, buf, 0, query, docs[0], TURBORERANK_TRUNC_LONGEST_FIRST, 64
    ));
    CHECK_EQ(wordpiece_hot_alloc_counter(), 0u);
    CHECK_EQ(buf->input_ids[0], 101);
    CHECK_EQ(buf->attention_mask[0], 1);
    turborerank::alloc_counter_reset();
    float s = 0;
    CHECK_ST(turborerank_forward(e, buf, 1, TURBORERANK_ACT_IDENTITY, &s));
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    CHECK_EQ(turbo_buffer_alloc_counter(), 0u);
    CHECK(s > 0.0f);
    turborerank_buffer_free(buf);

    // Dual rent of caller buffers after warmup: zero allocs.
    turborerank_buffer *warm1 = nullptr;
    turborerank_buffer *warm2 = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 1, 64, &warm1));
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 1, 64, &warm2));
    CHECK(warm1->input_ids != warm2->input_ids);
    turborerank_buffer_free(warm1);
    turborerank_buffer_free(warm2);
    turborerank::alloc_counter_reset();
    turborerank_buffer *b1 = nullptr;
    turborerank_buffer *b2 = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 1, 64, &b1));
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 1, 64, &b2));
    CHECK(b1->input_ids != b2->input_ids);
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    turborerank_buffer_free(b1);
    turborerank_buffer_free(b2);

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
        std::fprintf(stderr, "FAIL CUDA MiniLM CE: weights missing (make fetch-rerankers)\n");
        CHECK(weights_present());
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
    CHECK(e->cuda.cublaslt != nullptr);
    CHECK(std::strcmp(turborerank::impl::cuda_gemm_backend(), "cublasLtMatmul") == 0);
    CHECK(e->arena != nullptr);
    CHECK(e->cuda.arena == e->arena);
    CHECK(e->cuda.n_rented >= 11u);
    if (e->cuda.lt_workspace_bytes > 0) {
        CHECK(e->cuda.lt_workspace != nullptr);
        CHECK(turbo_buffer_arena_owns(e->arena, e->cuda.lt_workspace));
    }
    CHECK(turbo_buffer_arena_owns(e->arena, e->cuda.x));
    CHECK(turbo_buffer_arena_owns(e->arena, e->cuda.residual));
    CHECK(turbo_buffer_arena_owns(e->arena, e->cuda.q));
    CHECK(turbo_buffer_arena_owns(e->arena, e->cuda.attn));
    CHECK(turbo_buffer_arena_owns(e->arena, e->work->input_ids));
    CHECK(turbo_buffer_arena_owns(e->arena, e->work->attention_mask));
    void *mapped_ids = nullptr;
    void *mapped_mask = nullptr;
    CHECK_EQ(turbo_buffer_cuda_mapped_device_ptr(e->work->input_ids, &mapped_ids), 1);
    CHECK_EQ(
        turbo_buffer_cuda_mapped_device_ptr(e->work->attention_mask, &mapped_mask), 1
    );
    CHECK(mapped_ids != nullptr);
    CHECK(mapped_mask != nullptr);
    {
        float refuse_logit = 0;
        std::string refuse_err;
        int32_t unmapped[8] = {101, 7592, 102, 0, 0, 0, 0, 0};
        CHECK(!turborerank::impl::bert_forward_row_cuda(
            &e->cuda,
            e->cfg,
            unmapped,
            unmapped,
            unmapped,
            unmapped,
            4,
            &refuse_logit,
            &refuse_err
        ));
        CHECK(refuse_err.find("PINNED mapped") != std::string::npos);
        CHECK(refuse_err.find("refusing per-forward H2D") != std::string::npos);
    }
    for (uint32_t i = 0; i < e->cuda.n_rented; ++i) {
        CHECK_EQ(e->cuda.rented[i].placement, TURBO_BUFFER_PLACE_DEVICE);
        CHECK(turbo_buffer_arena_owns(e->arena, e->cuda.rented[i].ptr));
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
    CHECK(logits[0] != logits[1] || logits[1] != logits[2]);
    CHECK(logits[0] > logits[1]);
    CHECK(logits[1] > logits[2]);
    CHECK(logits[0] - logits[2] > 2.0f);
    CHECK(almost(logits[0], 8.84585285f, 2e-3f));
    CHECK(almost(logits[1], -4.32007599f, 2e-3f));
    CHECK(almost(logits[2], -11.27389431f, 2e-3f));

    turborerank::alloc_counter_reset();
    turbo_buffer_cuda_forward_allocs_reset();
    turbo_buffer_cuda_forward_h2d_reset();
#ifdef TURBORERANK_CUDA
    size_t mem_free0 = 0;
    size_t mem_total0 = 0;
    CHECK(cudaMemGetInfo(&mem_free0, &mem_total0) == cudaSuccess);
#endif
    float logits_steady[3] = {0, 0, 0};
    CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 3, &opts, logits_steady));
#ifdef TURBORERANK_CUDA
    size_t mem_free1 = 0;
    size_t mem_total1 = 0;
    CHECK(cudaMemGetInfo(&mem_free1, &mem_total1) == cudaSuccess);
    CHECK_EQ(mem_total1, mem_total0);
    CHECK(mem_free1 >= mem_free0);
    std::fprintf(
        stderr,
        "CUDA id H2D bytes/calls=%llu/%llu device used %zu -> %zu (delta %ld)\n",
        static_cast<unsigned long long>(turbo_buffer_cuda_forward_h2d_bytes()),
        static_cast<unsigned long long>(turbo_buffer_cuda_forward_h2d_calls()),
        mem_total0 - mem_free0,
        mem_total1 - mem_free1,
        static_cast<long>(mem_free0) - static_cast<long>(mem_free1)
    );
#endif
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    CHECK_EQ(turbo_buffer_alloc_counter(), 0u);
    CHECK_EQ(turbo_buffer_cuda_forward_allocs(), 0u);
    CHECK_EQ(turbo_buffer_cuda_forward_h2d_bytes(), 0u);
    CHECK_EQ(turbo_buffer_cuda_forward_h2d_calls(), 0u);
    CHECK(almost(logits_steady[0], logits[0], 1e-6f));
    CHECK(almost(logits_steady[1], logits[1], 1e-6f));
    CHECK(almost(logits_steady[2], logits[2], 1e-6f));

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
    turbo_buffer_cuda_forward_allocs_reset();
    turbo_buffer_cuda_forward_h2d_reset();
    float s = 0;
    CHECK_ST(turborerank_forward(e, buf, 1, TURBORERANK_ACT_IDENTITY, &s));
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    CHECK_EQ(turbo_buffer_cuda_forward_allocs(), 0u);
    CHECK_EQ(turbo_buffer_cuda_forward_h2d_bytes(), 0u);
    CHECK_EQ(turbo_buffer_cuda_forward_h2d_calls(), 0u);
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

#ifndef TURBORERANK_METAL
static void test_create_without_metal_fails() {
    CHECK(!turborerank::impl::metal_compiled());
    turborerank_engine *e = nullptr;
    CHECK(turborerank_engine_create(TURBORERANK_DEVICE_METAL, nullptr, &e) ==
          TURBORERANK_ERR_UNAVAILABLE);
    CHECK(e == nullptr);
    const char *msg = turborerank_last_error(nullptr);
    CHECK(std::strstr(msg, "Refusing CPU") != nullptr ||
          std::strstr(msg, "without Metal") != nullptr);
    turborerank_buffer *buf = nullptr;
    CHECK(turborerank_buffer_alloc(TURBORERANK_DEVICE_METAL, 1, 16, &buf) ==
          TURBORERANK_ERR_UNAVAILABLE);
    CHECK(buf == nullptr);
}
#endif

#ifndef TURBORERANK_OPENVINO
static void test_create_without_ov_fails() {
    CHECK(!turborerank::impl::ov_compiled());
    turborerank_engine *e = nullptr;
    CHECK(turborerank_engine_create(TURBORERANK_DEVICE_OPENVINO_GPU, nullptr, &e) ==
          TURBORERANK_ERR_UNAVAILABLE);
    CHECK(e == nullptr);
    const char *msg = turborerank_last_error(nullptr);
    CHECK(std::strstr(msg, "Refusing CPU") != nullptr ||
          std::strstr(msg, "without OpenVINO") != nullptr);
    CHECK(turborerank_engine_create(TURBORERANK_DEVICE_OPENVINO_CPU, nullptr, &e) ==
          TURBORERANK_ERR_NOT_IMPLEMENTED);
    turborerank_buffer *buf = nullptr;
    CHECK(turborerank_buffer_alloc(TURBORERANK_DEVICE_OPENVINO_GPU, 1, 16, &buf) ==
          TURBORERANK_ERR_UNAVAILABLE);
    CHECK(buf == nullptr);
}
#endif

static void test_ov_usm_buffer() {
    if (!ov_gpu_live()) {
        std::fprintf(stderr, "SKIP OV USM buffer (no GPU plugin)\n");
        return;
    }
    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_OPENVINO_GPU, 2, 16, &buf));
    CHECK(buf != nullptr);
    CHECK(buf->device == TURBORERANK_DEVICE_OPENVINO_GPU);
    auto aligned = [](const void *p) {
        return (reinterpret_cast<uintptr_t>(p) % 64u) == 0;
    };
    CHECK(aligned(buf->input_ids));
    CHECK(aligned(buf->attention_mask));
    turbo_buffer_placement place = TURBO_BUFFER_PLACE_HOST;
    CHECK_EQ(turbo_buffer_ze_query(buf->input_ids, &place), TURBO_BUFFER_OK);
    CHECK_EQ(place, TURBO_BUFFER_PLACE_SHARED);
    CHECK_EQ(turbo_buffer_ze_query(buf->attention_mask, &place), TURBO_BUFFER_OK);
    CHECK_EQ(place, TURBO_BUFFER_PLACE_SHARED);
    buf->input_ids[0] = 101;
    buf->input_ids[1] = 7592;
    CHECK_EQ(buf->input_ids[0], 101);
    CHECK_EQ(buf->input_ids[1], 7592);
    turborerank_buffer_free(buf);

    turborerank_buffer *warm1 = nullptr;
    turborerank_buffer *warm2 = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_OPENVINO_GPU, 2, 16, &warm1));
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_OPENVINO_GPU, 2, 16, &warm2));
    CHECK(warm1->input_ids != warm2->input_ids);
    turborerank_buffer_free(warm1);
    turborerank_buffer_free(warm2);
    turborerank::alloc_counter_reset();
    turborerank_buffer *b1 = nullptr;
    turborerank_buffer *b2 = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_OPENVINO_GPU, 2, 16, &b1));
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_OPENVINO_GPU, 2, 16, &b2));
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    CHECK_EQ(turbo_buffer_ze_query(b1->input_ids, &place), TURBO_BUFFER_OK);
    CHECK_EQ(place, TURBO_BUFFER_PLACE_SHARED);
    turborerank_buffer_free(b1);
    turborerank_buffer_free(b2);
}

static void test_ov_remote_usm_wrap_unavailable() {
    if (!ov_gpu_live()) {
        std::fprintf(stderr, "SKIP OV remote-USM wrap probe (no GPU plugin)\n");
        return;
    }
    std::string why;
    const bool live = turborerank::impl::ov_probe_remote_usm_wrap(&why);
    if (live) {
        std::fprintf(stderr, "OV remote-USM wrap LIVE: %s\n", why.c_str());
        CHECK(live);
        return;
    }
    CHECK(!live);
    CHECK(why.find("smaller size (0)") != std::string::npos ||
          why.find("USM") != std::string::npos ||
          why.find("shared") != std::string::npos);
    std::fprintf(stderr, "OV remote-USM wrap UNAVAILABLE: %s\n", why.c_str());
}

static void test_ov_real_model_scores(turborerank_device device) {
    if (device == TURBORERANK_DEVICE_OPENVINO_GPU && !ov_gpu_live()) {
        std::fprintf(stderr, "SKIP OV GPU MiniLM CE (no GPU plugin)\n");
        return;
    }
    if (device == TURBORERANK_DEVICE_OPENVINO_CPU && !ov_cpu_live()) {
        std::fprintf(stderr, "SKIP OV CPU MiniLM CE (no CPU plugin)\n");
        return;
    }
    if (!ov_ir_present()) {
        std::fprintf(stderr, "SKIP OV MiniLM CE scores (make convert-rerank-ov)\n");
        return;
    }
    turborerank_engine *e = nullptr;
    const std::string dir = ov_ir_dir();
    CHECK_ST(turborerank_engine_create(device, dir.c_str(), &e));
    CHECK(e->device == device);
    const turborerank_status load = turborerank_load_model(e, "ms-marco-minilm-l6", 0);
    CHECK_ST(load);
    if (load != TURBORERANK_OK) {
        std::fprintf(stderr, "OV load error: %s\n", turborerank_last_error(e));
        turborerank_engine_destroy(e);
        return;
    }
    CHECK(e->ov.enabled);
    CHECK(e->arena != nullptr);
    CHECK(e->work != nullptr);
    CHECK(turbo_buffer_arena_owns(e->arena, e->work->input_ids));
    CHECK(turbo_buffer_arena_owns(e->arena, e->work->attention_mask));
    if (device == TURBORERANK_DEVICE_OPENVINO_GPU) {
        turbo_buffer_placement place = TURBO_BUFFER_PLACE_HOST;
        CHECK_EQ(turbo_buffer_ze_query(e->work->input_ids, &place), TURBO_BUFFER_OK);
        CHECK_EQ(place, TURBO_BUFFER_PLACE_SHARED);
        CHECK_EQ(turbo_buffer_ze_query(e->work->attention_mask, &place), TURBO_BUFFER_OK);
        CHECK_EQ(place, TURBO_BUFFER_PLACE_SHARED);
        CHECK_EQ(turbo_buffer_ze_query(e->work->token_type_ids, &place), TURBO_BUFFER_OK);
        CHECK_EQ(place, TURBO_BUFFER_PLACE_SHARED);
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
        CHECK(almost(one[i], logits[i], 1e-4f));
    }

    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(device, 1, 64, &buf));
    CHECK(buf->device == device);
    CHECK_ST(turborerank_pack_text(
        e, buf, 0, query, docs[0], TURBORERANK_TRUNC_LONGEST_FIRST, 64
    ));
    turborerank::alloc_counter_reset();
    float s = 0;
    CHECK_ST(turborerank_forward(e, buf, 1, TURBORERANK_ACT_IDENTITY, &s));
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    CHECK_EQ(turbo_buffer_alloc_counter(), 0u);
    CHECK(almost(s, logits[0], 2e-3f));
    turborerank_buffer_free(buf);

    turborerank::alloc_counter_reset();
    float score_again[3] = {0, 0, 0};
    CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 3, &opts, score_again));
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    CHECK(almost(score_again[0], logits[0], 1e-4f));

    if (device == TURBORERANK_DEVICE_OPENVINO_GPU) {
        turborerank_buffer *cpu_buf = nullptr;
        CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 1, 64, &cpu_buf));
        CHECK_ST(turborerank_pack_text(
            e, cpu_buf, 0, query, docs[0], TURBORERANK_TRUNC_LONGEST_FIRST, 64
        ));
        float bad = 0;
        CHECK(turborerank_forward(e, cpu_buf, 1, TURBORERANK_ACT_IDENTITY, &bad) ==
              TURBORERANK_ERR_INTERNAL);
        CHECK(std::strstr(turborerank_last_error(e), "SHARED") != nullptr ||
              std::strstr(turborerank_last_error(e), "refus") != nullptr);
        turborerank_buffer_free(cpu_buf);
    }

    std::fprintf(
        stderr,
        "OV %s Berlin logits: %.8f %.8f %.8f (abs err %.3e %.3e %.3e)\n",
        turborerank_device_name(device),
        logits[0],
        logits[1],
        logits[2],
        std::fabs(logits[0] - 8.84585285f),
        std::fabs(logits[1] + 4.32007599f),
        std::fabs(logits[2] + 11.27389431f)
    );
    turborerank_engine_destroy(e);
}

static void test_metal_shared_buffer() {
    if (!metal_live()) {
        std::fprintf(stderr, "SKIP Metal shared buffer (no MTL GPU)\n");
        return;
    }
    turborerank_buffer *buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_METAL, 2, 16, &buf));
    CHECK(buf != nullptr);
    CHECK(buf->device == TURBORERANK_DEVICE_METAL);
    auto aligned = [](const void *p) {
        return (reinterpret_cast<uintptr_t>(p) % 64u) == 0;
    };
    CHECK(aligned(buf->input_ids));
    CHECK(aligned(buf->attention_mask));
    CHECK(turborerank::impl::metal_shared_owns(buf->input_ids));
    CHECK(turbo_buffer_metal_owns(buf->input_ids));
    CHECK(turbo_buffer_metal_owns(buf->attention_mask));
    CHECK(turbo_buffer_metal_owns(buf->token_type_ids));
    CHECK(turbo_buffer_metal_owns(buf->position_ids));
    void *native = nullptr;
    size_t off = 0;
    CHECK(turbo_buffer_metal_lookup(buf->input_ids, &native, &off) == 1);
    CHECK(native != nullptr);
    buf->input_ids[0] = 101;
    buf->input_ids[1] = 7592;
    CHECK_EQ(buf->input_ids[0], 101);
    CHECK_EQ(buf->input_ids[1], 7592);
    turborerank::alloc_counter_reset();
    buf->input_ids[2] = 102;
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    turborerank_buffer_free(buf);

    turborerank_buffer *w1 = nullptr;
    turborerank_buffer *w2 = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_METAL, 2, 16, &w1));
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_METAL, 2, 16, &w2));
    CHECK(w1->input_ids != w2->input_ids);
    turborerank_buffer_free(w1);
    turborerank_buffer_free(w2);
    turborerank::alloc_counter_reset();
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_METAL, 2, 16, &w1));
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_METAL, 2, 16, &w2));
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    CHECK(turbo_buffer_metal_owns(w1->input_ids));
    CHECK(turbo_buffer_metal_owns(w2->input_ids));
    turborerank_buffer_free(w1);
    turborerank_buffer_free(w2);

    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_AUTO, 1, 16, &buf));
    CHECK(buf->device == TURBORERANK_DEVICE_METAL);
    CHECK(turbo_buffer_metal_owns(buf->input_ids));
    turborerank_buffer_free(buf);
}

static void test_metal_real_model_scores() {
    if (!metal_live()) {
        std::fprintf(stderr, "SKIP Metal MiniLM CE (no MTL GPU)\n");
        return;
    }
    if (!weights_present()) {
        std::fprintf(stderr, "SKIP Metal MiniLM CE scores (make fetch-rerankers)\n");
        return;
    }
    turborerank_engine *e = nullptr;
    const std::string dir = model_dir();
    CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_METAL, dir.c_str(), &e));
    CHECK(e->device == TURBORERANK_DEVICE_METAL);
    const turborerank_status load = turborerank_load_model(e, "ms-marco-minilm-l6", 0);
    CHECK_ST(load);
    if (load != TURBORERANK_OK) {
        std::fprintf(stderr, "Metal load error: %s\n", turborerank_last_error(e));
        turborerank_engine_destroy(e);
        return;
    }
    CHECK(e->metal.enabled);
    CHECK(e->arena != nullptr);
    CHECK(e->work != nullptr);
    CHECK(turbo_buffer_arena_owns(e->arena, e->work->input_ids));
    CHECK(turbo_buffer_arena_owns(e->arena, e->work->attention_mask));
    CHECK(turbo_buffer_arena_owns(e->arena, e->work->token_type_ids));
    CHECK(turbo_buffer_metal_owns(e->work->input_ids));
    CHECK(turbo_buffer_metal_owns(e->work->attention_mask));

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

    turborerank::alloc_counter_reset();
    float score_again[3] = {0, 0, 0};
    CHECK_ST(turborerank_score(e, nullptr, 0, query, docs, 3, &opts, score_again));
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    CHECK_EQ(turbo_buffer_alloc_counter(), 0u);
    CHECK(almost(score_again[0], logits[0], 1e-5f));

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
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_METAL, 1, 64, &buf));
    CHECK(buf->device == TURBORERANK_DEVICE_METAL);
    CHECK(turbo_buffer_metal_owns(buf->input_ids));
    CHECK_ST(turborerank_pack_text(
        e, buf, 0, query, docs[0], TURBORERANK_TRUNC_LONGEST_FIRST, 64
    ));
    turborerank::alloc_counter_reset();
    float s = 0;
    CHECK_ST(turborerank_forward(e, buf, 1, TURBORERANK_ACT_IDENTITY, &s));
    CHECK_EQ(turborerank::alloc_counter_value(), 0u);
    CHECK_EQ(turbo_buffer_alloc_counter(), 0u);
    CHECK(almost(s, logits[0], 2e-3f));
    turborerank_buffer_free(buf);

    turborerank_buffer *cpu_buf = nullptr;
    CHECK_ST(turborerank_buffer_alloc(TURBORERANK_DEVICE_CPU, 1, 64, &cpu_buf));
    CHECK(!turbo_buffer_metal_owns(cpu_buf->input_ids));
    CHECK_ST(turborerank_pack_text(
        e, cpu_buf, 0, query, docs[0], TURBORERANK_TRUNC_LONGEST_FIRST, 64
    ));
    float bad = 0;
    CHECK(turborerank_forward(e, cpu_buf, 1, TURBORERANK_ACT_IDENTITY, &bad) ==
          TURBORERANK_ERR_INTERNAL);
    CHECK(std::strstr(turborerank_last_error(e), "SHARED") != nullptr ||
          std::strstr(turborerank_last_error(e), "refus") != nullptr);
    turborerank_buffer_free(cpu_buf);

    turborerank_engine *auto_e = nullptr;
    CHECK_ST(turborerank_engine_create(TURBORERANK_DEVICE_AUTO, dir.c_str(), &auto_e));
    CHECK(auto_e->device == TURBORERANK_DEVICE_METAL);
    CHECK_ST(turborerank_load_model(auto_e, "ms-marco-minilm-l6", 0));
    opts.activation = TURBORERANK_ACT_IDENTITY;
    float auto_logit = 0;
    CHECK_ST(turborerank_score(auto_e, nullptr, 0, query, &docs[0], 1, &opts, &auto_logit));
    CHECK(almost(auto_logit, logits[0], 2e-3f));
    turborerank_engine_destroy(auto_e);

    std::fprintf(
        stderr,
        "Metal Berlin logits: %.8f %.8f %.8f (abs err %.3e %.3e %.3e)\n",
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
    test_cuda_gemm_stack_is_cublaslt();
    test_pack_cls_sep_and_types();
    test_pack_empty_sides();
    test_pack_truncation();
    test_device_create_policy();
#ifndef TURBORERANK_CUDA
    test_create_without_cuda_fails();
#endif
#ifndef TURBORERANK_OPENVINO
    test_create_without_ov_fails();
#endif
#ifndef TURBORERANK_METAL
    test_create_without_metal_fails();
#endif
    test_missing_weights_fails_loud();
    test_wordpiece_fixture();
    test_real_model_scores();
    test_cuda_real_model_scores();
    test_ov_usm_buffer();
    test_ov_remote_usm_wrap_unavailable();
    test_ov_real_model_scores(TURBORERANK_DEVICE_OPENVINO_GPU);
    test_ov_real_model_scores(TURBORERANK_DEVICE_OPENVINO_CPU);
    test_metal_shared_buffer();
    test_metal_real_model_scores();

    std::fprintf(
        stderr,
        "turborerank_tests: %d passed, %d failed\n",
        g_passes,
        g_fails
    );
    return g_fails == 0 ? 0 : 1;
}
