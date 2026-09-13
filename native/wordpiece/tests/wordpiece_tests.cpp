// SPDX-License-Identifier: Apache-2.0
//
// SOLIDIFY (5): frozen vocab + write-through. No heap token staging.

#include "wordpiece.h"
#include "turbo_buffer.h"

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
            ++g_passes;                                                          \
        }                                                                        \
    } while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

static std::string workspace_root() {
    if (const char *e = std::getenv("INFERSTREAM_ROOT")) {
        return e;
    }
    return ".";
}

static bool file_exists(const std::string &p) {
    std::ifstream in(p);
    return static_cast<bool>(in);
}

static void test_tiny_fixture() {
    const std::string dir = workspace_root() + "/testdata/turborerank/tiny-vocab";
    if (!file_exists(dir + "/vocab.txt")) {
        std::fprintf(stderr, "SKIP tiny vocab\n");
        return;
    }
    wordpiece_vocab *v = nullptr;
    CHECK_EQ(wordpiece_vocab_load((dir + "/vocab.txt").c_str(), &v), WORDPIECE_OK);
    CHECK(v != nullptr);
    CHECK(wordpiece_vocab_is_loaded(v));
    CHECK_EQ(wordpiece_cls_id(v), 2);
    CHECK_EQ(wordpiece_sep_id(v), 3);
    CHECK_EQ(wordpiece_pad_id(v), 0);

    int32_t ids[16];
    size_t n = 0;
    const char hello[] = "hello world";
    CHECK_EQ(wordpiece_tokenize(v, hello, sizeof(hello) - 1, ids, 16, 4, &n), WORDPIECE_OK);
    CHECK_EQ(n, 2u);
    CHECK_EQ(ids[0], 4);
    CHECK_EQ(ids[1], 5);

    const char ing[] = "helloing";
    CHECK_EQ(wordpiece_tokenize(v, ing, sizeof(ing) - 1, ids, 16, 4, &n), WORDPIECE_OK);
    CHECK_EQ(n, 2u);
    CHECK_EQ(ids[0], 4);
    CHECK_EQ(ids[1], 6);

    wordpiece_hot_alloc_counter_reset();
    int32_t row_ids[8];
    int32_t mask[8];
    int32_t types[8];
    int32_t pos[8];
    CHECK_EQ(
        wordpiece_encode_sentence(
            v, hello, sizeof(hello) - 1, row_ids, mask, types, pos, 8, 8, 4
        ),
        WORDPIECE_OK
    );
    CHECK_EQ(wordpiece_hot_alloc_counter(), 0u);
    CHECK_EQ(row_ids[0], 2); // CLS
    CHECK_EQ(row_ids[1], 4);
    CHECK_EQ(row_ids[2], 5);
    CHECK_EQ(row_ids[3], 3); // SEP
    CHECK_EQ(row_ids[4], 0);
    CHECK_EQ(mask[0], 1);
    CHECK_EQ(mask[3], 1);
    CHECK_EQ(mask[4], 0);
    CHECK_EQ(types[1], 0);

    CHECK_EQ(
        wordpiece_pack_pair(
            v,
            hello,
            sizeof(hello) - 1,
            ing,
            sizeof(ing) - 1,
            row_ids,
            mask,
            types,
            pos,
            8,
            8,
            4,
            WORDPIECE_TRUNC_LONGEST_FIRST,
            8
        ),
        WORDPIECE_OK
    );
    CHECK_EQ(wordpiece_hot_alloc_counter(), 0u);
    CHECK_EQ(row_ids[0], 2);
    CHECK_EQ(row_ids[1], 4);
    CHECK_EQ(row_ids[2], 5);
    CHECK_EQ(row_ids[3], 3);
    CHECK_EQ(row_ids[4], 4);
    CHECK_EQ(row_ids[5], 6);
    CHECK_EQ(row_ids[6], 3);
    CHECK_EQ(types[4], 1);
    CHECK_EQ(types[5], 1);
    CHECK_EQ(types[6], 1);

    wordpiece_vocab_destroy(v);
}

static void test_arena_write_through() {
    const std::string dir = workspace_root() + "/testdata/turborerank/tiny-vocab";
    if (!file_exists(dir + "/vocab.txt")) {
        return;
    }
    wordpiece_vocab *v = nullptr;
    CHECK_EQ(wordpiece_vocab_load_dir(dir.c_str(), &v), WORDPIECE_OK);

    turbo_buffer_arena *arena = nullptr;
    CHECK_EQ(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &arena), TURBO_BUFFER_OK);
    turbo_buffer_view ids{};
    turbo_buffer_view mask{};
    turbo_buffer_view types{};
    turbo_buffer_view pos{};
    CHECK_EQ(
        turbo_buffer_arena_rent(
            arena, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 1, 16, 16, &ids
        ),
        TURBO_BUFFER_OK
    );
    CHECK_EQ(
        turbo_buffer_arena_rent(
            arena, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 1, 16, 16, &mask
        ),
        TURBO_BUFFER_OK
    );
    CHECK_EQ(
        turbo_buffer_arena_rent(
            arena, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 1, 16, 16, &types
        ),
        TURBO_BUFFER_OK
    );
    CHECK_EQ(
        turbo_buffer_arena_rent(
            arena, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 1, 16, 16, &pos
        ),
        TURBO_BUFFER_OK
    );
    CHECK(turbo_buffer_arena_owns(arena, ids.ptr));
    CHECK(turbo_buffer_arena_owns(arena, mask.ptr));

    turbo_buffer_alloc_counter_reset();
    wordpiece_hot_alloc_counter_reset();
    const char q[] = "hello";
    const char d[] = "world";
    CHECK_EQ(
        wordpiece_pack_pair(
            v,
            q,
            sizeof(q) - 1,
            d,
            sizeof(d) - 1,
            ids.ptr,
            mask.ptr,
            types.ptr,
            pos.ptr,
            16,
            16,
            4,
            WORDPIECE_TRUNC_LONGEST_FIRST,
            16
        ),
        WORDPIECE_OK
    );
    CHECK_EQ(turbo_buffer_alloc_counter(), 0u);
    CHECK_EQ(wordpiece_hot_alloc_counter(), 0u);
    CHECK_EQ(turbo_buffer_view_i32(&ids)[0], 2);
    CHECK_EQ(turbo_buffer_view_i32(&ids)[1], 4);
    CHECK_EQ(turbo_buffer_view_i32(&ids)[3], 5);

    (void)turbo_buffer_arena_return(arena, &ids);
    (void)turbo_buffer_arena_return(arena, &mask);
    (void)turbo_buffer_arena_return(arena, &types);
    (void)turbo_buffer_arena_return(arena, &pos);
    turbo_buffer_arena_destroy(arena);
    wordpiece_vocab_destroy(v);
}

static void test_minilm_json_if_present() {
    const std::string js = workspace_root() + "/models/onnx/minilm/tokenizer.json";
    const std::string txt =
        workspace_root() + "/models/rerank/ms-marco-minilm-l6/vocab.txt";
    wordpiece_vocab *v = nullptr;
    if (file_exists(js)) {
        CHECK_EQ(wordpiece_vocab_load(js.c_str(), &v), WORDPIECE_OK);
    } else if (file_exists(txt)) {
        CHECK_EQ(wordpiece_vocab_load(txt.c_str(), &v), WORDPIECE_OK);
    } else {
        std::fprintf(stderr, "SKIP MiniLM vocab (not fetched)\n");
        return;
    }
    CHECK(v != nullptr);
    CHECK_EQ(wordpiece_cls_id(v), 101);
    CHECK_EQ(wordpiece_sep_id(v), 102);
    CHECK_EQ(wordpiece_pad_id(v), 0);
    CHECK_EQ(wordpiece_unk_id(v), 100);

    int32_t ids[32];
    int32_t mask[32];
    int32_t types[32];
    int32_t pos[32];
    wordpiece_hot_alloc_counter_reset();
    const char hello[] = "hello world";
    CHECK_EQ(
        wordpiece_encode_sentence(
            v, hello, sizeof(hello) - 1, ids, mask, types, pos, 16, 16, 4
        ),
        WORDPIECE_OK
    );
    CHECK_EQ(wordpiece_hot_alloc_counter(), 0u);
    CHECK_EQ(ids[0], 101);
    CHECK_EQ(ids[1], 7592); // hello
    CHECK_EQ(ids[2], 2088); // world
    CHECK_EQ(ids[3], 102);
    CHECK_EQ(mask[3], 1);
    CHECK_EQ(mask[4], 0);
    wordpiece_vocab_destroy(v);
}

static void test_tokenizer_json_matches_txt() {
    const std::string txt =
        workspace_root() + "/models/rerank/ms-marco-minilm-l6/vocab.txt";
    const std::string js =
        workspace_root() + "/models/rerank/ms-marco-minilm-l6/tokenizer.json";
    if (!file_exists(txt) || !file_exists(js)) {
        return;
    }
    wordpiece_vocab *a = nullptr;
    wordpiece_vocab *b = nullptr;
    CHECK_EQ(wordpiece_vocab_load(txt.c_str(), &a), WORDPIECE_OK);
    CHECK_EQ(wordpiece_vocab_load(js.c_str(), &b), WORDPIECE_OK);
    const char hello[] = "hello world";
    int32_t ia[16];
    int32_t ib[16];
    int32_t ma[16];
    int32_t mb[16];
    int32_t ta[16];
    int32_t tb[16];
    CHECK_EQ(
        wordpiece_encode_sentence(a, hello, sizeof(hello) - 1, ia, ma, ta, nullptr, 16, 16, 4),
        WORDPIECE_OK
    );
    CHECK_EQ(
        wordpiece_encode_sentence(b, hello, sizeof(hello) - 1, ib, mb, tb, nullptr, 16, 16, 4),
        WORDPIECE_OK
    );
    for (int i = 0; i < 8; ++i) {
        CHECK_EQ(ia[i], ib[i]);
        CHECK_EQ(ma[i], mb[i]);
    }
    wordpiece_vocab_destroy(a);
    wordpiece_vocab_destroy(b);
}

int main() {
    test_tiny_fixture();
    test_arena_write_through();
    test_minilm_json_if_present();
    test_tokenizer_json_matches_txt();
    std::fprintf(stderr, "wordpiece_tests: %d pass, %d fail\n", g_passes, g_fails);
    return g_fails == 0 ? 0 : 1;
}
