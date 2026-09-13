// SPDX-License-Identifier: Apache-2.0
//
// SOLIDIFY (4) Machine B: GenAI token/result slots are arena USM.
// Steady-state embed must not increment turbo_buffer_alloc_counter.
// GPU tokens query as SHARED. No silent CPU arena.

#include "turbo_buffer.h"
#include "turboembed.h"
#include "wordpiece.h"

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

extern "C" turboembed_status turboembed_test_genai_arena_info(
    const turboembed_engine *engine,
    uint32_t *arena_device,
    uint32_t *token_placement,
    const void **token_ids,
    const void **hidden,
    int *owns_tokens,
    int *owns_hidden,
    int *hidden_used_arena
);

static void prove_device(turboembed_device device, turbo_buffer_placement want_place) {
    turboembed_engine *e = nullptr;
    CHECK_ST(turboembed_engine_create(device, nullptr, &e));
    CHECK(e != nullptr);
    CHECK_ST(turboembed_load_model(e, "minilm", 0));

    const char hello[] = "hello world";
    turboembed_embed_result *r1 = nullptr;
    CHECK_ST(turboembed_embed_one(e, "minilm", 0, hello, sizeof(hello) - 1, nullptr, &r1));
    CHECK(r1 != nullptr);
    CHECK(r1->dim == 384);
    CHECK(r1->count == 1);
    CHECK(r1->values != nullptr);

    uint32_t arena_dev = 0;
    uint32_t token_place = 0;
    const void *token_ids = nullptr;
    const void *hidden = nullptr;
    int owns_tokens = 0;
    int owns_hidden = 0;
    int hidden_arena = 0;
    CHECK_ST(turboembed_test_genai_arena_info(
        e,
        &arena_dev,
        &token_place,
        &token_ids,
        &hidden,
        &owns_tokens,
        &owns_hidden,
        &hidden_arena
    ));
    CHECK(arena_dev == TURBO_BUFFER_DEVICE_ZE);
    CHECK(token_place == static_cast<uint32_t>(want_place));
    CHECK(owns_tokens == 1);
    CHECK(owns_hidden == 1);
    CHECK(token_ids != nullptr);

    turbo_buffer_placement q = TURBO_BUFFER_PLACE_HOST;
    CHECK(turbo_buffer_ze_query(token_ids, &q) == TURBO_BUFFER_OK);
    CHECK(q == want_place);
    CHECK(turbo_buffer_ze_query(r1->values, &q) == TURBO_BUFFER_OK);
    CHECK(q == want_place);

    turboembed_embed_result_free(r1);

    turbo_buffer_alloc_counter_reset();
    wordpiece_hot_alloc_counter_reset();
    turboembed_embed_result *r2 = nullptr;
    CHECK_ST(turboembed_embed_one(e, "minilm", 0, hello, sizeof(hello) - 1, nullptr, &r2));
    CHECK(turbo_buffer_alloc_counter() == 0u);
    CHECK(wordpiece_hot_alloc_counter() == 0u);
    CHECK(r2 != nullptr);
    CHECK(r2->dim == 384);
    turboembed_embed_result_free(r2);
    turboembed_engine_destroy(e);
}

int main() {
    prove_device(TURBOEMBED_DEVICE_OPENVINO_GPU, TURBO_BUFFER_PLACE_SHARED);
    prove_device(TURBOEMBED_DEVICE_OPENVINO_CPU, TURBO_BUFFER_PLACE_HOST);

    turboembed_engine *gpu = nullptr;
    /* GPU create must not open a CPU arena when ZE SHARED is the contract.
     * Missing GPU still fails loud (existing policy). */
    const turboembed_status gst =
        turboembed_engine_create(TURBOEMBED_DEVICE_OPENVINO_GPU, nullptr, &gpu);
    if (gst == TURBOEMBED_OK && gpu != nullptr) {
        uint32_t arena_dev = 0;
        CHECK_ST(turboembed_load_model(gpu, "minilm", 0));
        CHECK_ST(turboembed_test_genai_arena_info(
            gpu, &arena_dev, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
        ));
        CHECK(arena_dev == TURBO_BUFFER_DEVICE_ZE);
        turboembed_engine_destroy(gpu);
    }

    std::fprintf(
        stderr, "genai_arena_tests: %d passed, %d failed\n", g_passes, g_fails
    );
    return g_fails == 0 ? 0 : 1;
}
