// SPDX-License-Identifier: Apache-2.0
//
// TurboEmbed mock path rents host FP32 rows from turbo_buffer.
// Steady-state embed must not increment the alloc counter.

#include "turbo_buffer.h"
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

int main() {
    turboembed_engine *e = nullptr;
    CHECK_ST(turboembed_engine_create(TURBOEMBED_DEVICE_MOCK, nullptr, &e));
    CHECK_ST(turboembed_load_model(e, "mock-embed", 0));

    const char hello[] = "hello world";
    turboembed_str texts[2] = {
        {hello, sizeof(hello) - 1},
        {hello, sizeof(hello) - 1},
    };
    turboembed_embed_result *r1 = nullptr;
    CHECK_ST(turboembed_embed(e, "mock-embed", 0, texts, 2, nullptr, &r1));
    CHECK(r1 != nullptr);
    CHECK(r1->dim == 8);
    CHECK(r1->count == 2);
    CHECK(r1->values != nullptr);
    const float first = r1->values[0];
    turboembed_embed_result_free(r1);

    turbo_buffer_alloc_counter_reset();
    turboembed_embed_result *r2 = nullptr;
    CHECK_ST(turboembed_embed(e, "mock-embed", 0, texts, 2, nullptr, &r2));
    CHECK(turbo_buffer_alloc_counter() == 0u);
    CHECK(r2 != nullptr);
    CHECK(r2->values[0] == first);
    turboembed_embed_result_free(r2);

    turboembed_engine_destroy(e);
    std::fprintf(
        stderr, "mock_arena_tests: %d passed, %d failed\n", g_passes, g_fails
    );
    return g_fails == 0 ? 0 : 1;
}
