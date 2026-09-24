// SPDX-License-Identifier: Apache-2.0
//
// Arena ABI edge-case tests. CPU device only. Complements
// turbo_buffer_tests.cpp with rent/return cycle reuse, double-return
// zeroing, arena_owns boundaries, 64-byte row padding, explicit stride,
// invalid enum rejection, degenerate shapes, raw_alloc round-trips, and
// the name/error accessors. No GPU backend is touched here.

#include "turbo_buffer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
#define CHECK_ST(st) CHECK((st) == TURBO_BUFFER_OK)

static bool aligned64(const void *p) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) % 64u) == 0;
}

static bool name_is(turbo_buffer_status s, const char *expected) {
    const char *n = turbo_buffer_status_name(s);
    return n != nullptr && std::strcmp(n, expected) == 0;
}

static bool name_is(turbo_buffer_device d, const char *expected) {
    const char *n = turbo_buffer_device_name(d);
    return n != nullptr && std::strcmp(n, expected) == 0;
}

static bool name_is(turbo_buffer_placement p, const char *expected) {
    const char *n = turbo_buffer_placement_name(p);
    return n != nullptr && std::strcmp(n, expected) == 0;
}

static void test_abi_version_and_names() {
    CHECK_EQ(turbo_buffer_abi_version(), 1u);
    CHECK_EQ(turbo_buffer_abi_version(), TURBO_BUFFER_ABI_VERSION);

    CHECK(name_is(TURBO_BUFFER_OK, "OK"));
    CHECK(name_is(TURBO_BUFFER_ERR_INVALID_ARGUMENT, "INVALID_ARGUMENT"));
    CHECK(name_is(TURBO_BUFFER_ERR_NOT_FOUND, "NOT_FOUND"));
    CHECK(name_is(TURBO_BUFFER_ERR_NOT_IMPLEMENTED, "NOT_IMPLEMENTED"));
    CHECK(name_is(TURBO_BUFFER_ERR_UNAVAILABLE, "UNAVAILABLE"));
    CHECK(name_is(TURBO_BUFFER_ERR_INTERNAL, "INTERNAL"));
    CHECK(name_is(TURBO_BUFFER_ERR_OUT_OF_MEMORY, "OUT_OF_MEMORY"));
    CHECK(name_is(TURBO_BUFFER_ERR_UNSUPPORTED_DEVICE, "UNSUPPORTED_DEVICE"));
    CHECK(name_is(TURBO_BUFFER_ERR_DOUBLE_FREE, "DOUBLE_FREE"));

    CHECK(name_is(TURBO_BUFFER_DEVICE_CPU, "CPU"));
    CHECK(name_is(TURBO_BUFFER_DEVICE_CUDA, "CUDA"));
    CHECK(name_is(TURBO_BUFFER_DEVICE_ZE, "ZE"));
    CHECK(name_is(TURBO_BUFFER_DEVICE_METAL, "METAL"));

    CHECK(name_is(TURBO_BUFFER_PLACE_HOST, "HOST"));
    CHECK(name_is(TURBO_BUFFER_PLACE_DEVICE, "DEVICE"));
    CHECK(name_is(TURBO_BUFFER_PLACE_SHARED, "SHARED"));
    CHECK(name_is(TURBO_BUFFER_PLACE_PINNED, "PINNED"));

    // Out-of-range enum values must still produce a non-null name.
    CHECK(name_is(static_cast<turbo_buffer_status>(999), "UNKNOWN"));
    CHECK(name_is(static_cast<turbo_buffer_device>(999), "UNKNOWN"));
    CHECK(name_is(static_cast<turbo_buffer_placement>(999), "UNKNOWN"));
}

static void test_last_error_never_null() {
    // No arena, no prior failure: the thread-local slot is still non-null.
    const char *m = turbo_buffer_last_error(nullptr);
    CHECK(m != nullptr);

    // Failed create of an unknown device records a TLS message, not NULL.
    turbo_buffer_arena *a = nullptr;
    CHECK_EQ(
        turbo_buffer_arena_create(static_cast<turbo_buffer_device>(999), &a),
        TURBO_BUFFER_ERR_UNSUPPORTED_DEVICE
    );
    CHECK(a == nullptr);
    m = turbo_buffer_last_error(nullptr);
    CHECK(m != nullptr);
    CHECK(std::strstr(m, "unknown") != nullptr);

    // Arena-scoped error survives and is non-null; success clears to "".
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &a));
    turbo_buffer_view v {};
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 0, 8, 8, &v
        ),
        TURBO_BUFFER_ERR_INVALID_ARGUMENT
    );
    CHECK(turbo_buffer_last_error(a) != nullptr);
    CHECK(std::strstr(turbo_buffer_last_error(a), "dtype/rows/cols") != nullptr);

    turbo_buffer_view ok {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 1, 8, 8, &ok
    ));
    // Cleared on success: still non-null, now empty.
    CHECK(turbo_buffer_last_error(a) != nullptr);
    CHECK(turbo_buffer_last_error(a)[0] == '\0');
    CHECK_ST(turbo_buffer_arena_return(a, &ok));
    turbo_buffer_arena_destroy(a);
}

static void test_rent_return_100_cycles_zero_alloc() {
    turbo_buffer_arena *a = nullptr;
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &a));

    // Warmup sizes the slab; steady state must not allocate again.
    turbo_buffer_view warm {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_PLACE_HOST, 2, 8, 8, &warm
    ));
    void *backing = warm.ptr;
    CHECK(backing != nullptr);
    CHECK_ST(turbo_buffer_arena_return(a, &warm));

    turbo_buffer_alloc_counter_reset();
    for (int i = 0; i < 100; ++i) {
        turbo_buffer_view v {};
        CHECK_ST(turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_PLACE_HOST, 2, 8, 8, &v
        ));
        // Same backing reused every cycle — no new slab, no drift.
        CHECK(v.ptr == backing);
        CHECK(v.handle != 0u);
        turbo_buffer_view_f32(&v)[0] = static_cast<float>(i);
        CHECK_EQ(turbo_buffer_view_f32(&v)[0], static_cast<float>(i));
        CHECK_ST(turbo_buffer_arena_return(a, &v));
        CHECK(v.ptr == nullptr);
    }
    CHECK_EQ(turbo_buffer_alloc_counter(), 0u);

    turbo_buffer_arena_destroy(a);
}

static void test_double_return_contract() {
    turbo_buffer_arena *a = nullptr;
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &a));

    turbo_buffer_view v {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 1, 8, 8, &v
    ));
    turbo_buffer_view copy = v;

    // Documented contract: a successful return zeros *view so a second
    // return of the same view cannot silently succeed.
    CHECK_ST(turbo_buffer_arena_return(a, &v));
    CHECK(v.ptr == nullptr);
    CHECK_EQ(v.handle, 0u);
    CHECK_EQ(v.rows, 0u);
    CHECK_EQ(v.cols, 0u);
    CHECK_EQ(v.row_stride, 0u);
    CHECK_EQ(v.dtype, static_cast<turbo_buffer_dtype>(0));
    CHECK_EQ(v.device, static_cast<turbo_buffer_device>(0));
    CHECK_EQ(v.placement, static_cast<turbo_buffer_placement>(0));

    // The zeroed view now fails as an empty view, not a double-free.
    CHECK_EQ(turbo_buffer_arena_return(a, &v), TURBO_BUFFER_ERR_INVALID_ARGUMENT);

    // The stale copy is detected as a double-return. Failure paths do not
    // zero the caller's view — the header promises zeroing on OK only.
    CHECK_EQ(turbo_buffer_arena_return(a, &copy), TURBO_BUFFER_ERR_DOUBLE_FREE);
    CHECK(copy.ptr != nullptr);
    CHECK(copy.handle != 0u);
    const char *msg = turbo_buffer_last_error(a);
    CHECK(msg != nullptr);
    CHECK(std::strstr(msg, "double-free") != nullptr);

    // The failure leaves arena state unchanged: a third attempt still
    // reports DOUBLE_FREE for the same stale copy.
    CHECK_EQ(turbo_buffer_arena_return(a, &copy), TURBO_BUFFER_ERR_DOUBLE_FREE);

    turbo_buffer_arena_destroy(a);
}

static void test_arena_owns_boundaries() {
    turbo_buffer_arena *a1 = nullptr;
    turbo_buffer_arena *a2 = nullptr;
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &a1));
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &a2));

    turbo_buffer_view v {};
    CHECK_ST(turbo_buffer_arena_rent(
        a1, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 2, 16, 16, &v
    ));
    // rows(2) * stride(16 elems) * 4B = 128 bytes exactly.
    char *base = static_cast<char *>(v.ptr);
    CHECK_EQ(turbo_buffer_arena_owns(a1, base), 1);
    CHECK_EQ(turbo_buffer_arena_owns(a1, base + 63), 1);
    CHECK_EQ(turbo_buffer_arena_owns(a1, base + 127), 1);
    CHECK_EQ(turbo_buffer_arena_owns(a1, base + 128), 0); // one past end
    CHECK_EQ(turbo_buffer_arena_owns(a1, nullptr), 0);

    int32_t stack_buf[4] = {0, 0, 0, 0};
    CHECK_EQ(turbo_buffer_arena_owns(a1, stack_buf), 0);

    void *heap = std::malloc(128);
    CHECK(heap != nullptr);
    CHECK_EQ(turbo_buffer_arena_owns(a1, heap), 0);
    std::free(heap);

    // A slab rented from a sibling arena is not owned by this one.
    CHECK_EQ(turbo_buffer_arena_owns(a2, base), 0);

    void *saved = v.ptr;
    CHECK_ST(turbo_buffer_arena_return(a1, &v));
    // Documented contract: owns is true for slabs "in-use or free", so the
    // backing stays owned after return while the slab table keeps it.
    CHECK_EQ(turbo_buffer_arena_owns(a1, saved), 1);
    // The returned view itself was zeroed, so its null ptr is not owned.
    CHECK_EQ(turbo_buffer_arena_owns(a1, v.ptr), 0);

    turbo_buffer_arena_destroy(a2);
    turbo_buffer_arena_destroy(a1);
}

static void test_row_stride_zero_pads_rows_to_64b() {
    turbo_buffer_arena *a = nullptr;
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &a));

    // i32, cols=10: 10 elems = 40B, padded to the 16-elem (64B) stride.
    turbo_buffer_view v {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 4, 10, 0, &v
    ));
    CHECK_EQ(v.row_stride, 16u);
    for (uint32_t r = 0; r < v.rows; ++r) {
        int32_t *row = turbo_buffer_i32_row(&v, r);
        CHECK(aligned64(row));
        if (r + 1 < v.rows) {
            const ptrdiff_t gap = static_cast<const char *>(
                static_cast<const void *>(turbo_buffer_i32_row(&v, r + 1))
            ) - static_cast<const char *>(static_cast<const void *>(row));
            CHECK_EQ(gap, 64); // 16 elems * 4B
        }
        row[0] = static_cast<int32_t>(1000 + r);
        row[9] = static_cast<int32_t>(2000 + r);
    }
    for (uint32_t r = 0; r < v.rows; ++r) {
        CHECK_EQ(turbo_buffer_i32_row(&v, r)[0], static_cast<int32_t>(1000 + r));
        CHECK_EQ(turbo_buffer_i32_row(&v, r)[9], static_cast<int32_t>(2000 + r));
    }
    CHECK_ST(turbo_buffer_arena_return(a, &v));

    // f32, cols=17: 68B per row, padded up to 32 elems (128B) per row.
    turbo_buffer_view f {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_PLACE_HOST, 3, 17, 0, &f
    ));
    CHECK_EQ(f.row_stride, 32u);
    for (uint32_t r = 0; r < f.rows; ++r) {
        float *row = turbo_buffer_f32_row(&f, r);
        CHECK(aligned64(row));
        if (r + 1 < f.rows) {
            const ptrdiff_t gap = static_cast<const char *>(
                static_cast<const void *>(turbo_buffer_f32_row(&f, r + 1))
            ) - static_cast<const char *>(static_cast<const void *>(row));
            CHECK_EQ(gap, 128); // 32 elems * 4B
        }
        row[16] = static_cast<float>(r) + 0.5f;
    }
    for (uint32_t r = 0; r < f.rows; ++r) {
        CHECK_EQ(turbo_buffer_f32_row(&f, r)[16], static_cast<float>(r) + 0.5f);
    }
    CHECK_ST(turbo_buffer_arena_return(a, &f));

    // cols already a multiple of 16 elems: no extra padding.
    turbo_buffer_view exact {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 1, 16, 0, &exact
    ));
    CHECK_EQ(exact.row_stride, 16u);
    CHECK(aligned64(turbo_buffer_i32_row(&exact, 0)));
    CHECK_ST(turbo_buffer_arena_return(a, &exact));

    turbo_buffer_arena_destroy(a);
}

static void test_explicit_row_stride_honored() {
    turbo_buffer_arena *a = nullptr;
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &a));

    // Stride larger than cols: rows spaced by stride elements, not cols.
    turbo_buffer_view v {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 3, 6, 8, &v
    ));
    CHECK_EQ(v.row_stride, 8u);
    for (uint32_t r = 0; r < v.rows; ++r) {
        int32_t *row = turbo_buffer_i32_row(&v, r);
        CHECK(row != nullptr);
        if (r + 1 < v.rows) {
            const ptrdiff_t gap = turbo_buffer_i32_row(&v, r + 1) - row;
            CHECK_EQ(gap, 8); // elements
        }
        row[5] = static_cast<int32_t>(3000 + r);
    }
    for (uint32_t r = 0; r < v.rows; ++r) {
        CHECK_EQ(turbo_buffer_i32_row(&v, r)[5], static_cast<int32_t>(3000 + r));
    }
    // Accessor guards: row out of range, dtype mismatch.
    CHECK(turbo_buffer_i32_row(&v, 3) == nullptr);
    CHECK(turbo_buffer_f32_row(&v, 0) == nullptr);
    CHECK(turbo_buffer_view_f32(&v) == nullptr);
    CHECK(turbo_buffer_view_i32(&v) != nullptr);
    CHECK_ST(turbo_buffer_arena_return(a, &v));

    // Packed stride == cols.
    turbo_buffer_view packed {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_PLACE_HOST, 2, 12, 12, &packed
    ));
    CHECK_EQ(packed.row_stride, 12u);
    const ptrdiff_t gap = turbo_buffer_f32_row(&packed, 1) -
                          turbo_buffer_f32_row(&packed, 0);
    CHECK_EQ(gap, 12);
    turbo_buffer_f32_row(&packed, 1)[11] = -2.5f;
    CHECK_EQ(turbo_buffer_f32_row(&packed, 1)[11], -2.5f);
    CHECK_ST(turbo_buffer_arena_return(a, &packed));

    turbo_buffer_arena_destroy(a);
}

static void test_invalid_enums_rejected() {
    turbo_buffer_arena *a = nullptr;
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &a));

    turbo_buffer_view v {};
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a,
            static_cast<turbo_buffer_dtype>(0),
            TURBO_BUFFER_PLACE_HOST,
            1,
            8,
            8,
            &v
        ),
        TURBO_BUFFER_ERR_INVALID_ARGUMENT
    );
    CHECK(v.ptr == nullptr);
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a,
            static_cast<turbo_buffer_dtype>(99),
            TURBO_BUFFER_PLACE_HOST,
            1,
            8,
            8,
            &v
        ),
        TURBO_BUFFER_ERR_INVALID_ARGUMENT
    );
    CHECK(v.ptr == nullptr);

    // Unknown placements fail loud (NOT_IMPLEMENTED), never remap to HOST.
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a,
            TURBO_BUFFER_DTYPE_I32,
            static_cast<turbo_buffer_placement>(0),
            1,
            8,
            8,
            &v
        ),
        TURBO_BUFFER_ERR_NOT_IMPLEMENTED
    );
    CHECK(v.ptr == nullptr);
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a,
            TURBO_BUFFER_DTYPE_I32,
            static_cast<turbo_buffer_placement>(99),
            1,
            8,
            8,
            &v
        ),
        TURBO_BUFFER_ERR_NOT_IMPLEMENTED
    );
    CHECK(v.ptr == nullptr);
    CHECK(std::strstr(turbo_buffer_last_error(a), "remap") != nullptr);

    // Unknown device on raw_alloc: no pointer and no CPU remap. The probe
    // runs placement_ok first, which rejects every placement for an unknown
    // device, so the status is NOT_IMPLEMENTED (arena_create reports
    // UNSUPPORTED_DEVICE for the same device — the codes differ by entry
    // point; both fail loud).
    void *p = nullptr;
    CHECK_EQ(
        turbo_buffer_raw_alloc(
            static_cast<turbo_buffer_device>(3),
            TURBO_BUFFER_PLACE_HOST,
            16,
            &p
        ),
        TURBO_BUFFER_ERR_NOT_IMPLEMENTED
    );
    CHECK(p == nullptr);

    turbo_buffer_arena_destroy(a);
}

static void test_zero_rows_cols_and_bad_stride_rejected() {
    turbo_buffer_arena *a = nullptr;
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &a));

    turbo_buffer_view v {};
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 0, 8, 8, &v
        ),
        TURBO_BUFFER_ERR_INVALID_ARGUMENT
    );
    CHECK(v.ptr == nullptr);
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 8, 0, 0, &v
        ),
        TURBO_BUFFER_ERR_INVALID_ARGUMENT
    );
    CHECK(v.ptr == nullptr);

    // Explicit stride below cols is invalid; stride==0 (padding) is fine.
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 1, 8, 4, &v
        ),
        TURBO_BUFFER_ERR_INVALID_ARGUMENT
    );
    CHECK(v.ptr == nullptr);
    CHECK(std::strstr(turbo_buffer_last_error(a), "row_stride") != nullptr);

    // Null out-view / null view on return are invalid arguments.
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 1, 8, 8, nullptr
        ),
        TURBO_BUFFER_ERR_INVALID_ARGUMENT
    );
    CHECK_EQ(turbo_buffer_arena_return(a, nullptr), TURBO_BUFFER_ERR_INVALID_ARGUMENT);

    // A zero-initialized (never rented) view is an empty view, not a
    // double-free, and does not match any slab.
    turbo_buffer_view foreign {};
    CHECK_EQ(turbo_buffer_arena_return(a, &foreign), TURBO_BUFFER_ERR_INVALID_ARGUMENT);

    turbo_buffer_arena_destroy(a);
}

static void test_raw_alloc_roundtrip() {
    turbo_buffer_alloc_counter_reset();

    // 1 byte: still 64-byte aligned, write/read round-trip.
    void *p1 = nullptr;
    CHECK_ST(turbo_buffer_raw_alloc(
        TURBO_BUFFER_DEVICE_CPU, TURBO_BUFFER_PLACE_HOST, 1, &p1
    ));
    CHECK(p1 != nullptr);
    CHECK(aligned64(p1));
    static_cast<unsigned char *>(p1)[0] = 0xABu;
    CHECK_EQ(static_cast<unsigned char *>(p1)[0], 0xABu);

    // 1 MiB: fill and verify every byte.
    const size_t kMiB = 1024u * 1024u;
    void *pm = nullptr;
    CHECK_ST(turbo_buffer_raw_alloc(
        TURBO_BUFFER_DEVICE_CPU, TURBO_BUFFER_PLACE_HOST, kMiB, &pm
    ));
    CHECK(pm != nullptr);
    CHECK(aligned64(pm));
    std::memset(pm, 0x5A, kMiB);
    static_cast<unsigned char *>(pm)[12345] = 0xC3u;
    const unsigned char *bytes = static_cast<const unsigned char *>(pm);
    size_t mismatches = 0;
    for (size_t i = 0; i < kMiB; ++i) {
        const unsigned char expected = i == 12345 ? 0xC3u : 0x5Au;
        if (bytes[i] != expected) {
            ++mismatches;
        }
    }
    CHECK_EQ(mismatches, 0u);

    // Both allocations are counted on the shared alloc counter.
    CHECK_EQ(turbo_buffer_alloc_counter(), 2u);

    // Degenerate sizes / null out are rejected without allocating.
    void *pz = reinterpret_cast<void *>(0x1);
    CHECK_EQ(
        turbo_buffer_raw_alloc(
            TURBO_BUFFER_DEVICE_CPU, TURBO_BUFFER_PLACE_HOST, 0, &pz
        ),
        TURBO_BUFFER_ERR_INVALID_ARGUMENT
    );
    CHECK(pz == nullptr);
    CHECK_EQ(
        turbo_buffer_raw_alloc(
            TURBO_BUFFER_DEVICE_CPU, TURBO_BUFFER_PLACE_HOST, 16, nullptr
        ),
        TURBO_BUFFER_ERR_INVALID_ARGUMENT
    );
    // CPU raw_alloc refuses GPU placements instead of remapping.
    void *pg = nullptr;
    CHECK_EQ(
        turbo_buffer_raw_alloc(
            TURBO_BUFFER_DEVICE_CPU, TURBO_BUFFER_PLACE_DEVICE, 16, &pg
        ),
        TURBO_BUFFER_ERR_NOT_IMPLEMENTED
    );
    CHECK(pg == nullptr);
    CHECK_EQ(turbo_buffer_alloc_counter(), 2u);

    turbo_buffer_raw_free(TURBO_BUFFER_DEVICE_CPU, TURBO_BUFFER_PLACE_HOST, p1);
    turbo_buffer_raw_free(TURBO_BUFFER_DEVICE_CPU, TURBO_BUFFER_PLACE_HOST, pm);
    // NULL free is a documented no-op.
    turbo_buffer_raw_free(TURBO_BUFFER_DEVICE_CPU, TURBO_BUFFER_PLACE_HOST, nullptr);
}

int main() {
    test_abi_version_and_names();
    test_last_error_never_null();
    test_rent_return_100_cycles_zero_alloc();
    test_double_return_contract();
    test_arena_owns_boundaries();
    test_row_stride_zero_pads_rows_to_64b();
    test_explicit_row_stride_honored();
    test_invalid_enums_rejected();
    test_zero_rows_cols_and_bad_stride_rejected();
    test_raw_alloc_roundtrip();

    std::fprintf(
        stderr, "turbo_buffer_edge_tests: %d passed, %d failed\n", g_passes, g_fails
    );
    return g_fails == 0 ? 0 : 1;
}
