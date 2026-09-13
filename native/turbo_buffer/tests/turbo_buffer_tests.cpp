// SPDX-License-Identifier: Apache-2.0
//
// Arena ABI tests. Fail if someone reintroduces per-forward malloc for
// buffers this ABI owns, remaps missing GPU backends to CPU, or
// double-frees a rent.

#include "turbo_buffer.h"

#include <cstdint>
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

#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_ST(st) CHECK((st) == TURBO_BUFFER_OK)

static bool aligned64(const void *p) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) % 64u) == 0;
}

static void test_abi_names() {
    CHECK_EQ(turbo_buffer_abi_version(), 1u);
    CHECK(std::strcmp(turbo_buffer_status_name(TURBO_BUFFER_ERR_DOUBLE_FREE),
                      "DOUBLE_FREE") == 0);
    CHECK(std::strcmp(turbo_buffer_device_name(TURBO_BUFFER_DEVICE_CPU), "CPU") ==
          0);
    CHECK(std::strcmp(turbo_buffer_placement_name(TURBO_BUFFER_PLACE_PINNED),
                      "PINNED") == 0);
}

static void test_cpu_alignment_and_stride() {
    turbo_buffer_arena *a = nullptr;
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &a));
    CHECK(a != nullptr);

    turbo_buffer_view v {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 3, 10, 0, &v
    ));
    CHECK(aligned64(v.ptr));
    CHECK_EQ(v.rows, 3u);
    CHECK_EQ(v.cols, 10u);
    CHECK(v.row_stride >= 10u);
    CHECK_EQ(v.row_stride % 16u, 0u); // 64 / sizeof(int32)
    int32_t *row1 = turbo_buffer_i32_row(&v, 1);
    CHECK(row1 != nullptr);
    CHECK(aligned64(row1));
    row1[0] = 101;
    row1[1] = 7592;
    CHECK_EQ(turbo_buffer_i32_row(&v, 1)[0], 101);
    CHECK_EQ(turbo_buffer_i32_row(&v, 1)[1], 7592);
    CHECK(turbo_buffer_arena_owns(a, v.ptr));
    CHECK(turbo_buffer_arena_owns(a, row1));
    CHECK_ST(turbo_buffer_arena_return(a, &v));
    CHECK(v.ptr == nullptr);

    turbo_buffer_view packed {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 2, 16, 16, &packed
    ));
    CHECK_EQ(packed.row_stride, 16u);
    CHECK(aligned64(packed.ptr));
    CHECK_ST(turbo_buffer_arena_return(a, &packed));

    turbo_buffer_view act {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_PLACE_HOST, 4, 384, 0, &act
    ));
    CHECK(aligned64(act.ptr));
    CHECK(turbo_buffer_view_f32(&act) != nullptr);
    CHECK(turbo_buffer_view_i32(&act) == nullptr);
    float *r0 = turbo_buffer_f32_row(&act, 0);
    r0[0] = 1.5f;
    CHECK_EQ(turbo_buffer_f32_row(&act, 0)[0], 1.5f);
    CHECK_ST(turbo_buffer_arena_return(a, &act));

    turbo_buffer_arena_destroy(a);
}

static void test_dual_rent_and_reuse_zero_alloc() {
    turbo_buffer_arena *a = nullptr;
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &a));

    turbo_buffer_view a1 {};
    turbo_buffer_view a2 {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_PLACE_HOST, 2, 8, 8, &a1
    ));
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_PLACE_HOST, 2, 8, 8, &a2
    ));
    CHECK(a1.ptr != a2.ptr);
    CHECK(a1.handle != a2.handle);
    turbo_buffer_view_f32(&a1)[0] = 3.0f;
    turbo_buffer_view_f32(&a2)[0] = 7.0f;
    CHECK_EQ(turbo_buffer_view_f32(&a1)[0], 3.0f);
    CHECK_EQ(turbo_buffer_view_f32(&a2)[0], 7.0f);

    CHECK_ST(turbo_buffer_arena_return(a, &a1));
    CHECK_ST(turbo_buffer_arena_return(a, &a2));

    turbo_buffer_alloc_counter_reset();
    turbo_buffer_view b1 {};
    turbo_buffer_view b2 {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_PLACE_HOST, 2, 8, 8, &b1
    ));
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_PLACE_HOST, 2, 8, 8, &b2
    ));
    CHECK_EQ(turbo_buffer_alloc_counter(), 0u);
    CHECK(b1.ptr != b2.ptr);
    CHECK_ST(turbo_buffer_arena_return(a, &b1));
    CHECK_ST(turbo_buffer_arena_return(a, &b2));

    turbo_buffer_arena_destroy(a);
}

static void test_no_double_free() {
    turbo_buffer_arena *a = nullptr;
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &a));
    turbo_buffer_view v {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 1, 8, 8, &v
    ));
    turbo_buffer_view copy = v;
    CHECK_ST(turbo_buffer_arena_return(a, &v));
    CHECK_EQ(v.handle, 0u);
    const turbo_buffer_status st = turbo_buffer_arena_return(a, &copy);
    CHECK_EQ(st, TURBO_BUFFER_ERR_DOUBLE_FREE);
    const char *msg = turbo_buffer_last_error(a);
    CHECK(msg != nullptr);
    CHECK(std::strstr(msg, "double-free") != nullptr);
    turbo_buffer_arena_destroy(a);
}

static void test_cpu_rejects_gpu_placement() {
    turbo_buffer_arena *a = nullptr;
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &a));
    turbo_buffer_view v {};
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_PLACE_DEVICE, 1, 4, 4, &v
        ),
        TURBO_BUFFER_ERR_NOT_IMPLEMENTED
    );
    CHECK(v.ptr == nullptr);
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_PLACE_PINNED, 1, 4, 4, &v
        ),
        TURBO_BUFFER_ERR_NOT_IMPLEMENTED
    );
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_PLACE_SHARED, 1, 4, 4, &v
        ),
        TURBO_BUFFER_ERR_NOT_IMPLEMENTED
    );
    turbo_buffer_arena_destroy(a);
}

static void test_gpu_backends_fail_loud_or_work() {
    const turbo_buffer_status cuda_probe =
        turbo_buffer_backend_probe(TURBO_BUFFER_DEVICE_CUDA, TURBO_BUFFER_PLACE_PINNED);
    turbo_buffer_arena *cuda = nullptr;
    const turbo_buffer_status cuda_st =
        turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CUDA, &cuda);
    if (cuda_probe == TURBO_BUFFER_OK) {
        CHECK_ST(cuda_st);
        CHECK(cuda != nullptr);
        turbo_buffer_view pin {};
        CHECK_ST(turbo_buffer_arena_rent(
            cuda,
            TURBO_BUFFER_DTYPE_I32,
            TURBO_BUFFER_PLACE_PINNED,
            2,
            16,
            16,
            &pin
        ));
        CHECK(aligned64(pin.ptr));
        turbo_buffer_i32_row(&pin, 0)[0] = 101;
        CHECK_EQ(turbo_buffer_i32_row(&pin, 0)[0], 101);
        turbo_buffer_view dev {};
        CHECK_ST(turbo_buffer_arena_rent(
            cuda,
            TURBO_BUFFER_DTYPE_F32,
            TURBO_BUFFER_PLACE_DEVICE,
            1,
            64,
            64,
            &dev
        ));
        CHECK(dev.ptr != nullptr);
        CHECK_ST(turbo_buffer_arena_return(cuda, &pin));
        CHECK_ST(turbo_buffer_arena_return(cuda, &dev));
        turbo_buffer_view host {};
        CHECK_EQ(
            turbo_buffer_arena_rent(
                cuda, TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_PLACE_HOST, 1, 4, 4, &host
            ),
            TURBO_BUFFER_ERR_NOT_IMPLEMENTED
        );
        turbo_buffer_arena_destroy(cuda);
    } else {
        CHECK(cuda_st == TURBO_BUFFER_ERR_NOT_IMPLEMENTED ||
              cuda_st == TURBO_BUFFER_ERR_UNAVAILABLE);
        CHECK(cuda == nullptr);
        const char *msg = turbo_buffer_last_error(nullptr);
        CHECK(msg != nullptr);
        CHECK(std::strstr(msg, "Refusing CPU") != nullptr ||
              std::strstr(msg, "refusing CPU") != nullptr);
#ifndef TURBO_BUFFER_CUDA
        CHECK_EQ(cuda_probe, TURBO_BUFFER_ERR_NOT_IMPLEMENTED);
#endif
    }

    const turbo_buffer_status ze_probe =
        turbo_buffer_backend_probe(TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_HOST);
    turbo_buffer_arena *ze = nullptr;
    const turbo_buffer_status ze_st =
        turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_ZE, &ze);
    if (ze_probe == TURBO_BUFFER_OK) {
        CHECK_ST(ze_st);
        CHECK(ze != nullptr);
        turbo_buffer_view host {};
        CHECK_ST(turbo_buffer_arena_rent(
            ze, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 1, 16, 16, &host
        ));
        CHECK(aligned64(host.ptr));
        CHECK_ST(turbo_buffer_arena_return(ze, &host));
        turbo_buffer_arena_destroy(ze);
    } else {
        CHECK(ze_st == TURBO_BUFFER_ERR_NOT_IMPLEMENTED ||
              ze_st == TURBO_BUFFER_ERR_UNAVAILABLE);
        CHECK(ze == nullptr);
        const char *msg = turbo_buffer_last_error(nullptr);
        CHECK(std::strstr(msg, "Refusing CPU") != nullptr ||
              std::strstr(msg, "refusing CPU") != nullptr ||
              std::strstr(msg, "Level Zero") != nullptr);
#ifndef TURBO_BUFFER_ZE
        CHECK_EQ(ze_probe, TURBO_BUFFER_ERR_NOT_IMPLEMENTED);
#endif
    }

    const turbo_buffer_status metal_probe = turbo_buffer_backend_probe(
        TURBO_BUFFER_DEVICE_METAL, TURBO_BUFFER_PLACE_SHARED
    );
    turbo_buffer_arena *metal = nullptr;
    const turbo_buffer_status metal_st =
        turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_METAL, &metal);
    if (metal_probe == TURBO_BUFFER_OK) {
        CHECK_ST(metal_st);
        CHECK(metal != nullptr);
        turbo_buffer_view sh {};
        CHECK_ST(turbo_buffer_arena_rent(
            metal,
            TURBO_BUFFER_DTYPE_I32,
            TURBO_BUFFER_PLACE_SHARED,
            2,
            16,
            16,
            &sh
        ));
        CHECK(aligned64(sh.ptr));
        CHECK(turbo_buffer_metal_owns(sh.ptr));
        CHECK_ST(turbo_buffer_arena_return(metal, &sh));
        turbo_buffer_arena_destroy(metal);
    } else {
        CHECK(metal_st == TURBO_BUFFER_ERR_NOT_IMPLEMENTED ||
              metal_st == TURBO_BUFFER_ERR_UNAVAILABLE);
        CHECK(metal == nullptr);
        const char *msg = turbo_buffer_last_error(nullptr);
        CHECK(std::strstr(msg, "Refusing CPU") != nullptr ||
              std::strstr(msg, "without Metal") != nullptr);
#ifndef TURBO_BUFFER_METAL
        CHECK_EQ(metal_probe, TURBO_BUFFER_ERR_NOT_IMPLEMENTED);
#endif
    }
}

static void test_invalid_rent() {
    turbo_buffer_arena *a = nullptr;
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CPU, &a));
    turbo_buffer_view v {};
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 0, 8, 8, &v
        ),
        TURBO_BUFFER_ERR_INVALID_ARGUMENT
    );
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 1, 8, 4, &v
        ),
        TURBO_BUFFER_ERR_INVALID_ARGUMENT
    );
    turbo_buffer_arena_destroy(a);
}

int main() {
    test_abi_names();
    test_cpu_alignment_and_stride();
    test_dual_rent_and_reuse_zero_alloc();
    test_no_double_free();
    test_cpu_rejects_gpu_placement();
    test_gpu_backends_fail_loud_or_work();
    test_invalid_rent();

    std::fprintf(
        stderr, "turbo_buffer_tests: %d passed, %d failed\n", g_passes, g_fails
    );
    return g_fails == 0 ? 0 : 1;
}
