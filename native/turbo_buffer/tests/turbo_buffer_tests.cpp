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

static void test_cuda_pinned_device_live_or_fail_loud() {
    const turbo_buffer_status pin_probe =
        turbo_buffer_backend_probe(TURBO_BUFFER_DEVICE_CUDA, TURBO_BUFFER_PLACE_PINNED);
    const turbo_buffer_status dev_probe =
        turbo_buffer_backend_probe(TURBO_BUFFER_DEVICE_CUDA, TURBO_BUFFER_PLACE_DEVICE);
    turbo_buffer_arena *cuda = nullptr;
    const turbo_buffer_status cuda_st =
        turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_CUDA, &cuda);
    if (pin_probe == TURBO_BUFFER_OK) {
        CHECK_EQ(dev_probe, TURBO_BUFFER_OK);
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
        CHECK_EQ(pin.placement, TURBO_BUFFER_PLACE_PINNED);
        CHECK_EQ(pin.device, TURBO_BUFFER_DEVICE_CUDA);
        turbo_buffer_i32_row(&pin, 0)[0] = 101;
        CHECK_EQ(turbo_buffer_i32_row(&pin, 0)[0], 101);
        CHECK(turbo_buffer_arena_owns(cuda, pin.ptr));

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
        CHECK_EQ(dev.placement, TURBO_BUFFER_PLACE_DEVICE);
        CHECK(turbo_buffer_arena_owns(cuda, dev.ptr));
        void *pin_ptr = pin.ptr;
        void *dev_ptr = dev.ptr;
        CHECK_ST(turbo_buffer_arena_return(cuda, &pin));
        CHECK_ST(turbo_buffer_arena_return(cuda, &dev));

        turbo_buffer_alloc_counter_reset();
        turbo_buffer_view pin2 {};
        turbo_buffer_view dev2 {};
        CHECK_ST(turbo_buffer_arena_rent(
            cuda,
            TURBO_BUFFER_DTYPE_I32,
            TURBO_BUFFER_PLACE_PINNED,
            2,
            16,
            16,
            &pin2
        ));
        CHECK_ST(turbo_buffer_arena_rent(
            cuda,
            TURBO_BUFFER_DTYPE_F32,
            TURBO_BUFFER_PLACE_DEVICE,
            1,
            64,
            64,
            &dev2
        ));
        CHECK_EQ(turbo_buffer_alloc_counter(), 0u);
        CHECK(pin2.ptr == pin_ptr);
        CHECK(dev2.ptr == dev_ptr);
        CHECK_ST(turbo_buffer_arena_return(cuda, &pin2));
        CHECK_ST(turbo_buffer_arena_return(cuda, &dev2));

        turbo_buffer_view host {};
        CHECK_EQ(
            turbo_buffer_arena_rent(
                cuda, TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_PLACE_HOST, 1, 4, 4, &host
            ),
            TURBO_BUFFER_ERR_NOT_IMPLEMENTED
        );
        turbo_buffer_cuda_forward_enter();
        turbo_buffer_cuda_forward_allocs_reset();
        void *trip = nullptr;
        CHECK_ST(turbo_buffer_raw_alloc(
            TURBO_BUFFER_DEVICE_CUDA, TURBO_BUFFER_PLACE_DEVICE, 256, &trip
        ));
        CHECK(trip != nullptr);
        CHECK(turbo_buffer_cuda_forward_allocs() >= 1u);
        turbo_buffer_raw_free(
            TURBO_BUFFER_DEVICE_CUDA, TURBO_BUFFER_PLACE_DEVICE, trip
        );
        turbo_buffer_cuda_forward_leave();
        turbo_buffer_cuda_forward_allocs_reset();
        CHECK_EQ(turbo_buffer_cuda_forward_allocs(), 0u);

        turbo_buffer_arena_destroy(cuda);
        std::fprintf(stderr, "CUDA PINNED+DEVICE rent/return LIVE (arena reuse, 0 allocs)\n");
    } else {
        CHECK(cuda_st == TURBO_BUFFER_ERR_NOT_IMPLEMENTED ||
              cuda_st == TURBO_BUFFER_ERR_UNAVAILABLE);
        CHECK(cuda == nullptr);
        const char *msg = turbo_buffer_last_error(nullptr);
        CHECK(msg != nullptr);
        CHECK(std::strstr(msg, "Refusing CPU") != nullptr ||
              std::strstr(msg, "refusing CPU") != nullptr);
#ifndef TURBO_BUFFER_CUDA
        CHECK_EQ(pin_probe, TURBO_BUFFER_ERR_NOT_IMPLEMENTED);
        CHECK_EQ(dev_probe, TURBO_BUFFER_ERR_NOT_IMPLEMENTED);
#endif
    }
}

static void test_gpu_backends_fail_loud_or_work() {
    const turbo_buffer_status ze_probe =
        turbo_buffer_backend_probe(TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_HOST);
    turbo_buffer_arena *ze = nullptr;
    const turbo_buffer_status ze_st =
        turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_ZE, &ze);
    if (ze_probe == TURBO_BUFFER_OK) {
        CHECK_ST(ze_st);
        CHECK(ze != nullptr);
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

static void test_ze_usm_host_shared_device() {
    const turbo_buffer_status host_probe =
        turbo_buffer_backend_probe(TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_HOST);
    if (host_probe != TURBO_BUFFER_OK) {
#ifdef TURBO_BUFFER_ZE
        std::fprintf(stderr, "SKIP ZE USM LIVE (L0 compiled, host probe failed)\n");
#else
        CHECK_EQ(host_probe, TURBO_BUFFER_ERR_NOT_IMPLEMENTED);
        CHECK_EQ(
            turbo_buffer_ze_query(nullptr, nullptr),
            TURBO_BUFFER_ERR_NOT_IMPLEMENTED
        );
#endif
        return;
    }

    turbo_buffer_arena *a = nullptr;
    CHECK_ST(turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_ZE, &a));
    CHECK(a != nullptr);

    turbo_buffer_view host {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 2, 16, 16, &host
    ));
    CHECK(aligned64(host.ptr));
    turbo_buffer_placement q = TURBO_BUFFER_PLACE_PINNED;
    CHECK_ST(turbo_buffer_ze_query(host.ptr, &q));
    CHECK_EQ(q, TURBO_BUFFER_PLACE_HOST);
    turbo_buffer_i32_row(&host, 0)[0] = 101;
    turbo_buffer_i32_row(&host, 0)[1] = 7592;
    CHECK_EQ(turbo_buffer_i32_row(&host, 0)[0], 101);
    CHECK(turbo_buffer_arena_owns(a, host.ptr));

    turbo_buffer_view pinned_reject {};
    CHECK_EQ(
        turbo_buffer_arena_rent(
            a,
            TURBO_BUFFER_DTYPE_F32,
            TURBO_BUFFER_PLACE_PINNED,
            1,
            4,
            4,
            &pinned_reject
        ),
        TURBO_BUFFER_ERR_NOT_IMPLEMENTED
    );
    CHECK(pinned_reject.ptr == nullptr);

    const turbo_buffer_status shared_probe =
        turbo_buffer_backend_probe(TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_SHARED);
    const turbo_buffer_status device_probe =
        turbo_buffer_backend_probe(TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_DEVICE);
    CHECK_EQ(shared_probe, device_probe);
    if (shared_probe != TURBO_BUFFER_OK) {
        const char *msg = turbo_buffer_last_error(nullptr);
        CHECK(msg != nullptr);
        CHECK(std::strstr(msg, "Refusing") != nullptr ||
              std::strstr(msg, "no GPU") != nullptr);
        CHECK_ST(turbo_buffer_arena_return(a, &host));
        turbo_buffer_arena_destroy(a);
        std::fprintf(stderr, "ZE HOST LIVE; SHARED/DEVICE UNAVAILABLE (no GPU)\n");
        return;
    }

    turbo_buffer_view shared {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_SHARED, 2, 16, 16, &shared
    ));
    CHECK(aligned64(shared.ptr));
    CHECK_ST(turbo_buffer_ze_query(shared.ptr, &q));
    CHECK_EQ(q, TURBO_BUFFER_PLACE_SHARED);
    CHECK(q != TURBO_BUFFER_PLACE_HOST);
    turbo_buffer_i32_row(&shared, 0)[0] = 202;
    CHECK_EQ(turbo_buffer_i32_row(&shared, 0)[0], 202);

    turbo_buffer_view device {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_DEVICE, 2, 16, 16, &device
    ));
    CHECK(device.ptr != nullptr);
    CHECK(aligned64(device.ptr));
    CHECK_ST(turbo_buffer_ze_query(device.ptr, &q));
    CHECK_EQ(q, TURBO_BUFFER_PLACE_DEVICE);
    CHECK(q != TURBO_BUFFER_PLACE_HOST);
    CHECK(q != TURBO_BUFFER_PLACE_SHARED);

    const int32_t pattern[4] = {101, 7592, 102, 0};
    std::memcpy(turbo_buffer_view_i32(&host), pattern, sizeof(pattern));
    CHECK_ST(turbo_buffer_ze_memcpy(device.ptr, host.ptr, sizeof(pattern)));
    std::memset(turbo_buffer_view_i32(&host), 0, sizeof(pattern));
    CHECK_ST(turbo_buffer_ze_memcpy(host.ptr, device.ptr, sizeof(pattern)));
    CHECK_EQ(turbo_buffer_view_i32(&host)[0], 101);
    CHECK_EQ(turbo_buffer_view_i32(&host)[1], 7592);
    CHECK_EQ(turbo_buffer_view_i32(&host)[2], 102);

    CHECK_ST(turbo_buffer_arena_return(a, &host));
    CHECK_ST(turbo_buffer_arena_return(a, &shared));
    CHECK_ST(turbo_buffer_arena_return(a, &device));

    turbo_buffer_alloc_counter_reset();
    turbo_buffer_view h2 {};
    turbo_buffer_view s2 {};
    turbo_buffer_view d2 {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 2, 16, 16, &h2
    ));
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_SHARED, 2, 16, 16, &s2
    ));
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_DEVICE, 2, 16, 16, &d2
    ));
    CHECK_EQ(turbo_buffer_alloc_counter(), 0u);
    CHECK_ST(turbo_buffer_ze_query(h2.ptr, &q));
    CHECK_EQ(q, TURBO_BUFFER_PLACE_HOST);
    CHECK_ST(turbo_buffer_ze_query(s2.ptr, &q));
    CHECK_EQ(q, TURBO_BUFFER_PLACE_SHARED);
    CHECK_ST(turbo_buffer_ze_query(d2.ptr, &q));
    CHECK_EQ(q, TURBO_BUFFER_PLACE_DEVICE);
    CHECK_ST(turbo_buffer_arena_return(a, &h2));
    CHECK_ST(turbo_buffer_arena_return(a, &s2));
    CHECK_ST(turbo_buffer_arena_return(a, &d2));

    turbo_buffer_view once {};
    CHECK_ST(turbo_buffer_arena_rent(
        a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_SHARED, 1, 8, 8, &once
    ));
    turbo_buffer_view stolen = once;
    CHECK_ST(turbo_buffer_arena_return(a, &once));
    CHECK_EQ(turbo_buffer_arena_return(a, &stolen), TURBO_BUFFER_ERR_DOUBLE_FREE);

    turbo_buffer_arena_destroy(a);
    std::fprintf(stderr, "ZE USM HOST/SHARED/DEVICE rent/return LIVE\n");
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
    test_cuda_pinned_device_live_or_fail_loud();
    test_gpu_backends_fail_loud_or_work();
    test_ze_usm_host_shared_device();
    test_invalid_rent();

    std::fprintf(
        stderr, "turbo_buffer_tests: %d passed, %d failed\n", g_passes, g_fails
    );
    return g_fails == 0 ? 0 : 1;
}
