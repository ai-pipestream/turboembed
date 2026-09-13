// SPDX-License-Identifier: Apache-2.0
//
// Unified arena: rent/return over CPU / CUDA / ZE / Metal backends.
// Slab table is fixed-capacity so rent after warmup does not grow it.

#include "internal.hpp"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <new>
#include <string>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace turbo_buffer {
namespace impl {

std::atomic<uint64_t> g_allocs{0};
std::atomic<uint32_t> g_cuda_fwd_depth{0};
std::atomic<uint64_t> g_cuda_fwd_allocs{0};
std::atomic<uint64_t> g_cuda_fwd_h2d_bytes{0};
std::atomic<uint64_t> g_cuda_fwd_h2d_calls{0};
std::atomic<uint64_t> g_ze_h2d_bytes{0};
std::atomic<uint64_t> g_ze_d2h_bytes{0};
std::atomic<uint64_t> g_ze_h2d_calls{0};
std::atomic<uint64_t> g_ze_d2h_calls{0};

void note_cuda_forward_alloc() {
    if (g_cuda_fwd_depth.load(std::memory_order_relaxed) > 0) {
        g_cuda_fwd_allocs.fetch_add(1, std::memory_order_relaxed);
    }
}

bool cuda_forward_window_open() {
    return g_cuda_fwd_depth.load(std::memory_order_relaxed) > 0;
}

void note_cuda_runtime_alloc_if_forward() {
    if (!cuda_forward_window_open()) {
        return;
    }
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    g_cuda_fwd_allocs.fetch_add(1, std::memory_order_relaxed);
}

void note_cuda_forward_h2d(size_t bytes) {
    if (!cuda_forward_window_open() || bytes == 0) {
        return;
    }
    g_cuda_fwd_h2d_bytes.fetch_add(bytes, std::memory_order_relaxed);
    g_cuda_fwd_h2d_calls.fetch_add(1, std::memory_order_relaxed);
}

void note_ze_xfer(size_t bytes, bool h2d) {
    if (bytes == 0) {
        return;
    }
    if (h2d) {
        g_ze_h2d_bytes.fetch_add(bytes, std::memory_order_relaxed);
        g_ze_h2d_calls.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_ze_d2h_bytes.fetch_add(bytes, std::memory_order_relaxed);
        g_ze_d2h_calls.fetch_add(1, std::memory_order_relaxed);
    }
}

namespace {

thread_local std::string g_tls_error;

} // namespace

void set_tls_error(const std::string &msg) {
    g_tls_error = msg;
}

const char *tls_error() {
    return g_tls_error.c_str();
}

void *cpu_alloc(size_t bytes, turbo_buffer_status *status) {
    if (bytes == 0) {
        if (status) {
            *status = TURBO_BUFFER_ERR_INVALID_ARGUMENT;
        }
        return nullptr;
    }
    void *ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(bytes, TURBO_BUFFER_CPU_ALIGNMENT);
    if (ptr == nullptr) {
        if (status) {
            *status = TURBO_BUFFER_ERR_OUT_OF_MEMORY;
        }
        return nullptr;
    }
#else
    const int rc = posix_memalign(&ptr, TURBO_BUFFER_CPU_ALIGNMENT, bytes);
    if (rc != 0 || ptr == nullptr) {
        if (status) {
            *status = TURBO_BUFFER_ERR_OUT_OF_MEMORY;
        }
        return nullptr;
    }
#endif
    std::memset(ptr, 0, bytes);
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (status) {
        *status = TURBO_BUFFER_OK;
    }
    return ptr;
}

void cpu_free(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

uint32_t elem_size(turbo_buffer_dtype dtype) {
    switch (dtype) {
    case TURBO_BUFFER_DTYPE_I32:
    case TURBO_BUFFER_DTYPE_F32:
        return 4u;
    default:
        return 0;
    }
}

uint32_t aligned_row_stride(turbo_buffer_dtype dtype, uint32_t cols) {
    const uint32_t es = elem_size(dtype);
    if (es == 0) {
        return cols;
    }
    const uint32_t align_elems = TURBO_BUFFER_CPU_ALIGNMENT / es;
    if (align_elems == 0) {
        return cols;
    }
    return ((cols + align_elems - 1u) / align_elems) * align_elems;
}

bool placement_ok(turbo_buffer_device device, turbo_buffer_placement placement) {
    switch (device) {
    case TURBO_BUFFER_DEVICE_CPU:
        return placement == TURBO_BUFFER_PLACE_HOST;
    case TURBO_BUFFER_DEVICE_CUDA:
        return placement == TURBO_BUFFER_PLACE_PINNED ||
               placement == TURBO_BUFFER_PLACE_DEVICE;
    case TURBO_BUFFER_DEVICE_ZE:
        return placement == TURBO_BUFFER_PLACE_HOST ||
               placement == TURBO_BUFFER_PLACE_SHARED ||
               placement == TURBO_BUFFER_PLACE_DEVICE;
    case TURBO_BUFFER_DEVICE_METAL:
        return placement == TURBO_BUFFER_PLACE_SHARED ||
               placement == TURBO_BUFFER_PLACE_HOST;
    default:
        return false;
    }
}

} // namespace impl
} // namespace turbo_buffer

struct Slab {
    void *ptr = nullptr;
    size_t bytes = 0;
    turbo_buffer_placement placement = TURBO_BUFFER_PLACE_HOST;
    uint32_t handle = 0;
    bool in_use = false;
};

struct turbo_buffer_arena {
    turbo_buffer_device device = TURBO_BUFFER_DEVICE_CPU;
    std::string last_error;
    mutable std::mutex mu;
    Slab slabs[256];
    uint32_t n_slabs = 0;
    uint32_t next_handle = 1;
};

namespace {

void set_err(turbo_buffer_arena *arena, const std::string &msg) {
    if (arena != nullptr) {
        arena->last_error = msg;
    } else {
        turbo_buffer::impl::set_tls_error(msg);
    }
}

turbo_buffer_status dispatch_probe(
    turbo_buffer_device device,
    turbo_buffer_placement placement
) {
    if (!turbo_buffer::impl::placement_ok(device, placement)) {
        turbo_buffer::impl::set_tls_error(
            "placement is not implemented for this backend; refusing a "
            "silent remap"
        );
        return TURBO_BUFFER_ERR_NOT_IMPLEMENTED;
    }
    switch (device) {
    case TURBO_BUFFER_DEVICE_CPU:
        return TURBO_BUFFER_OK;
    case TURBO_BUFFER_DEVICE_CUDA:
        return turbo_buffer::impl::cuda_probe(placement);
    case TURBO_BUFFER_DEVICE_ZE:
        return turbo_buffer::impl::ze_probe(placement);
    case TURBO_BUFFER_DEVICE_METAL:
        return turbo_buffer::impl::metal_probe(placement);
    default:
        turbo_buffer::impl::set_tls_error("unknown turbo_buffer device");
        return TURBO_BUFFER_ERR_UNSUPPORTED_DEVICE;
    }
}

void *dispatch_alloc(
    turbo_buffer_device device,
    turbo_buffer_placement placement,
    size_t bytes,
    turbo_buffer_status *status
) {
    switch (device) {
    case TURBO_BUFFER_DEVICE_CPU:
        return turbo_buffer::impl::cpu_alloc(bytes, status);
    case TURBO_BUFFER_DEVICE_CUDA:
        return turbo_buffer::impl::cuda_alloc(placement, bytes, status);
    case TURBO_BUFFER_DEVICE_ZE:
        return turbo_buffer::impl::ze_alloc(placement, bytes, status);
    case TURBO_BUFFER_DEVICE_METAL:
        return turbo_buffer::impl::metal_alloc(bytes, status);
    default:
        if (status) {
            *status = TURBO_BUFFER_ERR_UNSUPPORTED_DEVICE;
        }
        return nullptr;
    }
}

void dispatch_free(
    turbo_buffer_device device,
    turbo_buffer_placement placement,
    void *ptr
) {
    switch (device) {
    case TURBO_BUFFER_DEVICE_CPU:
        turbo_buffer::impl::cpu_free(ptr);
        break;
    case TURBO_BUFFER_DEVICE_CUDA:
        turbo_buffer::impl::cuda_free(placement, ptr);
        break;
    case TURBO_BUFFER_DEVICE_ZE:
        turbo_buffer::impl::ze_free(ptr);
        break;
    case TURBO_BUFFER_DEVICE_METAL:
        turbo_buffer::impl::metal_free(ptr);
        break;
    default:
        break;
    }
}

} // namespace

extern "C" {

uint32_t turbo_buffer_abi_version(void) {
    return TURBO_BUFFER_ABI_VERSION;
}

const char *turbo_buffer_status_name(turbo_buffer_status status) {
    switch (status) {
    case TURBO_BUFFER_OK:
        return "OK";
    case TURBO_BUFFER_ERR_INVALID_ARGUMENT:
        return "INVALID_ARGUMENT";
    case TURBO_BUFFER_ERR_NOT_FOUND:
        return "NOT_FOUND";
    case TURBO_BUFFER_ERR_NOT_IMPLEMENTED:
        return "NOT_IMPLEMENTED";
    case TURBO_BUFFER_ERR_UNAVAILABLE:
        return "UNAVAILABLE";
    case TURBO_BUFFER_ERR_INTERNAL:
        return "INTERNAL";
    case TURBO_BUFFER_ERR_OUT_OF_MEMORY:
        return "OUT_OF_MEMORY";
    case TURBO_BUFFER_ERR_UNSUPPORTED_DEVICE:
        return "UNSUPPORTED_DEVICE";
    case TURBO_BUFFER_ERR_DOUBLE_FREE:
        return "DOUBLE_FREE";
    default:
        return "UNKNOWN";
    }
}

const char *turbo_buffer_device_name(turbo_buffer_device device) {
    switch (device) {
    case TURBO_BUFFER_DEVICE_CPU:
        return "CPU";
    case TURBO_BUFFER_DEVICE_CUDA:
        return "CUDA";
    case TURBO_BUFFER_DEVICE_ZE:
        return "ZE";
    case TURBO_BUFFER_DEVICE_METAL:
        return "METAL";
    default:
        return "UNKNOWN";
    }
}

const char *turbo_buffer_placement_name(turbo_buffer_placement placement) {
    switch (placement) {
    case TURBO_BUFFER_PLACE_HOST:
        return "HOST";
    case TURBO_BUFFER_PLACE_DEVICE:
        return "DEVICE";
    case TURBO_BUFFER_PLACE_SHARED:
        return "SHARED";
    case TURBO_BUFFER_PLACE_PINNED:
        return "PINNED";
    default:
        return "UNKNOWN";
    }
}

const char *turbo_buffer_last_error(const turbo_buffer_arena *arena) {
    if (arena != nullptr) {
        return arena->last_error.empty() ? "" : arena->last_error.c_str();
    }
    return turbo_buffer::impl::tls_error();
}

turbo_buffer_status turbo_buffer_backend_probe(
    turbo_buffer_device device,
    turbo_buffer_placement placement
) {
    return dispatch_probe(device, placement);
}

turbo_buffer_status turbo_buffer_raw_alloc(
    turbo_buffer_device device,
    turbo_buffer_placement placement,
    size_t bytes,
    void **out
) {
    if (out == nullptr) {
        turbo_buffer::impl::set_tls_error("raw_alloc: null out");
        return TURBO_BUFFER_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    const turbo_buffer_status probe = dispatch_probe(device, placement);
    if (probe != TURBO_BUFFER_OK) {
        return probe;
    }
    turbo_buffer_status st = TURBO_BUFFER_OK;
    void *ptr = dispatch_alloc(device, placement, bytes, &st);
    if (ptr == nullptr) {
        if (st == TURBO_BUFFER_OK) {
            st = TURBO_BUFFER_ERR_OUT_OF_MEMORY;
        }
        turbo_buffer::impl::set_tls_error("raw_alloc failed");
        return st;
    }
    *out = ptr;
    return TURBO_BUFFER_OK;
}

void turbo_buffer_raw_free(
    turbo_buffer_device device,
    turbo_buffer_placement placement,
    void *ptr
) {
    dispatch_free(device, placement, ptr);
}

turbo_buffer_status turbo_buffer_arena_create(
    turbo_buffer_device device,
    turbo_buffer_arena **out
) {
    if (out == nullptr) {
        turbo_buffer::impl::set_tls_error("arena_create: null out");
        return TURBO_BUFFER_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;

    turbo_buffer_placement probe_place = TURBO_BUFFER_PLACE_HOST;
    switch (device) {
    case TURBO_BUFFER_DEVICE_CPU:
        probe_place = TURBO_BUFFER_PLACE_HOST;
        break;
    case TURBO_BUFFER_DEVICE_CUDA:
        probe_place = TURBO_BUFFER_PLACE_PINNED;
        break;
    case TURBO_BUFFER_DEVICE_ZE:
        probe_place = TURBO_BUFFER_PLACE_HOST;
        break;
    case TURBO_BUFFER_DEVICE_METAL:
        probe_place = TURBO_BUFFER_PLACE_SHARED;
        break;
    default:
        turbo_buffer::impl::set_tls_error(
            "unknown turbo_buffer device; refusing CPU fallback"
        );
        return TURBO_BUFFER_ERR_UNSUPPORTED_DEVICE;
    }

    const turbo_buffer_status probe = dispatch_probe(device, probe_place);
    if (probe != TURBO_BUFFER_OK) {
        return probe;
    }

    auto *arena = new (std::nothrow) turbo_buffer_arena();
    if (arena == nullptr) {
        turbo_buffer::impl::set_tls_error("arena_create: OOM");
        return TURBO_BUFFER_ERR_OUT_OF_MEMORY;
    }
    arena->device = device;
    *out = arena;
    turbo_buffer::impl::set_tls_error("");
    return TURBO_BUFFER_OK;
}

void turbo_buffer_arena_destroy(turbo_buffer_arena *arena) {
    if (arena == nullptr) {
        return;
    }
    for (uint32_t i = 0; i < arena->n_slabs; ++i) {
        Slab &s = arena->slabs[i];
        dispatch_free(arena->device, s.placement, s.ptr);
        s.ptr = nullptr;
        s.in_use = false;
    }
    delete arena;
}

turbo_buffer_status turbo_buffer_arena_rent(
    turbo_buffer_arena *arena,
    turbo_buffer_dtype dtype,
    turbo_buffer_placement placement,
    uint32_t rows,
    uint32_t cols,
    uint32_t row_stride,
    turbo_buffer_view *out
) {
    if (arena == nullptr || out == nullptr) {
        turbo_buffer::impl::set_tls_error("rent: null argument");
        return TURBO_BUFFER_ERR_INVALID_ARGUMENT;
    }
    std::memset(out, 0, sizeof(*out));
    const uint32_t es = turbo_buffer::impl::elem_size(dtype);
    if (es == 0 || rows == 0 || cols == 0) {
        set_err(arena, "rent: dtype/rows/cols invalid");
        return TURBO_BUFFER_ERR_INVALID_ARGUMENT;
    }
    if (!turbo_buffer::impl::placement_ok(arena->device, placement)) {
        set_err(
            arena,
            "rent: placement is not implemented for this arena device; "
            "refusing a silent remap"
        );
        return TURBO_BUFFER_ERR_NOT_IMPLEMENTED;
    }
    const turbo_buffer_status probe = dispatch_probe(arena->device, placement);
    if (probe != TURBO_BUFFER_OK) {
        set_err(arena, turbo_buffer::impl::tls_error());
        return probe;
    }
    uint32_t stride = row_stride;
    if (stride == 0) {
        stride = turbo_buffer::impl::aligned_row_stride(dtype, cols);
    }
    if (stride < cols) {
        set_err(arena, "rent: row_stride < cols");
        return TURBO_BUFFER_ERR_INVALID_ARGUMENT;
    }
    const size_t bytes =
        static_cast<size_t>(rows) * static_cast<size_t>(stride) * static_cast<size_t>(es);

    std::lock_guard<std::mutex> lock(arena->mu);
    int best = -1;
    for (uint32_t i = 0; i < arena->n_slabs; ++i) {
        Slab &s = arena->slabs[i];
        if (s.in_use || s.placement != placement || s.bytes < bytes) {
            continue;
        }
        if (best < 0 || s.bytes < arena->slabs[best].bytes) {
            best = static_cast<int>(i);
        }
    }
    Slab *slot = nullptr;
    if (best >= 0) {
        slot = &arena->slabs[best];
    } else {
        if (arena->n_slabs >= 256) {
            set_err(arena, "rent: slab table full");
            return TURBO_BUFFER_ERR_OUT_OF_MEMORY;
        }
        turbo_buffer_status st = TURBO_BUFFER_OK;
        void *ptr = dispatch_alloc(arena->device, placement, bytes, &st);
        if (ptr == nullptr) {
            set_err(arena, "rent: backend alloc failed");
            return st == TURBO_BUFFER_OK ? TURBO_BUFFER_ERR_OUT_OF_MEMORY : st;
        }
        slot = &arena->slabs[arena->n_slabs++];
        slot->ptr = ptr;
        slot->bytes = bytes;
        slot->placement = placement;
        slot->handle = 0;
        slot->in_use = false;
    }
    slot->in_use = true;
    slot->handle = arena->next_handle++;
    if (arena->next_handle == 0) {
        arena->next_handle = 1;
    }
    out->ptr = slot->ptr;
    out->rows = rows;
    out->cols = cols;
    out->row_stride = stride;
    out->dtype = dtype;
    out->device = arena->device;
    out->placement = placement;
    out->handle = slot->handle;
    arena->last_error.clear();
    return TURBO_BUFFER_OK;
}

turbo_buffer_status turbo_buffer_arena_return(
    turbo_buffer_arena *arena,
    turbo_buffer_view *view
) {
    if (arena == nullptr || view == nullptr) {
        turbo_buffer::impl::set_tls_error("return: null argument");
        return TURBO_BUFFER_ERR_INVALID_ARGUMENT;
    }
    if (view->handle == 0 || view->ptr == nullptr) {
        set_err(arena, "return: empty view");
        return TURBO_BUFFER_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(arena->mu);
    Slab *slot = nullptr;
    for (uint32_t i = 0; i < arena->n_slabs; ++i) {
        if (arena->slabs[i].handle == view->handle &&
            arena->slabs[i].ptr == view->ptr) {
            slot = &arena->slabs[i];
            break;
        }
    }
    if (slot == nullptr) {
        set_err(arena, "return: view is not from this arena");
        return TURBO_BUFFER_ERR_NOT_FOUND;
    }
    if (!slot->in_use) {
        set_err(arena, "return: double-free of rented view");
        return TURBO_BUFFER_ERR_DOUBLE_FREE;
    }
    slot->in_use = false;
    std::memset(view, 0, sizeof(*view));
    arena->last_error.clear();
    return TURBO_BUFFER_OK;
}

int turbo_buffer_arena_owns(const turbo_buffer_arena *arena, const void *ptr) {
    if (arena == nullptr || ptr == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(arena->mu);
    const char *p = static_cast<const char *>(ptr);
    for (uint32_t i = 0; i < arena->n_slabs; ++i) {
        const Slab &s = arena->slabs[i];
        if (s.ptr == nullptr) {
            continue;
        }
        const char *start = static_cast<const char *>(s.ptr);
        if (p >= start && p < start + static_cast<ptrdiff_t>(s.bytes)) {
            return 1;
        }
    }
    return 0;
}

int32_t *turbo_buffer_view_i32(const turbo_buffer_view *view) {
    if (view == nullptr || view->dtype != TURBO_BUFFER_DTYPE_I32) {
        return nullptr;
    }
    return static_cast<int32_t *>(view->ptr);
}

float *turbo_buffer_view_f32(const turbo_buffer_view *view) {
    if (view == nullptr || view->dtype != TURBO_BUFFER_DTYPE_F32) {
        return nullptr;
    }
    return static_cast<float *>(view->ptr);
}

int32_t *turbo_buffer_i32_row(const turbo_buffer_view *view, uint32_t row) {
    int32_t *base = turbo_buffer_view_i32(view);
    if (base == nullptr || row >= view->rows) {
        return nullptr;
    }
    return base + static_cast<size_t>(row) * view->row_stride;
}

float *turbo_buffer_f32_row(const turbo_buffer_view *view, uint32_t row) {
    float *base = turbo_buffer_view_f32(view);
    if (base == nullptr || row >= view->rows) {
        return nullptr;
    }
    return base + static_cast<size_t>(row) * view->row_stride;
}

void turbo_buffer_alloc_counter_reset(void) {
    turbo_buffer::impl::g_allocs.store(0, std::memory_order_relaxed);
}

uint64_t turbo_buffer_alloc_counter(void) {
    return turbo_buffer::impl::g_allocs.load(std::memory_order_relaxed);
}

void turbo_buffer_note_alloc(void) {
    turbo_buffer::impl::g_allocs.fetch_add(1, std::memory_order_relaxed);
}

void turbo_buffer_cuda_forward_enter(void) {
    turbo_buffer::impl::g_cuda_fwd_depth.fetch_add(1, std::memory_order_relaxed);
}

void turbo_buffer_cuda_forward_leave(void) {
    uint32_t depth =
        turbo_buffer::impl::g_cuda_fwd_depth.load(std::memory_order_relaxed);
    if (depth > 0) {
        turbo_buffer::impl::g_cuda_fwd_depth.fetch_sub(1, std::memory_order_relaxed);
    }
}

void turbo_buffer_cuda_forward_allocs_reset(void) {
    turbo_buffer::impl::g_cuda_fwd_allocs.store(0, std::memory_order_relaxed);
}

uint64_t turbo_buffer_cuda_forward_allocs(void) {
    return turbo_buffer::impl::g_cuda_fwd_allocs.load(std::memory_order_relaxed);
}

void turbo_buffer_cuda_forward_h2d_reset(void) {
    turbo_buffer::impl::g_cuda_fwd_h2d_bytes.store(0, std::memory_order_relaxed);
    turbo_buffer::impl::g_cuda_fwd_h2d_calls.store(0, std::memory_order_relaxed);
}

uint64_t turbo_buffer_cuda_forward_h2d_bytes(void) {
    return turbo_buffer::impl::g_cuda_fwd_h2d_bytes.load(std::memory_order_relaxed);
}

uint64_t turbo_buffer_cuda_forward_h2d_calls(void) {
    return turbo_buffer::impl::g_cuda_fwd_h2d_calls.load(std::memory_order_relaxed);
}

void turbo_buffer_ze_xfer_reset(void) {
    turbo_buffer::impl::g_ze_h2d_bytes.store(0, std::memory_order_relaxed);
    turbo_buffer::impl::g_ze_d2h_bytes.store(0, std::memory_order_relaxed);
    turbo_buffer::impl::g_ze_h2d_calls.store(0, std::memory_order_relaxed);
    turbo_buffer::impl::g_ze_d2h_calls.store(0, std::memory_order_relaxed);
}

uint64_t turbo_buffer_ze_xfer_h2d_bytes(void) {
    return turbo_buffer::impl::g_ze_h2d_bytes.load(std::memory_order_relaxed);
}

uint64_t turbo_buffer_ze_xfer_d2h_bytes(void) {
    return turbo_buffer::impl::g_ze_d2h_bytes.load(std::memory_order_relaxed);
}

uint64_t turbo_buffer_ze_xfer_h2d_calls(void) {
    return turbo_buffer::impl::g_ze_h2d_calls.load(std::memory_order_relaxed);
}

uint64_t turbo_buffer_ze_xfer_d2h_calls(void) {
    return turbo_buffer::impl::g_ze_d2h_calls.load(std::memory_order_relaxed);
}

turbo_buffer_status turbo_buffer_ze_query(
    const void *ptr,
    turbo_buffer_placement *out
) {
    return turbo_buffer::impl::ze_query(ptr, out);
}

turbo_buffer_status turbo_buffer_ze_memcpy(
    void *dst,
    const void *src,
    size_t bytes
) {
    return turbo_buffer::impl::ze_memcpy(dst, src, bytes);
}

} // extern "C"
