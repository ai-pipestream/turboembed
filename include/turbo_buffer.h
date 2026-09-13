/* SPDX-License-Identifier: Apache-2.0
 *
 * TurboBuffer C ABI — frozen surface v1.
 *
 * Single device-buffer arena that TurboEmbed and TurboRerank rent from.
 * Do not fork this file per language.
 *
 * Canonical path: include/turbo_buffer.h
 * Apple copy (keep identical): swift/Sources/TurboBufferC/include/turbo_buffer.h
 *
 * ---------------------------------------------------------------------------
 * Ownership
 * ---------------------------------------------------------------------------
 *
 * The arena owns every backing allocation. Engines (and tests) rent a
 * typed view and return it. Do not free `view.ptr`. Destroying the
 * arena releases every slab, including any still checked out.
 *
 * Rent of a previously-sized slab does not allocate. After warmup
 * (load / first rent of that shape), steady-state forward MUST see
 * turbo_buffer_alloc_counter() == 0. There is no apology path that
 * mallocs a private copy for buffers this ABI owns.
 *
 * ---------------------------------------------------------------------------
 * Backends (fail loud — never silent CPU)
 * ---------------------------------------------------------------------------
 *
 *   CPU:     64-byte posix_memalign. Placement HOST only.
 *   CUDA:    PINNED = cudaHostAlloc; DEVICE = cudaMalloc.
 *            Create/rent without nvcc → NOT_IMPLEMENTED.
 *            Compiled but no CUDA device → UNAVAILABLE.
 *            LIVE on Machine A (PINNED token rent + DEVICE activation
 *            scratch). Steady-state CUDA forward must see
 *            turbo_buffer_alloc_counter() == 0 and
 *            turbo_buffer_cuda_forward_allocs() == 0.
 *   ZE:      Level Zero USM. HOST / SHARED / DEVICE as requested.
 *            SHARED/DEVICE without a GPU device → UNAVAILABLE, not HOST.
 *            Machine B. Fail loud when L0 is missing.
 *   METAL:   MTLResourceStorageModeShared (SHARED; HOST aliases SHARED
 *            on unified memory). Machine C. Fail loud without Metal.
 *
 * PINNED on a CPU arena, DEVICE on Metal, SHARED on CUDA, etc. are
 * NOT_IMPLEMENTED — they are not remapped to a working placement.
 *
 * ---------------------------------------------------------------------------
 * Views
 * ---------------------------------------------------------------------------
 *
 * I32 token rows and F32 activations. Layout is [rows, cols] with
 * `row_stride >= cols` (elements, not bytes). `row_stride == 0` on
 * rent pads each row to 64-byte alignment. Pass an explicit stride
 * (often `stride == cols`) when a packed [batch, seq] contract is
 * required (TurboRerank OpenVINO).
 */

#ifndef TURBO_BUFFER_H
#define TURBO_BUFFER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Bump only for breaking layout / semantic changes. */
#define TURBO_BUFFER_ABI_VERSION 1u

/** CPU / ggml-mappable alignment. CUDA pinned and MTL shared are page-aligned. */
#define TURBO_BUFFER_CPU_ALIGNMENT 64u

/* -------------------------------------------------------------------------- */
/* Status / device / placement / dtype                                        */
/* -------------------------------------------------------------------------- */

typedef enum turbo_buffer_status {
    TURBO_BUFFER_OK = 0,
    TURBO_BUFFER_ERR_INVALID_ARGUMENT = 1,
    TURBO_BUFFER_ERR_NOT_FOUND = 2,
    TURBO_BUFFER_ERR_NOT_IMPLEMENTED = 3,
    TURBO_BUFFER_ERR_UNAVAILABLE = 4,
    TURBO_BUFFER_ERR_INTERNAL = 5,
    TURBO_BUFFER_ERR_OUT_OF_MEMORY = 6,
    TURBO_BUFFER_ERR_UNSUPPORTED_DEVICE = 7,
    /** Return of a view that is not checked out (or already returned). */
    TURBO_BUFFER_ERR_DOUBLE_FREE = 8
} turbo_buffer_status;

/**
 * Concrete backends. Numbers match turboembed/turborerank CPU/CUDA/Metal
 * so a façade can switch without a translation table for those three.
 * ZE is 4 (OpenVINO USM); it is not OPENVINO_CPU vs GPU — placement
 * selects host / shared / device.
 */
typedef enum turbo_buffer_device {
    TURBO_BUFFER_DEVICE_CPU = 1,
    TURBO_BUFFER_DEVICE_CUDA = 2,
    TURBO_BUFFER_DEVICE_ZE = 4,
    TURBO_BUFFER_DEVICE_METAL = 7
} turbo_buffer_device;

typedef enum turbo_buffer_placement {
    /** CPU aligned / ZE host / Metal unified (alias of SHARED). */
    TURBO_BUFFER_PLACE_HOST = 1,
    /** CUDA device / ZE device. Not host-visible unless the backend says so. */
    TURBO_BUFFER_PLACE_DEVICE = 2,
    /** ZE shared / Metal shared. */
    TURBO_BUFFER_PLACE_SHARED = 3,
    /** CUDA cudaHostAlloc pinned. */
    TURBO_BUFFER_PLACE_PINNED = 4
} turbo_buffer_placement;

typedef enum turbo_buffer_dtype {
    TURBO_BUFFER_DTYPE_I32 = 1,
    TURBO_BUFFER_DTYPE_F32 = 2
} turbo_buffer_dtype;

/**
 * Non-owning typed window into an arena slab.
 *
 * `ptr` is 64-byte aligned on every backend this ABI implements.
 * `row_stride` is in elements. Row `r` starts at
 *   ptr + r * row_stride * elem_size.
 * `handle` is the rent id; pass the same view to return.
 */
typedef struct turbo_buffer_view {
    void *ptr;
    uint32_t rows;
    uint32_t cols;
    uint32_t row_stride;
    turbo_buffer_dtype dtype;
    turbo_buffer_device device;
    turbo_buffer_placement placement;
    uint32_t handle;
} turbo_buffer_view;

typedef struct turbo_buffer_arena turbo_buffer_arena;

/* -------------------------------------------------------------------------- */
/* Lifecycle                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Create an arena for one backend. `*out` is non-NULL on OK.
 *
 * CUDA / ZE / Metal without that backend compiled → NOT_IMPLEMENTED.
 * Compiled but the device / loader is missing → UNAVAILABLE.
 * Never creates a CPU arena as a stand-in.
 */
turbo_buffer_status turbo_buffer_arena_create(
    turbo_buffer_device device,
    turbo_buffer_arena **out
);

/** Destroy an arena and every slab. NULL is a no-op. */
void turbo_buffer_arena_destroy(turbo_buffer_arena *arena);

uint32_t turbo_buffer_abi_version(void);

const char *turbo_buffer_status_name(turbo_buffer_status status);

const char *turbo_buffer_device_name(turbo_buffer_device device);

const char *turbo_buffer_placement_name(turbo_buffer_placement placement);

/**
 * Last error for `arena`. If `arena` is NULL, the thread-local message
 * from the most recent failed create/rent. Never NULL.
 */
const char *turbo_buffer_last_error(const turbo_buffer_arena *arena);

/* -------------------------------------------------------------------------- */
/* Rent / return                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Rent `[rows, cols]` of `dtype` at `placement`.
 *
 * `row_stride == 0` → pad so each row is 64-byte aligned.
 * `row_stride != 0` must be `>= cols`.
 *
 * Reuses a free slab of the same placement with `bytes >= need`.
 * A new backing allocation increments turbo_buffer_alloc_counter.
 */
turbo_buffer_status turbo_buffer_arena_rent(
    turbo_buffer_arena *arena,
    turbo_buffer_dtype dtype,
    turbo_buffer_placement placement,
    uint32_t rows,
    uint32_t cols,
    uint32_t row_stride,
    turbo_buffer_view *out
);

/**
 * Return a rented view. Double-return → DOUBLE_FREE. Zeros `*view` on
 * OK so a second return cannot silently succeed.
 */
turbo_buffer_status turbo_buffer_arena_return(
    turbo_buffer_arena *arena,
    turbo_buffer_view *view
);

/** 1 if `ptr` is inside a slab owned by `arena` (in-use or free). */
int turbo_buffer_arena_owns(const turbo_buffer_arena *arena, const void *ptr);

int32_t *turbo_buffer_view_i32(const turbo_buffer_view *view);

float *turbo_buffer_view_f32(const turbo_buffer_view *view);

/** Row pointer. NULL if `row >= rows` or dtype mismatch. */
int32_t *turbo_buffer_i32_row(const turbo_buffer_view *view, uint32_t row);

float *turbo_buffer_f32_row(const turbo_buffer_view *view, uint32_t row);

/* -------------------------------------------------------------------------- */
/* Raw backend alloc (same counters; prefer rent)                             */
/* -------------------------------------------------------------------------- */

/**
 * One-shot backend allocation. Counted. Used by the arena and by
 * engines that need a non-pooled pointer (tests). CUDA/ZE/Metal fail
 * loud — never CPU.
 */
turbo_buffer_status turbo_buffer_raw_alloc(
    turbo_buffer_device device,
    turbo_buffer_placement placement,
    size_t bytes,
    void **out
);

void turbo_buffer_raw_free(
    turbo_buffer_device device,
    turbo_buffer_placement placement,
    void *ptr
);

/**
 * Probe a backend+placement.
 *   OK              — can allocate now
 *   NOT_IMPLEMENTED — not compiled into this binary
 *   UNAVAILABLE     — compiled, device/loader missing
 */
turbo_buffer_status turbo_buffer_backend_probe(
    turbo_buffer_device device,
    turbo_buffer_placement placement
);

/**
 * Metal: look up the MTLBuffer for a SHARED pointer this ABI allocated.
 * `out_native` receives `id<MTLBuffer>` as `void *`. Offset in bytes.
 * Returns 1 on hit. On non-Metal binaries always 0.
 */
int turbo_buffer_metal_owns(const void *ptr);

int turbo_buffer_metal_lookup(
    const void *ptr,
    void **out_native,
    size_t *out_offset
);

/* -------------------------------------------------------------------------- */
/* Alloc counter (tests). Process-wide.                                       */
/* -------------------------------------------------------------------------- */

void turbo_buffer_alloc_counter_reset(void);

uint64_t turbo_buffer_alloc_counter(void);

/** Count a sibling allocation (e.g. load-time weight copy) on the same counter. */
void turbo_buffer_note_alloc(void);

/**
 * CUDA forward window (tests). Enter/leave wrap TurboRerank CUDA
 * `forward`. Any successful `cudaMalloc` / `cudaHostAlloc` in our
 * translation units while the window is open increments
 * `turbo_buffer_cuda_forward_allocs`. Load-time weight `cudaMalloc`
 * is outside the window. No-ops when CUDA is not compiled.
 *
 * Tests reset the counter after warmup and require 0 after the next
 * forward. Reintroducing per-forward `cudaMalloc` for PINNED tokens
 * or DEVICE activations fails that check.
 */
void turbo_buffer_cuda_forward_enter(void);

void turbo_buffer_cuda_forward_leave(void);

void turbo_buffer_cuda_forward_allocs_reset(void);

uint64_t turbo_buffer_cuda_forward_allocs(void);

#ifdef __cplusplus
}
#endif

#endif /* TURBO_BUFFER_H */
