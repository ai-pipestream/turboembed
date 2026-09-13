// SPDX-License-Identifier: Apache-2.0
//
// CUDA PINNED mapped (cudaHostAllocMapped) + DEVICE (cudaMalloc).
// PINNED pages are host-writable and device-visible so TurboRerank
// kernels read tokens without a per-forward H2D. Fail loud when this
// binary has no nvcc/cudart, zero devices, or canMapHostMemory == 0.
// Machine A is the live proof host. Do not include
// cuda_runtime_hooks.hpp here — this file implements the real calls
// the intercept wraps.

#include "internal.hpp"

#ifdef TURBO_BUFFER_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace turbo_buffer {
namespace impl {

#ifdef TURBO_BUFFER_CUDA

struct MappedPin {
    void *host = nullptr;
    void *device = nullptr;
    size_t bytes = 0;
};

std::mutex g_map_mu;
std::vector<MappedPin> g_mapped;

bool ensure_mapped_ready() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        return false;
    }
    // Flags must be set before the runtime creates a context.
    (void)cudaSetDeviceFlags(cudaDeviceMapHost);
    (void)cudaSetDevice(0);
    cudaDeviceProp prop {};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || !prop.canMapHostMemory) {
        return false;
    }
    return true;
}

void register_mapped(void *host, void *device, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_map_mu);
    g_mapped.push_back({host, device, bytes});
}

void unregister_mapped(void *host) {
    std::lock_guard<std::mutex> lock(g_map_mu);
    g_mapped.erase(
        std::remove_if(
            g_mapped.begin(),
            g_mapped.end(),
            [host](const MappedPin &m) { return m.host == host; }
        ),
        g_mapped.end()
    );
}

bool lookup_mapped(const void *host_ptr, void **device_ptr) {
    if (host_ptr == nullptr || device_ptr == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_map_mu);
    const char *p = static_cast<const char *>(host_ptr);
    for (const MappedPin &m : g_mapped) {
        if (m.host == nullptr || m.device == nullptr) {
            continue;
        }
        const char *start = static_cast<const char *>(m.host);
        if (p >= start && p < start + static_cast<ptrdiff_t>(m.bytes)) {
            *device_ptr = static_cast<char *>(m.device) + (p - start);
            return true;
        }
    }
    return false;
}

#endif

turbo_buffer_status cuda_probe(turbo_buffer_placement placement) {
    if (placement != TURBO_BUFFER_PLACE_PINNED &&
        placement != TURBO_BUFFER_PLACE_DEVICE) {
        set_tls_error(
            "CUDA arena implements PINNED (cudaHostAllocMapped) and DEVICE "
            "(cudaMalloc) only; refusing a silent CPU remap"
        );
        return TURBO_BUFFER_ERR_NOT_IMPLEMENTED;
    }
#ifdef TURBO_BUFFER_CUDA
    int count = 0;
    const cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        set_tls_error(
            std::string("TURBO_BUFFER_DEVICE_CUDA requested but "
                        "cudaGetDeviceCount failed (") +
            cudaGetErrorString(e) + "). Refusing CPU fallback."
        );
        return TURBO_BUFFER_ERR_UNAVAILABLE;
    }
    if (count <= 0) {
        set_tls_error(
            "TURBO_BUFFER_DEVICE_CUDA requested but CUDA runtime reports "
            "zero devices. Refusing CPU fallback."
        );
        return TURBO_BUFFER_ERR_UNAVAILABLE;
    }
    if (placement == TURBO_BUFFER_PLACE_PINNED && !ensure_mapped_ready()) {
        set_tls_error(
            "CUDA PINNED requires canMapHostMemory so tokens are "
            "device-visible without a per-forward H2D. Refusing a "
            "cudaMemcpy stand-in and refusing CPU fallback."
        );
        return TURBO_BUFFER_ERR_UNAVAILABLE;
    }
    return TURBO_BUFFER_OK;
#else
    set_tls_error(
        "TURBO_BUFFER_DEVICE_CUDA requested but this binary was built "
        "without CUDA (nvcc/cudart). Refusing CPU fallback."
    );
    return TURBO_BUFFER_ERR_NOT_IMPLEMENTED;
#endif
}

void *cuda_alloc(
    turbo_buffer_placement placement,
    size_t bytes,
    turbo_buffer_status *status
) {
#ifdef TURBO_BUFFER_CUDA
    if (bytes == 0) {
        if (status) {
            *status = TURBO_BUFFER_ERR_INVALID_ARGUMENT;
        }
        return nullptr;
    }
    if (placement == TURBO_BUFFER_PLACE_PINNED && !ensure_mapped_ready()) {
        if (status) {
            *status = TURBO_BUFFER_ERR_UNAVAILABLE;
        }
        return nullptr;
    }
    void *ptr = nullptr;
    cudaError_t e = cudaSuccess;
    if (placement == TURBO_BUFFER_PLACE_PINNED) {
        e = cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
    } else if (placement == TURBO_BUFFER_PLACE_DEVICE) {
        e = cudaMalloc(&ptr, bytes);
    } else {
        if (status) {
            *status = TURBO_BUFFER_ERR_NOT_IMPLEMENTED;
        }
        return nullptr;
    }
    if (e != cudaSuccess || ptr == nullptr) {
        if (status) {
            *status = TURBO_BUFFER_ERR_OUT_OF_MEMORY;
        }
        return nullptr;
    }
    if (placement == TURBO_BUFFER_PLACE_PINNED) {
        void *dptr = nullptr;
        if (cudaHostGetDevicePointer(&dptr, ptr, 0) != cudaSuccess ||
            dptr == nullptr) {
            (void)cudaFreeHost(ptr);
            if (status) {
                *status = TURBO_BUFFER_ERR_UNAVAILABLE;
            }
            return nullptr;
        }
        std::memset(ptr, 0, bytes);
        register_mapped(ptr, dptr, bytes);
    } else {
        (void)cudaMemset(ptr, 0, bytes);
    }
    turbo_buffer_note_alloc();
    note_cuda_forward_alloc();
    if (status) {
        *status = TURBO_BUFFER_OK;
    }
    return ptr;
#else
    (void)placement;
    (void)bytes;
    if (status) {
        *status = TURBO_BUFFER_ERR_NOT_IMPLEMENTED;
    }
    return nullptr;
#endif
}

void cuda_free(turbo_buffer_placement placement, void *ptr) {
    if (ptr == nullptr) {
        return;
    }
#ifdef TURBO_BUFFER_CUDA
    if (placement == TURBO_BUFFER_PLACE_PINNED) {
        unregister_mapped(ptr);
        (void)cudaFreeHost(ptr);
    } else {
        (void)cudaFree(ptr);
    }
#else
    (void)placement;
    (void)ptr;
#endif
}

} // namespace impl
} // namespace turbo_buffer

extern "C" {

int turbo_buffer_cuda_mapped_device_ptr(
    const void *host_ptr,
    void **device_ptr
) {
    if (device_ptr != nullptr) {
        *device_ptr = nullptr;
    }
#ifdef TURBO_BUFFER_CUDA
    if (!turbo_buffer::impl::lookup_mapped(host_ptr, device_ptr)) {
        return 0;
    }
    return 1;
#else
    (void)host_ptr;
    return 0;
#endif
}

#ifdef TURBO_BUFFER_CUDA

cudaError_t turbo_buffer_cuda_malloc(void **ptr, size_t bytes) {
    if (ptr == nullptr) {
        return cudaErrorInvalidValue;
    }
    *ptr = nullptr;
    const cudaError_t e = cudaMalloc(ptr, bytes);
    if (e == cudaSuccess && *ptr != nullptr) {
        turbo_buffer::impl::note_cuda_runtime_alloc_if_forward();
    }
    return e;
}

cudaError_t turbo_buffer_cuda_host_alloc(
    void **ptr,
    size_t bytes,
    unsigned int flags
) {
    if (ptr == nullptr) {
        return cudaErrorInvalidValue;
    }
    *ptr = nullptr;
    const cudaError_t e = cudaHostAlloc(ptr, bytes, flags);
    if (e == cudaSuccess && *ptr != nullptr) {
        turbo_buffer::impl::note_cuda_runtime_alloc_if_forward();
    }
    return e;
}

cudaError_t turbo_buffer_cuda_memcpy(
    void *dst,
    const void *src,
    size_t count,
    enum cudaMemcpyKind kind
) {
    if (kind == cudaMemcpyHostToDevice) {
        turbo_buffer::impl::note_cuda_forward_h2d(count);
    }
    return cudaMemcpy(dst, src, count, kind);
}

cudaError_t turbo_buffer_cuda_memcpy_async(
    void *dst,
    const void *src,
    size_t count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream
) {
    if (kind == cudaMemcpyHostToDevice) {
        turbo_buffer::impl::note_cuda_forward_h2d(count);
    }
    return cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t turbo_buffer_cuda_memcpy2d(
    void *dst,
    size_t dpitch,
    const void *src,
    size_t spitch,
    size_t width,
    size_t height,
    enum cudaMemcpyKind kind
) {
    if (kind == cudaMemcpyHostToDevice) {
        turbo_buffer::impl::note_cuda_forward_h2d(width * height);
    }
    return cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

#endif

} // extern "C"
