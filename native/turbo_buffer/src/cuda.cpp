// SPDX-License-Identifier: Apache-2.0
//
// CUDA pinned (cudaHostAlloc) + device (cudaMalloc). Fail loud when this
// binary has no nvcc/cudart or the runtime reports zero devices.
// Machine A is the live proof host. Do not include cuda_runtime_hooks.hpp
// here — this file implements the real calls the intercept wraps.

#include "internal.hpp"

#ifdef TURBO_BUFFER_CUDA
#include <cuda_runtime.h>
#endif

#include <cstring>
#include <string>

namespace turbo_buffer {
namespace impl {

turbo_buffer_status cuda_probe(turbo_buffer_placement placement) {
    if (placement != TURBO_BUFFER_PLACE_PINNED &&
        placement != TURBO_BUFFER_PLACE_DEVICE) {
        set_tls_error(
            "CUDA arena implements PINNED (cudaHostAlloc) and DEVICE "
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
    void *ptr = nullptr;
    cudaError_t e = cudaSuccess;
    if (placement == TURBO_BUFFER_PLACE_PINNED) {
        e = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
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
        std::memset(ptr, 0, bytes);
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

#ifdef TURBO_BUFFER_CUDA
extern "C" {

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

} // extern "C"
#endif
