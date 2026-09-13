// SPDX-License-Identifier: Apache-2.0

#include "cuda_api.hpp"
#include "internal.hpp"
#include "ov_api.hpp"

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <stdlib.h>
#endif

#ifdef TURBORERANK_CUDA
#include <cuda_runtime.h>
#endif

namespace turborerank {
namespace {

std::atomic<uint64_t> g_allocs{0};
thread_local std::string g_create_error;

} // namespace

void *aligned_alloc_bytes(size_t bytes, size_t alignment, Status *status) {
    if (bytes == 0) {
        if (status) {
            *status = Status::InvalidArgument;
        }
        return nullptr;
    }
    if (alignment < sizeof(void *)) {
        alignment = sizeof(void *);
    }
    // posix_memalign requires alignment to be a power of two and a
    // multiple of sizeof(void*).
    void *ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(bytes, alignment);
    if (ptr == nullptr) {
        if (status) {
            *status = Status::OutOfMemory;
        }
        return nullptr;
    }
#else
    const int rc = posix_memalign(&ptr, alignment, bytes);
    if (rc != 0 || ptr == nullptr) {
        if (status) {
            *status = Status::OutOfMemory;
        }
        return nullptr;
    }
#endif
    std::memset(ptr, 0, bytes);
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (status) {
        *status = Status::Ok;
    }
    return ptr;
}

void aligned_free_bytes(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

namespace impl {

void *pinned_alloc_bytes(size_t bytes, Status *status) {
#ifdef TURBORERANK_CUDA
    if (bytes == 0) {
        if (status) {
            *status = Status::InvalidArgument;
        }
        return nullptr;
    }
    void *ptr = nullptr;
    const cudaError_t e = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (e != cudaSuccess || ptr == nullptr) {
        if (status) {
            *status = Status::OutOfMemory;
        }
        return nullptr;
    }
    std::memset(ptr, 0, bytes);
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (status) {
        *status = Status::Ok;
    }
    return ptr;
#else
    (void)bytes;
    if (status) {
        *status = Status::Unavailable;
    }
    return nullptr;
#endif
}

void pinned_free_bytes(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
#ifdef TURBORERANK_CUDA
    (void)cudaFreeHost(ptr);
#else
    std::free(ptr);
#endif
}

void set_create_error(const std::string &msg) {
    g_create_error = msg;
}

const char *create_error() {
    return g_create_error.c_str();
}

void set_engine_error(turborerank_engine *engine, const std::string &msg) {
    if (engine != nullptr) {
        engine->last_error = msg;
    } else {
        set_create_error(msg);
    }
}

bool device_is_accelerator(turborerank_device d) {
    switch (d) {
    case TURBORERANK_DEVICE_AUTO:
    case TURBORERANK_DEVICE_CUDA:
    case TURBORERANK_DEVICE_TENSORRT:
    case TURBORERANK_DEVICE_OPENVINO_GPU:
    case TURBORERANK_DEVICE_OPENVINO_NPU:
    case TURBORERANK_DEVICE_METAL:
        return true;
    default:
        return false;
    }
}

bool accelerator_unavailable(turborerank_device d, std::string *why) {
    std::string cuda_why;
    std::string ov_why;
    switch (d) {
    case TURBORERANK_DEVICE_AUTO:
        if (cuda_device_present(&cuda_why)) {
            return false;
        }
        if (ov_gpu_present(&ov_why)) {
            return false;
        }
        if (why) {
            *why = "TURBORERANK_DEVICE_AUTO requested host-default GPU; " +
                   cuda_why + " " + ov_why +
                   " Refusing CPU fallback. Use TURBORERANK_DEVICE_CPU "
                   "for the MiniLM CE kernel.";
        }
        return true;
    case TURBORERANK_DEVICE_CUDA:
        if (cuda_device_present(&cuda_why)) {
            return false;
        }
        if (why) {
            *why = cuda_why;
        }
        return true;
    case TURBORERANK_DEVICE_TENSORRT:
        if (why) {
            *why = "TURBORERANK_DEVICE_TENSORRT is not implemented. "
                   "Refusing CUDA/CPU fallback.";
        }
        return true;
    case TURBORERANK_DEVICE_OPENVINO_GPU:
        if (ov_gpu_present(&ov_why)) {
            return false;
        }
        if (why) {
            *why = ov_why.empty()
                       ? "TURBORERANK_DEVICE_OPENVINO_GPU is unavailable "
                         "(Level Zero USM + ov::Tensor). Refusing CPU fallback."
                       : ov_why;
        }
        return true;
    case TURBORERANK_DEVICE_OPENVINO_NPU:
        if (why) {
            *why = "TURBORERANK_DEVICE_OPENVINO_NPU is not implemented. "
                   "Refusing CPU fallback.";
        }
        return true;
    case TURBORERANK_DEVICE_OPENVINO_CPU:
        if (ov_cpu_present(&ov_why)) {
            return false;
        }
        if (why) {
            *why = ov_why.empty()
                       ? "TURBORERANK_DEVICE_OPENVINO_CPU is not implemented "
                         "(OpenVINO CompiledModel). Use TURBORERANK_DEVICE_CPU "
                         "for the first-party MiniLM CE kernel. Refusing a "
                         "silent stand-in."
                       : ov_why;
        }
        return true;
    case TURBORERANK_DEVICE_METAL:
        if (why) {
            *why = "TURBORERANK_DEVICE_METAL is not implemented "
                   "(MTL shared / MLX). Refusing CPU fallback.";
        }
        return true;
    default:
        return false;
    }
}

turborerank_device resolve_create_device(turborerank_device requested) {
    if (requested == TURBORERANK_DEVICE_AUTO) {
        std::string why;
        if (cuda_device_present(&why)) {
            return TURBORERANK_DEVICE_CUDA;
        }
        if (ov_gpu_present(&why)) {
            return TURBORERANK_DEVICE_OPENVINO_GPU;
        }
    }
    return requested;
}

} // namespace impl

void alloc_counter_reset() {
    g_allocs.store(0, std::memory_order_relaxed);
}

uint64_t alloc_counter_value() {
    return g_allocs.load(std::memory_order_relaxed);
}

void note_alloc() {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
}

} // namespace turborerank
