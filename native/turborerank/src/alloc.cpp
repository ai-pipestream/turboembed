// SPDX-License-Identifier: Apache-2.0

#include "internal.hpp"

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

void alloc_counter_reset() {
    g_allocs.store(0, std::memory_order_relaxed);
}

uint64_t alloc_counter_value() {
    return g_allocs.load(std::memory_order_relaxed);
}

namespace impl {

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
    const char *name = turborerank_device_name(d);
    std::string msg;
    switch (d) {
    case TURBORERANK_DEVICE_AUTO:
        msg = "TURBORERANK_DEVICE_AUTO requested host-default GPU; Phase 1 "
              "has no CUDA/Metal/OpenVINO GPU provider compiled. Refusing "
              "CPU fallback. Use TURBORERANK_DEVICE_CPU for the MiniLM CE "
              "kernel.";
        break;
    case TURBORERANK_DEVICE_CUDA:
        msg = "TURBORERANK_DEVICE_CUDA is not implemented in Phase 1 "
              "(cudaHostAlloc + ggml CUDA is later work). Refusing CPU "
              "fallback.";
        break;
    case TURBORERANK_DEVICE_TENSORRT:
        msg = "TURBORERANK_DEVICE_TENSORRT is not implemented in Phase 1. "
              "Refusing CUDA/CPU fallback.";
        break;
    case TURBORERANK_DEVICE_OPENVINO_GPU:
        msg = "TURBORERANK_DEVICE_OPENVINO_GPU is not implemented in Phase 1 "
              "(Level Zero USM + ov::Tensor). Refusing CPU fallback.";
        break;
    case TURBORERANK_DEVICE_OPENVINO_NPU:
        msg = "TURBORERANK_DEVICE_OPENVINO_NPU is not implemented in Phase 1. "
              "Refusing CPU fallback.";
        break;
    case TURBORERANK_DEVICE_OPENVINO_CPU:
        msg = "TURBORERANK_DEVICE_OPENVINO_CPU is not implemented in Phase 1 "
              "(OpenVINO CompiledModel). Use TURBORERANK_DEVICE_CPU for the "
              "first-party MiniLM CE kernel. Refusing a silent stand-in.";
        break;
    case TURBORERANK_DEVICE_METAL:
        msg = "TURBORERANK_DEVICE_METAL is not implemented in Phase 1 "
              "(MTL shared / MLX). Refusing CPU fallback.";
        break;
    default:
        return false;
    }
    (void)name;
    if (why) {
        *why = msg;
    }
    return true;
}

} // namespace impl
} // namespace turborerank
