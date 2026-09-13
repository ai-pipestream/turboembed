// SPDX-License-Identifier: Apache-2.0

#include "cuda_api.hpp"
#include "internal.hpp"
#include "metal_api.hpp"
#include "ov_api.hpp"
#include "turbo_buffer.h"

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
    turbo_buffer_note_alloc();
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
    void *ptr = nullptr;
    const turbo_buffer_status st = turbo_buffer_raw_alloc(
        TURBO_BUFFER_DEVICE_CUDA, TURBO_BUFFER_PLACE_PINNED, bytes, &ptr
    );
    if (st != TURBO_BUFFER_OK || ptr == nullptr) {
        if (status) {
            if (st == TURBO_BUFFER_ERR_NOT_IMPLEMENTED) {
                *status = Status::Unavailable;
            } else if (st == TURBO_BUFFER_ERR_UNAVAILABLE) {
                *status = Status::Unavailable;
            } else if (st == TURBO_BUFFER_ERR_INVALID_ARGUMENT) {
                *status = Status::InvalidArgument;
            } else {
                *status = Status::OutOfMemory;
            }
        }
        return nullptr;
    }
    if (status) {
        *status = Status::Ok;
    }
    return ptr;
}

void pinned_free_bytes(void *ptr) {
    turbo_buffer_raw_free(
        TURBO_BUFFER_DEVICE_CUDA, TURBO_BUFFER_PLACE_PINNED, ptr
    );
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
    std::string metal_why;
    switch (d) {
    case TURBORERANK_DEVICE_AUTO:
        if (cuda_device_present(&cuda_why)) {
            return false;
        }
        if (ov_gpu_present(&ov_why)) {
            return false;
        }
        if (metal_device_present(&metal_why)) {
            return false;
        }
        if (why) {
            *why = "TURBORERANK_DEVICE_AUTO requested host-default GPU; " +
                   cuda_why + " " + ov_why + " " + metal_why +
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
        if (metal_device_present(&metal_why)) {
            return false;
        }
        if (why) {
            *why = metal_why.empty()
                       ? "TURBORERANK_DEVICE_METAL requested but Metal is "
                         "unavailable (MTLResourceStorageModeShared / first-"
                         "party Metal MiniLM CE). Refusing CPU fallback."
                       : metal_why;
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
        if (metal_device_present(&why)) {
            return TURBORERANK_DEVICE_METAL;
        }
    }
    return requested;
}

turbo_buffer_device buffer_device_for(turborerank_device d) {
    switch (d) {
    case TURBORERANK_DEVICE_CUDA:
        return TURBO_BUFFER_DEVICE_CUDA;
    case TURBORERANK_DEVICE_OPENVINO_CPU:
    case TURBORERANK_DEVICE_OPENVINO_GPU:
        return TURBO_BUFFER_DEVICE_ZE;
    case TURBORERANK_DEVICE_METAL:
        return TURBO_BUFFER_DEVICE_METAL;
    default:
        return TURBO_BUFFER_DEVICE_CPU;
    }
}

turbo_buffer_placement host_visible_placement(turborerank_device d) {
    switch (d) {
    case TURBORERANK_DEVICE_CUDA:
        return TURBO_BUFFER_PLACE_PINNED;
    case TURBORERANK_DEVICE_OPENVINO_GPU:
        return TURBO_BUFFER_PLACE_SHARED;
    case TURBORERANK_DEVICE_OPENVINO_CPU:
        return TURBO_BUFFER_PLACE_HOST;
    case TURBORERANK_DEVICE_METAL:
        return TURBO_BUFFER_PLACE_SHARED;
    default:
        return TURBO_BUFFER_PLACE_HOST;
    }
}

} // namespace impl

void alloc_counter_reset() {
    turbo_buffer_alloc_counter_reset();
}

uint64_t alloc_counter_value() {
    return turbo_buffer_alloc_counter();
}

void note_alloc() {
    turbo_buffer_note_alloc();
}

} // namespace turborerank
