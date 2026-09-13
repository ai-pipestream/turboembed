// SPDX-License-Identifier: Apache-2.0
//
// MTLResourceStorageModeShared token/activation workspace for Machine C.
// Caller writes unified memory. Pointers are page-aligned (hence 64-byte).

#ifdef TURBO_BUFFER_METAL

#include "internal.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace turbo_buffer {
namespace impl {

struct SharedEntry {
    id<MTLBuffer> buffer = nil;
    size_t bytes = 0;
};

struct SharedRegistry {
    std::mutex mu;
    std::unordered_map<const void *, SharedEntry> map;
};

SharedRegistry &registry() {
    static SharedRegistry r;
    return r;
}

struct MetalState {
    id<MTLDevice> device = nil;
    bool ready = false;
    std::string why;
};

MetalState &state() {
    static MetalState s;
    return s;
}

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        MetalState &s = state();
        s.device = MTLCreateSystemDefaultDevice();
        if (s.device == nil) {
            s.why = "TURBO_BUFFER_DEVICE_METAL requested but no MTL GPU is "
                    "present. Refusing CPU fallback.";
            return;
        }
        s.ready = true;
    });
}

turbo_buffer_status metal_probe(turbo_buffer_placement placement) {
    if (placement != TURBO_BUFFER_PLACE_SHARED &&
        placement != TURBO_BUFFER_PLACE_HOST) {
        set_tls_error(
            "Metal arena implements SHARED (and HOST as unified-memory "
            "alias) only; refusing DEVICE/PINNED remap"
        );
        return TURBO_BUFFER_ERR_NOT_IMPLEMENTED;
    }
    init_once();
    if (!state().ready) {
        set_tls_error(
            state().why.empty()
                ? "TURBO_BUFFER_DEVICE_METAL requested but Metal is "
                  "unavailable. Refusing CPU fallback."
                : state().why
        );
        return TURBO_BUFFER_ERR_UNAVAILABLE;
    }
    return TURBO_BUFFER_OK;
}

void *metal_alloc(size_t bytes, turbo_buffer_status *status) {
    if (bytes == 0) {
        if (status) {
            *status = TURBO_BUFFER_ERR_INVALID_ARGUMENT;
        }
        return nullptr;
    }
    init_once();
    if (!state().ready || state().device == nil) {
        if (status) {
            *status = TURBO_BUFFER_ERR_UNAVAILABLE;
        }
        return nullptr;
    }
    id<MTLBuffer> buf = [state().device newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];
    if (buf == nil || buf.contents == nullptr) {
        if (status) {
            *status = TURBO_BUFFER_ERR_OUT_OF_MEMORY;
        }
        return nullptr;
    }
    void *ptr = buf.contents;
    if ((reinterpret_cast<uintptr_t>(ptr) % 64u) != 0) {
        if (status) {
            *status = TURBO_BUFFER_ERR_INTERNAL;
        }
        return nullptr;
    }
    std::memset(ptr, 0, bytes);
    {
        SharedRegistry &reg = registry();
        std::lock_guard<std::mutex> lock(reg.mu);
        reg.map[ptr] = SharedEntry{buf, bytes};
    }
    turbo_buffer_note_alloc();
    if (status) {
        *status = TURBO_BUFFER_OK;
    }
    return ptr;
}

void metal_free(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    SharedRegistry &reg = registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    reg.map.erase(ptr);
}

} // namespace impl
} // namespace turbo_buffer

extern "C" {

int turbo_buffer_metal_owns(const void *ptr) {
    if (ptr == nullptr) {
        return 0;
    }
    turbo_buffer::impl::SharedRegistry &reg = turbo_buffer::impl::registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    return reg.map.find(ptr) != reg.map.end() ? 1 : 0;
}

int turbo_buffer_metal_lookup(
    const void *ptr,
    void **out_native,
    size_t *out_offset
) {
    if (out_native) {
        *out_native = nullptr;
    }
    if (out_offset) {
        *out_offset = 0;
    }
    if (ptr == nullptr) {
        return 0;
    }
    turbo_buffer::impl::SharedRegistry &reg = turbo_buffer::impl::registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    auto it = reg.map.find(ptr);
    if (it != reg.map.end()) {
        if (out_native) {
            *out_native = (__bridge void *)it->second.buffer;
        }
        if (out_offset) {
            *out_offset = 0;
        }
        return 1;
    }
    const char *p = static_cast<const char *>(ptr);
    for (const auto &kv : reg.map) {
        const char *start = static_cast<const char *>(kv.first);
        const size_t n = kv.second.bytes;
        if (p >= start && p < start + static_cast<ptrdiff_t>(n)) {
            if (out_native) {
                *out_native = (__bridge void *)kv.second.buffer;
            }
            if (out_offset) {
                *out_offset = static_cast<size_t>(p - start);
            }
            return 1;
        }
    }
    return 0;
}

} // extern "C"

#endif /* TURBO_BUFFER_METAL */
