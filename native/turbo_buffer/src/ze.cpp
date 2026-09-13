// SPDX-License-Identifier: Apache-2.0
//
// Level Zero USM: HOST / SHARED / DEVICE. SHARED and DEVICE require a
// GPU device handle — they never become HOST. Machine B proves the live
// path. Missing L0 is NOT_IMPLEMENTED, not a CPU stand-in.

#include "internal.hpp"

#ifdef TURBO_BUFFER_ZE
#include <level_zero/ze_api.h>
#endif

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace turbo_buffer {
namespace impl {

#ifdef TURBO_BUFFER_ZE
namespace {

struct L0State {
    ze_driver_handle_t drv = nullptr;
    ze_device_handle_t dev = nullptr;
    ze_context_handle_t ctx = nullptr;
    ze_command_queue_handle_t q = nullptr;
    ze_command_list_handle_t cl = nullptr;
    bool ready = false;
    bool have_gpu = false;
    std::string why;
};

L0State &l0();

bool ensure_copy_queue() {
    L0State &s = l0();
    if (s.q != nullptr && s.cl != nullptr) {
        return true;
    }
    if (!s.ready || s.ctx == nullptr || s.dev == nullptr || !s.have_gpu) {
        return false;
    }
    ze_command_queue_desc_t qd {};
    qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qd.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    qd.ordinal = 0;
    if (zeCommandQueueCreate(s.ctx, s.dev, &qd, &s.q) != ZE_RESULT_SUCCESS) {
        s.q = nullptr;
        return false;
    }
    ze_command_list_desc_t ld {};
    ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    if (zeCommandListCreate(s.ctx, s.dev, &ld, &s.cl) != ZE_RESULT_SUCCESS) {
        (void)zeCommandQueueDestroy(s.q);
        s.q = nullptr;
        s.cl = nullptr;
        return false;
    }
    return true;
}

turbo_buffer_status placement_of(const void *ptr, turbo_buffer_placement *out) {
    L0State &s = l0();
    if (ptr == nullptr || s.ctx == nullptr) {
        return TURBO_BUFFER_ERR_NOT_FOUND;
    }
    ze_memory_allocation_properties_t props {};
    props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    ze_device_handle_t assoc = nullptr;
    if (zeMemGetAllocProperties(s.ctx, ptr, &props, &assoc) != ZE_RESULT_SUCCESS) {
        return TURBO_BUFFER_ERR_NOT_FOUND;
    }
    turbo_buffer_placement place = TURBO_BUFFER_PLACE_HOST;
    switch (props.type) {
    case ZE_MEMORY_TYPE_HOST:
        place = TURBO_BUFFER_PLACE_HOST;
        break;
    case ZE_MEMORY_TYPE_SHARED:
        place = TURBO_BUFFER_PLACE_SHARED;
        break;
    case ZE_MEMORY_TYPE_DEVICE:
        place = TURBO_BUFFER_PLACE_DEVICE;
        break;
    default:
        return TURBO_BUFFER_ERR_NOT_FOUND;
    }
    if (out != nullptr) {
        *out = place;
    }
    return TURBO_BUFFER_OK;
}

L0State &l0() {
    static L0State s;
    return s;
}

void l0_init_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        L0State &s = l0();
        const ze_result_t init = zeInit(0);
        if (init != ZE_RESULT_SUCCESS) {
            s.why = "Level Zero zeInit failed; cannot allocate USM. "
                    "Refusing CPU fallback.";
            return;
        }
        uint32_t ndrv = 0;
        if (zeDriverGet(&ndrv, nullptr) != ZE_RESULT_SUCCESS || ndrv == 0) {
            s.why = "Level Zero reports zero drivers. Refusing CPU fallback.";
            return;
        }
        std::vector<ze_driver_handle_t> drvs(ndrv);
        zeDriverGet(&ndrv, drvs.data());
        for (uint32_t i = 0; i < ndrv; ++i) {
            uint32_t ndev = 0;
            if (zeDeviceGet(drvs[i], &ndev, nullptr) != ZE_RESULT_SUCCESS ||
                ndev == 0) {
                continue;
            }
            std::vector<ze_device_handle_t> devs(ndev);
            zeDeviceGet(drvs[i], &ndev, devs.data());
            for (uint32_t j = 0; j < ndev; ++j) {
                ze_device_properties_t prop {};
                prop.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                if (zeDeviceGetProperties(devs[j], &prop) != ZE_RESULT_SUCCESS) {
                    continue;
                }
                if (prop.type != ZE_DEVICE_TYPE_GPU) {
                    continue;
                }
                ze_context_desc_t cd {};
                cd.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
                if (zeContextCreate(drvs[i], &cd, &s.ctx) != ZE_RESULT_SUCCESS) {
                    continue;
                }
                s.drv = drvs[i];
                s.dev = devs[j];
                s.have_gpu = true;
                s.ready = true;
                return;
            }
        }
        // Host-only context: zeMemAllocHost for OV-CPU. SHARED/DEVICE stay
        // UNAVAILABLE — this is not a silent GPU stand-in.
        if (!drvs.empty()) {
            ze_context_desc_t cd {};
            cd.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
            if (zeContextCreate(drvs[0], &cd, &s.ctx) == ZE_RESULT_SUCCESS) {
                s.drv = drvs[0];
                s.ready = true;
                return;
            }
        }
        s.why = "Level Zero found no usable context for USM. Refusing CPU "
                "fallback.";
    });
}

} // namespace
#endif

turbo_buffer_status ze_probe(turbo_buffer_placement placement) {
    if (placement != TURBO_BUFFER_PLACE_HOST &&
        placement != TURBO_BUFFER_PLACE_SHARED &&
        placement != TURBO_BUFFER_PLACE_DEVICE) {
        set_tls_error(
            "ZE arena implements HOST / SHARED / DEVICE USM only; "
            "refusing a silent CPU remap"
        );
        return TURBO_BUFFER_ERR_NOT_IMPLEMENTED;
    }
#ifdef TURBO_BUFFER_ZE
    l0_init_once();
    if (!l0().ready) {
        set_tls_error(
            l0().why.empty()
                ? "Level Zero USM is not available. Refusing CPU fallback."
                : l0().why
        );
        return TURBO_BUFFER_ERR_UNAVAILABLE;
    }
    if ((placement == TURBO_BUFFER_PLACE_SHARED ||
         placement == TURBO_BUFFER_PLACE_DEVICE) &&
        !l0().have_gpu) {
        set_tls_error(
            "ZE SHARED/DEVICE requested but Level Zero has no GPU device. "
            "Refusing HOST/CPU fallback."
        );
        return TURBO_BUFFER_ERR_UNAVAILABLE;
    }
    return TURBO_BUFFER_OK;
#else
    set_tls_error(
        "TURBO_BUFFER_DEVICE_ZE requested but this binary was built "
        "without Level Zero. Refusing CPU fallback."
    );
    return TURBO_BUFFER_ERR_NOT_IMPLEMENTED;
#endif
}

void *ze_alloc(
    turbo_buffer_placement placement,
    size_t bytes,
    turbo_buffer_status *status
) {
#ifdef TURBO_BUFFER_ZE
    if (bytes == 0) {
        if (status) {
            *status = TURBO_BUFFER_ERR_INVALID_ARGUMENT;
        }
        return nullptr;
    }
    l0_init_once();
    if (!l0().ready || l0().ctx == nullptr) {
        if (status) {
            *status = TURBO_BUFFER_ERR_UNAVAILABLE;
        }
        return nullptr;
    }
    if ((placement == TURBO_BUFFER_PLACE_SHARED ||
         placement == TURBO_BUFFER_PLACE_DEVICE) &&
        (l0().dev == nullptr || !l0().have_gpu)) {
        if (status) {
            *status = TURBO_BUFFER_ERR_UNAVAILABLE;
        }
        return nullptr;
    }
    void *ptr = nullptr;
    ze_result_t zr = ZE_RESULT_ERROR_UNKNOWN;
    if (placement == TURBO_BUFFER_PLACE_SHARED) {
        ze_device_mem_alloc_desc_t dd {};
        dd.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        ze_host_mem_alloc_desc_t hd {};
        hd.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
        zr = zeMemAllocShared(l0().ctx, &dd, &hd, bytes, 64, l0().dev, &ptr);
    } else if (placement == TURBO_BUFFER_PLACE_DEVICE) {
        ze_device_mem_alloc_desc_t dd {};
        dd.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        zr = zeMemAllocDevice(l0().ctx, &dd, bytes, 64, l0().dev, &ptr);
    } else {
        ze_host_mem_alloc_desc_t hd {};
        hd.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
        zr = zeMemAllocHost(l0().ctx, &hd, bytes, 64, &ptr);
    }
    if (zr != ZE_RESULT_SUCCESS || ptr == nullptr) {
        if (status) {
            *status = TURBO_BUFFER_ERR_OUT_OF_MEMORY;
        }
        return nullptr;
    }
    if (placement != TURBO_BUFFER_PLACE_DEVICE) {
        std::memset(ptr, 0, bytes);
    }
    turbo_buffer_note_alloc();
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

void ze_free(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
#ifdef TURBO_BUFFER_ZE
    l0_init_once();
    if (l0().ctx != nullptr) {
        (void)zeMemFree(l0().ctx, ptr);
    }
#else
    (void)ptr;
#endif
}

turbo_buffer_status ze_query(const void *ptr, turbo_buffer_placement *out) {
#ifdef TURBO_BUFFER_ZE
    if (ptr == nullptr) {
        set_tls_error("ze_query: null pointer");
        return TURBO_BUFFER_ERR_INVALID_ARGUMENT;
    }
    l0_init_once();
    if (!l0().ready) {
        set_tls_error(
            l0().why.empty()
                ? "Level Zero USM is not available. Refusing CPU fallback."
                : l0().why
        );
        return TURBO_BUFFER_ERR_UNAVAILABLE;
    }
    const turbo_buffer_status st = placement_of(ptr, out);
    if (st != TURBO_BUFFER_OK) {
        set_tls_error("ze_query: pointer is not a Level Zero USM allocation");
    }
    return st;
#else
    (void)ptr;
    (void)out;
    set_tls_error(
        "turbo_buffer_ze_query requested but this binary was built "
        "without Level Zero. Refusing CPU fallback."
    );
    return TURBO_BUFFER_ERR_NOT_IMPLEMENTED;
#endif
}

turbo_buffer_status ze_memcpy(void *dst, const void *src, size_t bytes) {
#ifdef TURBO_BUFFER_ZE
    if (dst == nullptr || src == nullptr || bytes == 0) {
        set_tls_error("ze_memcpy: null pointer or zero bytes");
        return TURBO_BUFFER_ERR_INVALID_ARGUMENT;
    }
    l0_init_once();
    if (!l0().ready || l0().ctx == nullptr) {
        set_tls_error(
            l0().why.empty()
                ? "Level Zero USM is not available. Refusing CPU fallback."
                : l0().why
        );
        return TURBO_BUFFER_ERR_UNAVAILABLE;
    }
    turbo_buffer_placement dst_p = TURBO_BUFFER_PLACE_HOST;
    turbo_buffer_placement src_p = TURBO_BUFFER_PLACE_HOST;
    if (placement_of(dst, &dst_p) != TURBO_BUFFER_OK ||
        placement_of(src, &src_p) != TURBO_BUFFER_OK) {
        set_tls_error("ze_memcpy: src/dst are not Level Zero USM");
        return TURBO_BUFFER_ERR_NOT_FOUND;
    }
    const bool needs_device =
        dst_p == TURBO_BUFFER_PLACE_DEVICE || src_p == TURBO_BUFFER_PLACE_DEVICE;
    if (!needs_device) {
        std::memcpy(dst, src, bytes);
        return TURBO_BUFFER_OK;
    }
    if (!ensure_copy_queue()) {
        set_tls_error(
            "ze_memcpy: DEVICE copy needs a Level Zero GPU queue. "
            "Refusing a host memcpy stand-in."
        );
        return TURBO_BUFFER_ERR_UNAVAILABLE;
    }
    L0State &s = l0();
    if (zeCommandListReset(s.cl) != ZE_RESULT_SUCCESS) {
        set_tls_error("ze_memcpy: command list reset failed");
        return TURBO_BUFFER_ERR_INTERNAL;
    }
    if (zeCommandListAppendMemoryCopy(
            s.cl, dst, src, bytes, nullptr, 0, nullptr
        ) != ZE_RESULT_SUCCESS) {
        set_tls_error("ze_memcpy: zeCommandListAppendMemoryCopy failed");
        return TURBO_BUFFER_ERR_INTERNAL;
    }
    if (zeCommandListClose(s.cl) != ZE_RESULT_SUCCESS) {
        set_tls_error("ze_memcpy: command list close failed");
        return TURBO_BUFFER_ERR_INTERNAL;
    }
    if (zeCommandQueueExecuteCommandLists(s.q, 1, &s.cl, nullptr) !=
        ZE_RESULT_SUCCESS) {
        set_tls_error("ze_memcpy: execute failed");
        return TURBO_BUFFER_ERR_INTERNAL;
    }
    if (zeCommandQueueSynchronize(s.q, UINT64_MAX) != ZE_RESULT_SUCCESS) {
        set_tls_error("ze_memcpy: synchronize failed");
        return TURBO_BUFFER_ERR_INTERNAL;
    }
    return TURBO_BUFFER_OK;
#else
    (void)dst;
    (void)src;
    (void)bytes;
    set_tls_error(
        "turbo_buffer_ze_memcpy requested but this binary was built "
        "without Level Zero. Refusing CPU fallback."
    );
    return TURBO_BUFFER_ERR_NOT_IMPLEMENTED;
#endif
}

} // namespace impl
} // namespace turbo_buffer
