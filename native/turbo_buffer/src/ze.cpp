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
    bool ready = false;
    bool have_gpu = false;
    std::string why;
};

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

} // namespace impl
} // namespace turbo_buffer
