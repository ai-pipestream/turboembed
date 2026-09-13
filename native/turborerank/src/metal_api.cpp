// SPDX-License-Identifier: Apache-2.0
//
// Metal presence + fail-loud stubs. Device BERT lives in metal_api.mm
// when TURBORERANK_METAL is set (macOS + Metal.framework).

#include "metal_api.hpp"

#include <string>

namespace turborerank {
namespace impl {

bool metal_compiled() {
#ifdef TURBORERANK_METAL
    return true;
#else
    return false;
#endif
}

#ifndef TURBORERANK_METAL

bool metal_device_present(std::string *why) {
    if (why) {
        *why = "TURBORERANK_DEVICE_METAL requested but this binary was built "
               "without Metal (MTL / Apple GPU). Refusing CPU fallback.";
    }
    return false;
}

bool metal_gpu_name(std::string *name) {
    if (name) {
        *name = {};
    }
    return false;
}

void *metal_shared_alloc_bytes(size_t bytes, Status *status) {
    (void)bytes;
    if (status) {
        *status = Status::Unavailable;
    }
    return nullptr;
}

void metal_shared_free_bytes(void *ptr) {
    (void)ptr;
}

bool metal_shared_owns(const void *ptr) {
    (void)ptr;
    return false;
}

bool metal_resources_init(
    MetalResources *r,
    const BertConfig &,
    const BertWeights &,
    std::string *err
) {
    if (r) {
        *r = MetalResources{};
    }
    if (err) {
        *err = "Metal MiniLM CE is not compiled into this binary; refusing CPU "
               "fallback";
    }
    return false;
}

void metal_resources_free(MetalResources *r) {
    if (r) {
        *r = MetalResources{};
    }
}

bool bert_forward_row_metal(
    MetalResources *,
    const BertConfig &,
    const int32_t *,
    const int32_t *,
    const int32_t *,
    const int32_t *,
    uint32_t,
    float *,
    std::string *err
) {
    if (err) {
        *err = "Metal MiniLM CE is not compiled into this binary; refusing CPU "
               "fallback";
    }
    return false;
}

#endif

} // namespace impl
} // namespace turborerank
