// SPDX-License-Identifier: Apache-2.0
//
// Metal presence + fail-loud stubs. Live MTL shared alloc lives in
// metal.mm when TURBO_BUFFER_METAL is set (Machine C).

#include "internal.hpp"

#include <string>

#ifndef TURBO_BUFFER_METAL

namespace turbo_buffer {
namespace impl {

turbo_buffer_status metal_probe(turbo_buffer_placement placement) {
    (void)placement;
    set_tls_error(
        "TURBO_BUFFER_DEVICE_METAL requested but this binary was built "
        "without Metal (MTL / Apple GPU). Refusing CPU fallback."
    );
    return TURBO_BUFFER_ERR_NOT_IMPLEMENTED;
}

void *metal_alloc(size_t bytes, turbo_buffer_status *status) {
    (void)bytes;
    if (status) {
        *status = TURBO_BUFFER_ERR_NOT_IMPLEMENTED;
    }
    return nullptr;
}

void metal_free(void *ptr) {
    (void)ptr;
}

} // namespace impl
} // namespace turbo_buffer

extern "C" {

int turbo_buffer_metal_owns(const void *ptr) {
    (void)ptr;
    return 0;
}

int turbo_buffer_metal_lookup(
    const void *ptr,
    void **out_native,
    size_t *out_offset
) {
    (void)ptr;
    if (out_native) {
        *out_native = nullptr;
    }
    if (out_offset) {
        *out_offset = 0;
    }
    return 0;
}

} // extern "C"

#endif /* !TURBO_BUFFER_METAL */
