// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "internal.hpp"

#include <string>

namespace turborerank {
namespace impl {

/** True when this binary was compiled with Metal (`TURBORERANK_METAL`). */
bool metal_compiled();

/**
 * Runtime probe: compiled with Metal *and* an MTL GPU device exists.
 * On failure writes a message that always refuses CPU fallback.
 */
bool metal_device_present(std::string *why);

/** GPU name (e.g. "Apple M2"), or empty if Metal is missing. */
bool metal_gpu_name(std::string *name);

/**
 * MTLResourceStorageModeShared token workspace. Caller writes into
 * unified memory. Pointers are page-aligned (hence 64-byte).
 */
void *metal_shared_alloc_bytes(size_t bytes, Status *status);
void metal_shared_free_bytes(void *ptr);

/** Look up the MTLBuffer for a shared pointer allocated by us. */
bool metal_shared_owns(const void *ptr);

bool metal_resources_init(
    MetalResources *r,
    const BertConfig &cfg,
    const BertWeights &w,
    std::string *err
);

void metal_resources_free(MetalResources *r);

/**
 * Metal MiniLM CE. Token pointers must be MTL shared (caller-written
 * unified memory). Kernels bind those MTLBuffers — no std::vector and
 * no extra token copy. Weights/activations live in MTL buffers reserved
 * at load.
 */
bool bert_forward_row_metal(
    MetalResources *r,
    const BertConfig &cfg,
    const int32_t *input_ids,
    const int32_t *attention_mask,
    const int32_t *token_type_ids,
    const int32_t *position_ids,
    uint32_t seq,
    float *logit_out,
    std::string *err
);

} // namespace impl
} // namespace turborerank
