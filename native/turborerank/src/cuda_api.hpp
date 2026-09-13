// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "internal.hpp"

#include <string>

namespace turborerank {
namespace impl {

/** True when this binary was compiled with nvcc + cudart (`TURBORERANK_CUDA`). */
bool cuda_compiled();

/**
 * Runtime probe: compiled with CUDA *and* `cudaGetDeviceCount` > 0.
 * On failure writes a message that always refuses CPU fallback.
 */
bool cuda_device_present(std::string *why);

/** `cudaGetDeviceProperties` name, or empty if CUDA is missing. */
bool cuda_gpu_name(std::string *name);

void *pinned_alloc_bytes(size_t bytes, Status *status);
void pinned_free_bytes(void *ptr);

bool cuda_resources_init(
    CudaResources *r,
    const BertConfig &cfg,
    const BertWeights &w,
    std::string *err
);

void cuda_resources_free(CudaResources *r);

/**
 * Device MiniLM CE. Tokens are read from caller pinned (or any host)
 * pointers; one H2D of the packed int32 row, then cuBLAS + kernels.
 * No host heap allocation.
 */
bool bert_forward_row_cuda(
    CudaResources *r,
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
