// SPDX-License-Identifier: Apache-2.0
//
// Device-side MiniLM pool for TurboEmbed ORT CUDA / TensorRT.
// Hidden states stay on DEVICE. The kernel writes the pooled row into
// a device-visible pointer (PINNED mapped or DEVICE). No activation D2H.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// hidden_dev: f32 [batch, seq, dim] DEVICE
// mask_dev:   i64 [batch, seq] device-visible (PINNED mapped ok)
// out_dev:    f32 [batch, dim] device-visible (PINNED mapped ok)
// normalize:  non-zero → L2 each row (catalog MiniLM)
// Returns 0 on success, else a cudaError_t (or 1 for bad args).
int turboembed_cuda_pool_mean_l2(
    const float *hidden_dev,
    const int64_t *mask_dev,
    float *out_dev,
    int batch,
    int seq,
    int dim,
    int normalize
);

int turboembed_cuda_pool_cls_l2(
    const float *hidden_dev,
    float *out_dev,
    int batch,
    int seq,
    int dim,
    int normalize
);

#ifdef __cplusplus
}
#endif
