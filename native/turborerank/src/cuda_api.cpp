// SPDX-License-Identifier: Apache-2.0
//
// CUDA presence + fail-loud copy. Device BERT lives in bert_cuda.cu
// when TURBORERANK_CUDA is set.

#include "cuda_api.hpp"

#include <string>

#ifdef TURBORERANK_CUDA
#include <cuda_runtime.h>
#endif

namespace turborerank {
namespace impl {

bool cuda_compiled() {
#ifdef TURBORERANK_CUDA
    return true;
#else
    return false;
#endif
}

bool cuda_device_present(std::string *why) {
#ifdef TURBORERANK_CUDA
    int count = 0;
    const cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        if (why) {
            *why = std::string("TURBORERANK_DEVICE_CUDA requested but "
                               "cudaGetDeviceCount failed (") +
                   cudaGetErrorString(e) + "). Refusing CPU fallback.";
        }
        return false;
    }
    if (count <= 0) {
        if (why) {
            *why = "TURBORERANK_DEVICE_CUDA requested but CUDA runtime "
                   "reports zero devices. Refusing CPU fallback.";
        }
        return false;
    }
    return true;
#else
    if (why) {
        *why = "TURBORERANK_DEVICE_CUDA requested but this binary was built "
               "without CUDA (nvcc/cudart). Refusing CPU fallback.";
    }
    return false;
#endif
}

bool cuda_gpu_name(std::string *name) {
#ifdef TURBORERANK_CUDA
    cudaDeviceProp prop {};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        if (name) {
            *name = {};
        }
        return false;
    }
    if (name) {
        *name = prop.name;
    }
    return true;
#else
    if (name) {
        *name = {};
    }
    return false;
#endif
}

#ifndef TURBORERANK_CUDA

const char *cuda_gemm_backend() { return "unavailable"; }

bool cuda_resources_init(
    CudaResources *r,
    const BertConfig &,
    const BertWeights &,
    turbo_buffer_arena *,
    std::string *err
) {
    if (r) {
        *r = CudaResources{};
    }
    if (err) {
        *err = "CUDA MiniLM CE is not compiled into this binary";
    }
    return false;
}

void cuda_resources_free(CudaResources *r) {
    if (r) {
        *r = CudaResources{};
    }
}

bool bert_forward_row_cuda(
    CudaResources *,
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
        *err = "CUDA MiniLM CE is not compiled into this binary";
    }
    return false;
}

#endif

} // namespace impl
} // namespace turborerank
