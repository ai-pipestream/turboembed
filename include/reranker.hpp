// SPDX-License-Identifier: Apache-2.0
//
// TurboRerank C++ surface — buffer abstraction + forward that takes
// raw pointers. No std::vector for token / attention / type / position
// storage on the forward-pass hot path.
//
// The frozen language-stable ABI is include/turborerank.h. This header
// is the typed C++ view of the same contract (CPU + CUDA Phase 2a +
// OpenVINO Phase 2b + Metal Phase 2c; TensorRT / NPU still fail loud).

#ifndef TURBORERANK_RERANKER_HPP
#define TURBORERANK_RERANKER_HPP

#include "turborerank.h"

#include <cstddef>
#include <cstdint>

namespace turborerank {

inline constexpr uint32_t kAbiVersion = TURBORERANK_ABI_VERSION;
inline constexpr uint32_t kCpuAlignment = 64;
/** cudaHostAllocMapped is page-aligned; we still require 64-byte for ggml views. */
inline constexpr uint32_t kPinnedAlignment = 64;
inline constexpr int32_t kPadId = 0;
inline constexpr int32_t kUnkId = 100;
inline constexpr int32_t kClsId = 101;
inline constexpr int32_t kSepId = 102;
inline constexpr int32_t kMaskId = 103;
inline constexpr uint32_t kDefaultMaxLength = 512;
inline constexpr uint32_t kSpecials = 3; // [CLS] + [SEP] + [SEP]

enum class Device : uint32_t {
    Auto = TURBORERANK_DEVICE_AUTO,
    Cpu = TURBORERANK_DEVICE_CPU,
    Cuda = TURBORERANK_DEVICE_CUDA,
    TensorRt = TURBORERANK_DEVICE_TENSORRT,
    OpenVinoCpu = TURBORERANK_DEVICE_OPENVINO_CPU,
    OpenVinoGpu = TURBORERANK_DEVICE_OPENVINO_GPU,
    OpenVinoNpu = TURBORERANK_DEVICE_OPENVINO_NPU,
    Metal = TURBORERANK_DEVICE_METAL,
    Mock = TURBORERANK_DEVICE_MOCK
};

enum class Status : uint32_t {
    Ok = TURBORERANK_OK,
    InvalidArgument = TURBORERANK_ERR_INVALID_ARGUMENT,
    NotFound = TURBORERANK_ERR_NOT_FOUND,
    NotImplemented = TURBORERANK_ERR_NOT_IMPLEMENTED,
    Unavailable = TURBORERANK_ERR_UNAVAILABLE,
    Internal = TURBORERANK_ERR_INTERNAL,
    OutOfMemory = TURBORERANK_ERR_OUT_OF_MEMORY,
    UnsupportedDevice = TURBORERANK_ERR_UNSUPPORTED_DEVICE
};

enum class Truncation : uint32_t {
    LongestFirst = TURBORERANK_TRUNC_LONGEST_FIRST,
    QueryPriority = TURBORERANK_TRUNC_QUERY_PRIORITY,
    Error = TURBORERANK_TRUNC_ERROR
};

enum class Activation : uint32_t {
    Sigmoid = TURBORERANK_ACT_SIGMOID,
    Identity = TURBORERANK_ACT_IDENTITY
};

/** Non-owning view of a device token workspace. */
struct TokenSpan {
    int32_t *input_ids;
    int32_t *attention_mask;
    int32_t *token_type_ids;
    int32_t *position_ids;
    uint32_t batch;
    uint32_t seq;
    uint32_t row_stride;
};

inline TokenSpan row_at(const TokenSpan &buf, uint32_t row) {
    TokenSpan out = buf;
    if (buf.input_ids != nullptr && row < buf.batch) {
        const size_t off = static_cast<size_t>(row) * buf.row_stride;
        out.input_ids = buf.input_ids + off;
        out.attention_mask = buf.attention_mask + off;
        out.token_type_ids = buf.token_type_ids + off;
        out.position_ids = buf.position_ids + off;
        out.batch = 1;
    }
    return out;
}

/**
 * Write [CLS] q [SEP] d [SEP] into caller memory. `q` / `d` are raw
 * WordPiece ids without specials. Dest pointers must have `seq`
 * capacity. Returns packed length (including specials) or 0 on error
 * (`*status` set).
 *
 * No heap allocation.
 */
uint32_t pack_pair_ids(
    int32_t *input_ids,
    int32_t *attention_mask,
    int32_t *token_type_ids,
    int32_t *position_ids,
    uint32_t seq_capacity,
    const int32_t *query_ids,
    size_t n_query,
    const int32_t *doc_ids,
    size_t n_doc,
    Truncation truncation,
    uint32_t max_length,
    Status *status
);

/** Load-time weight copy only. Token/activation scratch is turbo_buffer. */
void *aligned_alloc_bytes(size_t bytes, size_t alignment, Status *status);

void aligned_free_bytes(void *ptr);

/** Reset / read the process-local alloc counter (tests). */
void alloc_counter_reset();
uint64_t alloc_counter_value();
/** Count a device/USM allocation the same way as posix_memalign. */
void note_alloc();

inline float sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = __builtin_expf(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = __builtin_expf(x);
    return z / (1.0f + z);
}

} // namespace turborerank

#endif /* TURBORERANK_RERANKER_HPP */
