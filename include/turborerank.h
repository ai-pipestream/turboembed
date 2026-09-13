/* SPDX-License-Identifier: Apache-2.0
 *
 * TurboRerank C ABI — frozen surface v1.
 *
 * Library-first cross-encoder (BERT) reranker. Same ownership spirit as
 * include/turboembed.h. Do not fork this file per language.
 *
 * Canonical path: include/turborerank.h
 * Apple copy (keep identical): swift/Sources/TurboRerankC/include/turborerank.h
 *
 * ---------------------------------------------------------------------------
 * Ownership
 * ---------------------------------------------------------------------------
 *
 * Inputs are pointer+length views. The caller owns the memory. Pointers
 * must stay valid for the duration of the call only. Strings are UTF-8
 * and are NOT required to be NUL-terminated unless a `const char *`
 * parameter is documented as a C string (config path, status names).
 *
 * Token / attention / type / position buffers come from
 * turborerank_buffer_alloc, which rents i32 rows from the shared
 * turbo_buffer arena (include/turbo_buffer.h). The caller writes tokens
 * into those pointers (or calls pack_* which writes through them). Do
 * not free individual fields. Free the buffer with turborerank_buffer_free
 * (returns the rent). Engine scratch is rented from the same ABI.
 *
 * The ABI is not thread-safe on a single engine. Serialize calls.
 * Distinct engines may be used from distinct threads.
 *
 * ---------------------------------------------------------------------------
 * Zero-copy / hot path
 * ---------------------------------------------------------------------------
 *
 * turborerank_forward MUST NOT malloc/new/grow std containers for the
 * token path. Buffers already hold [CLS] query [SEP] doc [SEP] and the
 * attention mask. Scratch for activations is reserved at load.
 *
 * GPU / Metal / AUTO requested without that accelerator → error, never
 * CPU and never a mock relevance score. MOCK is explicit ABI smoke and
 * does not score catalog cross-encoders.
 *
 * CUDA (Phase 2a): token/mask/type/position buffers are
 * cudaHostAllocMapped (PINNED mapped). The caller writes tokens in
 * those host-visible pages; kernels read the mapped device pointer
 * with no per-forward id H2D. AUTO resolves to CUDA when a device
 * is present. Create without a CUDA device fails loud. Forward runs
 * the MiniLM CE on device (first-party CUDA kernels), not a mock.
 *
 * OpenVINO (Phase 2b): token/mask/type/position buffers are Level Zero
 * USM. AUTO resolves to OPENVINO_GPU when CUDA is absent and a GPU
 * plugin is present. GPU create without a GPU fails loud. Forward
 * wraps USM pointers with ov::Tensor(..., usm_pointer) — no
 * std::vector on that path.
 *
 * Metal (Phase 2c, Machine C): token/mask/type/position buffers are
 * MTLResourceStorageModeShared. AUTO resolves to METAL when CUDA and
 * OpenVINO GPU are absent and an MTL GPU is present. Create without
 * Metal fails loud. Forward binds those MTLBuffers — no std::vector
 * and no extra token copy. Weights are copied once at load.
 */

#ifndef TURBORERANK_H
#define TURBORERANK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Bump only for breaking layout / semantic changes. */
#define TURBORERANK_ABI_VERSION 1u

/* -------------------------------------------------------------------------- */
/* Status / device                                                            */
/* -------------------------------------------------------------------------- */

typedef enum turborerank_status {
    TURBORERANK_OK = 0,
    TURBORERANK_ERR_INVALID_ARGUMENT = 1,
    TURBORERANK_ERR_NOT_FOUND = 2,
    TURBORERANK_ERR_NOT_IMPLEMENTED = 3,
    TURBORERANK_ERR_UNAVAILABLE = 4,
    TURBORERANK_ERR_INTERNAL = 5,
    TURBORERANK_ERR_OUT_OF_MEMORY = 6,
    TURBORERANK_ERR_UNSUPPORTED_DEVICE = 7
} turborerank_status;

typedef enum turborerank_device {
    TURBORERANK_DEVICE_AUTO = 0,
    TURBORERANK_DEVICE_CPU = 1,
    TURBORERANK_DEVICE_CUDA = 2,
    TURBORERANK_DEVICE_TENSORRT = 3,
    TURBORERANK_DEVICE_OPENVINO_CPU = 4,
    TURBORERANK_DEVICE_OPENVINO_GPU = 5,
    TURBORERANK_DEVICE_OPENVINO_NPU = 6,
    TURBORERANK_DEVICE_METAL = 7,
    TURBORERANK_DEVICE_MOCK = 8
} turborerank_device;

/*
 * Device policy (ABI v1 semantics, not a layout bump):
 *
 *   GPU / accelerator requests — AUTO, CUDA, TENSORRT, OPENVINO_GPU,
 *   OPENVINO_NPU, METAL — fail with UNAVAILABLE or UNSUPPORTED_DEVICE
 *   when that device is missing. They never fall back to CPU or mock.
 *   AUTO is "host default GPU", not "CPU if GPU is down".
 *
 *   CPU runs the first-party MiniLM CE kernel.
 *   CUDA / AUTO (when a CUDA device is present) run the device CE
 *   (first-party CUDA kernels) with cudaHostAllocMapped token
 *   buffers (kernels read mapped pointers; 0 token-row H2D).
 *   OPENVINO_GPU / AUTO (when CUDA is absent and an Intel GPU plugin
 *   is present) run OpenVINO CompiledModel with Level Zero USM
 *   token buffers. OPENVINO_CPU is explicit CompiledModel-on-CPU.
 *   GPU requested without GPU → UNAVAILABLE, never silent CPU.
 *   METAL / AUTO-with-Metal run the first-party Metal MiniLM CE with
 *   MTLResourceStorageModeShared token buffers.
 *   MOCK is the explicit ABI-smoke device and never returns catalog
 *   cross-encoder scores.
 */

typedef enum turborerank_truncation {
    /** HF / TEI / ST default: drop from the longer sequence. */
    TURBORERANK_TRUNC_LONGEST_FIRST = 0,
    /** only_second: keep the query; truncate the document. */
    TURBORERANK_TRUNC_QUERY_PRIORITY = 1,
    /** Over-length pairs fail with INVALID_ARGUMENT. */
    TURBORERANK_TRUNC_ERROR = 2
} turborerank_truncation;

typedef enum turborerank_activation {
    /** Product relevance: sigmoid(CLS logit). TEI default (`raw_scores=false`). */
    TURBORERANK_ACT_SIGMOID = 0,
    /** Raw CLS logit. sentence-transformers default for ms-marco-MiniLM-L6-v2. */
    TURBORERANK_ACT_IDENTITY = 1
} turborerank_activation;

/* -------------------------------------------------------------------------- */
/* Views                                                                      */
/* -------------------------------------------------------------------------- */

/** UTF-8 view. `ptr` may be NULL only when `len == 0`. */
typedef struct turborerank_str {
    const char *ptr;
    size_t len;
} turborerank_str;

/**
 * Device-backed token workspace. All four int32 arrays are
 * `[batch, seq]` row-major with `row_stride == seq`. Pointers are
 * 64-byte aligned on CPU (ggml-mappable). Caller writes tokens here.
 *
 * Do not free the inner pointers. Free with turborerank_buffer_free.
 */
typedef struct turborerank_buffer {
    int32_t *input_ids;
    int32_t *attention_mask;
    int32_t *token_type_ids;
    int32_t *position_ids;
    uint32_t batch;
    uint32_t seq;
    uint32_t row_stride;
    turborerank_device device;
} turborerank_buffer;

typedef struct turborerank_score_options {
    turborerank_truncation truncation;
    turborerank_activation activation;
    /** 0 = model default (512 for MiniLM CE). */
    uint32_t max_length;
} turborerank_score_options;

typedef struct turborerank_model_info {
    turborerank_str alias;
    uint32_t max_length;
    uint32_t hidden_size;
    turborerank_device device;
    int32_t ready;
} turborerank_model_info;

typedef struct turborerank_engine turborerank_engine;

/* -------------------------------------------------------------------------- */
/* Lifecycle                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Create an engine.
 *
 * `config_path` is a NUL-terminated filesystem path to a config, a
 * model directory, or NULL (workspace default `models/rerank/`).
 * GPU/Metal/AUTO without that accelerator → error, never CPU.
 * AUTO with CUDA present resolves to CUDA; otherwise OpenVINO GPU
 * when the plugin is present; otherwise Metal when an MTL GPU is
 * present.
 */
turborerank_status turborerank_engine_create(
    turborerank_device device,
    const char *config_path,
    turborerank_engine **out
);

void turborerank_engine_destroy(turborerank_engine *engine);

uint32_t turborerank_abi_version(void);

const char *turborerank_status_name(turborerank_status status);

const char *turborerank_device_name(turborerank_device device);

/**
 * Last error for `engine`. If `engine` is NULL, the thread-local
 * message from the most recent failed create. Never NULL.
 */
const char *turborerank_last_error(const turborerank_engine *engine);

/* -------------------------------------------------------------------------- */
/* Models                                                                     */
/* -------------------------------------------------------------------------- */

turborerank_status turborerank_list_models(
    turborerank_engine *engine,
    turborerank_model_info **out_infos,
    size_t *out_count
);

void turborerank_model_list_free(turborerank_model_info *infos, size_t count);

/**
 * Load a catalog alias (`ms-marco-minilm-l6`) or a directory containing
 * `model.safetensors` + `vocab.txt`.
 *
 * `alias_len == 0` means `alias` is a NUL-terminated C string.
 * Missing weights → UNAVAILABLE (path named). GPU aliases on a CPU
 * engine → NOT_IMPLEMENTED / UNAVAILABLE. MOCK never loads a CE alias.
 */
turborerank_status turborerank_load_model(
    turborerank_engine *engine,
    const char *alias,
    size_t alias_len
);

/* -------------------------------------------------------------------------- */
/* Buffers + packing                                                          */
/* -------------------------------------------------------------------------- */

/**
 * Allocate a device-backed token workspace. CPU: 64-byte aligned
 * posix_memalign. CUDA / AUTO-with-CUDA: cudaHostAllocMapped
 * PINNED (host write lands in device-visible pages; 64-byte).
 * OpenVINO GPU: Level Zero USM (caller writes into USM). Metal /
 * AUTO-with-Metal: MTLResourceStorageModeShared (caller writes
 * unified memory). TensorRT / NPU still UNAVAILABLE.
 */
turborerank_status turborerank_buffer_alloc(
    turborerank_device device,
    uint32_t batch,
    uint32_t seq,
    turborerank_buffer **out
);

void turborerank_buffer_free(turborerank_buffer *buffer);

/**
 * Pack already-tokenized query/doc ids (no specials) into `row`.
 * Writes [CLS] query [SEP] doc [SEP], type ids, mask, position ids.
 * Pads the remainder of the row with 0. Does not allocate.
 */
turborerank_status turborerank_pack_ids(
    turborerank_buffer *buffer,
    uint32_t row,
    const int32_t *query_ids,
    size_t n_query,
    const int32_t *doc_ids,
    size_t n_doc,
    turborerank_truncation truncation,
    uint32_t max_length
);

/**
 * WordPiece + pack one UTF-8 pair into `row`. Uses the engine vocab.
 * Tokenize scratch is engine-owned (reserved at load), not a token-path
 * std::vector handed to forward.
 */
turborerank_status turborerank_pack_text(
    turborerank_engine *engine,
    turborerank_buffer *buffer,
    uint32_t row,
    turborerank_str query,
    turborerank_str document,
    turborerank_truncation truncation,
    uint32_t max_length
);

/* -------------------------------------------------------------------------- */
/* Forward / score                                                            */
/* -------------------------------------------------------------------------- */

/**
 * Run the cross-encoder on `n_rows` packed rows (1..buffer->batch).
 * `scores_out` is caller-owned, length >= n_rows.
 *
 * No malloc on this path. Missing model / wrong device → error, never
 * a mock relevance score.
 */
turborerank_status turborerank_forward(
    turborerank_engine *engine,
    const turborerank_buffer *buffer,
    uint32_t n_rows,
    turborerank_activation activation,
    float *scores_out
);

/**
 * Convenience: tokenize + pack + forward using engine-owned buffers.
 * Scores are in document input order (RPC sorts). `opts` may be NULL
 * (sigmoid, longest-first, model max_length).
 */
turborerank_status turborerank_score(
    turborerank_engine *engine,
    const char *alias,
    size_t alias_len,
    turborerank_str query,
    const turborerank_str *documents,
    size_t n_documents,
    const turborerank_score_options *opts,
    float *scores_out
);

#ifdef __cplusplus
}
#endif

#endif /* TURBORERANK_H */
