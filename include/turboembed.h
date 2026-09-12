/* SPDX-License-Identifier: Apache-2.0
 *
 * TurboEmbed C ABI — frozen surface v1.
 *
 * One header for every host. C++ (ORT / OpenVINO GenAI) implements these
 * symbols on nvidia/intel. Apple Swift (MLX) exports the same names via
 * `@_cdecl` / a bridging header. The Rust crate `turboembed` is a safe
 * wrapper over this ABI. Do not fork this file per language.
 *
 * Canonical path: include/turboembed.h
 * Apple copy (keep identical): swift/Sources/TurboEmbedC/include/turboembed.h
 *
 * ---------------------------------------------------------------------------
 * Ownership (read this before calling anything)
 * ---------------------------------------------------------------------------
 *
 * Inputs are pointer+length *views*. The caller owns the memory. Pointers
 * must stay valid for the duration of the call only. Strings are UTF-8 and
 * are NOT required to be NUL-terminated unless a `const char *` parameter
 * is documented as a C string (config path, status names).
 *
 * Outputs are *engine-owned* until the matching free function runs:
 *   - turboembed_embed_result_free
 *   - turboembed_model_list_free
 * Views returned inside those structs are invalidated by the free, and
 * also by destroying the engine. Do not free individual fields.
 *
 * Stream callbacks: `values` is valid only for the duration of the
 * callback. Copy if you need the row later.
 *
 * The ABI is not thread-safe on a single engine. Serialize calls.
 * Distinct engines may be used from distinct threads.
 *
 * Providers (ORT, GenAI, MLX, later model2vec) register behind this ABI.
 * The stub build implements a deterministic mock plus NotImplemented for
 * real devices — inferstream servers stay as they are.
 */

#ifndef TURBOEMBED_H
#define TURBOEMBED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Bump only for breaking layout / semantic changes. */
#define TURBOEMBED_ABI_VERSION 1u

/* -------------------------------------------------------------------------- */
/* Status / device                                                            */
/* -------------------------------------------------------------------------- */

typedef enum turboembed_status {
    TURBOEMBED_OK = 0,
    TURBOEMBED_ERR_INVALID_ARGUMENT = 1,
    TURBOEMBED_ERR_NOT_FOUND = 2,
    TURBOEMBED_ERR_NOT_IMPLEMENTED = 3,
    TURBOEMBED_ERR_UNAVAILABLE = 4,
    TURBOEMBED_ERR_INTERNAL = 5,
    TURBOEMBED_ERR_OUT_OF_MEMORY = 6,
    TURBOEMBED_ERR_UNSUPPORTED_DEVICE = 7
} turboembed_status;

typedef enum turboembed_device {
    TURBOEMBED_DEVICE_AUTO = 0,
    TURBOEMBED_DEVICE_CPU = 1,
    TURBOEMBED_DEVICE_CUDA = 2,
    TURBOEMBED_DEVICE_TENSORRT = 3,
    TURBOEMBED_DEVICE_OPENVINO_CPU = 4,
    TURBOEMBED_DEVICE_OPENVINO_GPU = 5,
    TURBOEMBED_DEVICE_OPENVINO_NPU = 6,
    TURBOEMBED_DEVICE_METAL = 7,
    TURBOEMBED_DEVICE_MOCK = 8
} turboembed_device;

typedef enum turboembed_pooling {
    TURBOEMBED_POOLING_DEFAULT = 0,
    TURBOEMBED_POOLING_MEAN = 1,
    TURBOEMBED_POOLING_CLS = 2,
    TURBOEMBED_POOLING_LAST = 3
} turboembed_pooling;

/** How embed results are presented. Both views may alias the same buffer. */
typedef enum turboembed_output_format {
    TURBOEMBED_OUTPUT_TYPED = 0,
    TURBOEMBED_OUTPUT_PACKED_BYTES = 1
} turboembed_output_format;

/* -------------------------------------------------------------------------- */
/* Views (caller-owned inputs)                                                */
/* -------------------------------------------------------------------------- */

/** UTF-8 view. `ptr` may be NULL only when `len == 0`. */
typedef struct turboembed_str {
    const char *ptr;
    size_t len;
} turboembed_str;

typedef struct turboembed_embed_options {
    turboembed_pooling pooling;
    /** -1 = provider default, 0 = false, 1 = true */
    int32_t normalize;
    /** 0 = provider / model default */
    uint32_t truncate_to;
    turboembed_output_format output_format;
} turboembed_embed_options;

typedef struct turboembed_model_info {
    /** Alias view; owned by the list until turboembed_model_list_free. */
    turboembed_str alias;
    uint32_t dim;
    turboembed_device device;
    /** 1 = loaded and ready, 0 = catalogued but not loaded. */
    int32_t ready;
} turboembed_model_info;

/**
 * Batch embed result. Row-major FP32, `count` rows of `dim` columns.
 *
 * `values` is always populated on OK (typed view).
 * `packed` is the same bytes as little-endian FP32 (`packed_len ==
 * count * dim * sizeof(float)`). Callers that asked for packed bytes can
 * treat `packed` as the wire blob; others can ignore it.
 *
 * Free with turboembed_embed_result_free. Do not free `values` / `packed`.
 */
typedef struct turboembed_embed_result {
    uint32_t dim;
    uint32_t count;
    const float *values;
    const uint8_t *packed;
    size_t packed_len;
} turboembed_embed_result;

typedef struct turboembed_engine turboembed_engine;

/**
 * Optional per-row stream callback.
 *
 * `values` / `dim` describe one row. Valid only until the callback returns.
 * `is_final` is 1 on the last row of the batch.
 */
typedef void (*turboembed_stream_cb)(
    void *user_data,
    uint32_t index,
    const float *values,
    uint32_t dim,
    int32_t is_final
);

/**
 * Provider vtable sketch (ORT / GenAI / MLX / later model2vec).
 * Registration is reserved; the stub returns NOT_IMPLEMENTED.
 * Layout is part of ABI v1 so later plugins can link without a header bump.
 */
typedef struct turboembed_provider_vtbl {
    const char *id;
    turboembed_status (*load)(
        void *ctx,
        const char *alias,
        size_t alias_len
    );
    turboembed_status (*embed)(
        void *ctx,
        const turboembed_str *texts,
        size_t n_texts,
        const turboembed_embed_options *opts,
        turboembed_embed_result **out
    );
    void *ctx;
} turboembed_provider_vtbl;

/* -------------------------------------------------------------------------- */
/* Lifecycle                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Create an engine.
 *
 * `config_path` is a NUL-terminated filesystem path, or NULL.
 * On OK, `*out` is non-NULL and the caller must destroy it.
 * On error, `*out` is NULL; turboembed_last_error(NULL) may explain.
 */
turboembed_status turboembed_engine_create(
    turboembed_device device,
    const char *config_path,
    turboembed_engine **out
);

/** Destroy an engine. NULL is a no-op. Invalidates outstanding results. */
void turboembed_engine_destroy(turboembed_engine *engine);

uint32_t turboembed_abi_version(void);

const char *turboembed_status_name(turboembed_status status);

const char *turboembed_device_name(turboembed_device device);

/**
 * Last error message for `engine`. If `engine` is NULL, the thread-local
 * message from the most recent failed create. Pointer valid until the next
 * mutating call on that engine (or create, if engine is NULL). Never NULL.
 */
const char *turboembed_last_error(const turboembed_engine *engine);

/* -------------------------------------------------------------------------- */
/* Models                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * List catalogued aliases known to this engine.
 * Free with turboembed_model_list_free(*out_infos, *out_count).
 */
turboembed_status turboembed_list_models(
    turboembed_engine *engine,
    turboembed_model_info **out_infos,
    size_t *out_count
);

void turboembed_model_list_free(turboembed_model_info *infos, size_t count);

/**
 * Load a catalog alias (e.g. "minilm", "bge-small").
 *
 * `alias_len == 0` means `alias` is a NUL-terminated C string.
 * Stub: "mock-embed" / "mock" succeed; real engine aliases return
 * NOT_IMPLEMENTED until the ORT / GenAI / MLX providers are wired.
 */
turboembed_status turboembed_load_model(
    turboembed_engine *engine,
    const char *alias,
    size_t alias_len
);

/* -------------------------------------------------------------------------- */
/* Embed                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * Embed one UTF-8 text. Convenience for n=1; same ownership as batch.
 */
turboembed_status turboembed_embed_one(
    turboembed_engine *engine,
    const char *alias,
    size_t alias_len,
    const char *text,
    size_t text_len,
    const turboembed_embed_options *opts,
    turboembed_embed_result **out
);

/**
 * Embed a batch. `texts` is an array of `n_texts` views. `opts` may be NULL
 * (provider defaults). `*out` must be released with
 * turboembed_embed_result_free.
 */
turboembed_status turboembed_embed(
    turboembed_engine *engine,
    const char *alias,
    size_t alias_len,
    const turboembed_str *texts,
    size_t n_texts,
    const turboembed_embed_options *opts,
    turboembed_embed_result **out
);

/**
 * Optional stream: one callback per row, then OK.
 * Stub implements this by embedding the batch then invoking `cb`.
 * `cb` may be NULL (then this is identical to turboembed_embed and `out`
 * is still populated).
 */
turboembed_status turboembed_embed_stream(
    turboembed_engine *engine,
    const char *alias,
    size_t alias_len,
    const turboembed_str *texts,
    size_t n_texts,
    const turboembed_embed_options *opts,
    turboembed_stream_cb cb,
    void *user_data,
    turboembed_embed_result **out
);

void turboembed_embed_result_free(turboembed_embed_result *result);

/**
 * Generic buffer release for any pointer the engine allocated that is not
 * a typed result/list. NULL is a no-op. Prefer the typed free functions.
 */
void turboembed_buffer_free(void *ptr);

/**
 * Register a provider vtable. Stub: NOT_IMPLEMENTED.
 * Future: ORT / GenAI / MLX / model2vec plug in here.
 */
turboembed_status turboembed_register_provider(
    const turboembed_provider_vtbl *vtbl
);

#ifdef __cplusplus
}
#endif

#endif /* TURBOEMBED_H */
