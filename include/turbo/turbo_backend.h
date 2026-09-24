/* SPDX-License-Identifier: Apache-2.0
 *
 * The boundary between libturbo's core and a backend. Not part of the
 * public interface: callers use turbo.h. A backend is written in whatever
 * language suits its vendor's API (C++ for CUDA, Objective-C++ for Metal,
 * Rust for the CPU) and is linked into libturbo at build time; the core
 * reaches it only through this table.
 *
 * Rules
 *   - Every function follows turbo.h's conventions: it returns a status,
 *     fills the caller's turbo_error when it is not NULL, and never panics
 *     or throws across the boundary.
 *   - A backend lists only devices that are present and answered a probe.
 *     A driver that is missing lists nothing; it is not an error.
 *   - A backend says what it has built for a (task, precision) on a device:
 *     TURBO_CAP_EXPERIMENTAL or TURBO_CAP_UNSUPPORTED, with a reason. Only
 *     the core says SUPPORTED, and only when a benchmark record for that
 *     cell exists; a backend cannot claim it.
 *   - Every function in the table may be called from any thread, at the
 *     same time as any other. A backend that needs a lock takes its own.
 *   - The core checks ordinal against device_count, and task and precision
 *     against the TURBO_* values, before it calls. A backend need not, and
 *     answers only for the ordinals it listed.
 *   - A backend keeps no per-runtime state and has no init or shutdown.
 *     What it owns hangs off the handles later functions return, a context
 *     first. It has no log function of its own: its warnings reach the
 *     caller through the log function the core hands it with a context.
 *   - The table grows only at the end, and struct_size says how much of it
 *     a backend fills. The core refuses a table whose struct_size is not
 *     one it knows. A function added after the first three may be NULL,
 *     and the core calls it only when struct_size covers it.
 *   - A backend that offers context_create offers context_release, one
 *     that offers buffer_alloc or buffer_import offers buffer_release, and
 *     one that offers model_load offers model_release. The core refuses a
 *     table that does not.
 *   - The core checks every struct's struct_size, every enumeration and
 *     every shape before it calls, and fills in turbo_buffer_desc.bytes.
 *     It releases every buffer and model before the context it was made
 *     on, and each handle once. Release functions return nothing, as
 *     turbo.h's do.
 */

#ifndef TURBO_BACKEND_H
#define TURBO_BACKEND_H

#include <turbo/turbo.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A model's weights, as model_load receives them. */

/* Where each tensor of a BERT encoder sits in turbo_backend_model.tensors:
 * the embedding tensors first, then each layer's, layer 0 first. Layer l's
 * tensor r is at TURBO_BERT_EMBEDDING_TENSORS + l * TURBO_BERT_LAYER_TENSORS + r.
 * Linear weights are [out, in]; biases and LayerNorm tensors are [hidden],
 * or [intermediate] for ffn_in's bias. */
#define TURBO_BERT_WORD_EMBEDDINGS        0   /* [vocab_size, hidden] */
#define TURBO_BERT_POSITION_EMBEDDINGS    1   /* [max_positions, hidden] */
#define TURBO_BERT_TOKEN_TYPE_EMBEDDINGS  2   /* [token_types, hidden] */
#define TURBO_BERT_EMBEDDINGS_LN_WEIGHT   3
#define TURBO_BERT_EMBEDDINGS_LN_BIAS     4
#define TURBO_BERT_EMBEDDING_TENSORS      5

#define TURBO_BERT_Q_WEIGHT        0   /* [hidden, hidden] */
#define TURBO_BERT_Q_BIAS          1
#define TURBO_BERT_K_WEIGHT        2   /* [hidden, hidden] */
#define TURBO_BERT_K_BIAS          3
#define TURBO_BERT_V_WEIGHT        4   /* [hidden, hidden] */
#define TURBO_BERT_V_BIAS          5
#define TURBO_BERT_ATTN_OUT_WEIGHT 6   /* [hidden, hidden] */
#define TURBO_BERT_ATTN_OUT_BIAS   7
#define TURBO_BERT_ATTN_LN_WEIGHT  8
#define TURBO_BERT_ATTN_LN_BIAS    9
#define TURBO_BERT_FFN_IN_WEIGHT  10   /* [intermediate, hidden] */
#define TURBO_BERT_FFN_IN_BIAS    11   /* [intermediate] */
#define TURBO_BERT_FFN_OUT_WEIGHT 12   /* [hidden, intermediate] */
#define TURBO_BERT_FFN_OUT_BIAS   13
#define TURBO_BERT_FFN_LN_WEIGHT  14
#define TURBO_BERT_FFN_LN_BIAS    15
#define TURBO_BERT_LAYER_TENSORS  16

/* The encoder families a model may be. */
#define TURBO_FAMILY_BERT 1   /* GELU (erf), absolute positions, post-LayerNorm */

/* One tensor, in the bytes the core read from the weights file and checked
 * against the manifest's hash. Packed row-major, little-endian. */
typedef struct turbo_backend_tensor {
    const char *name;       /* its name in the weights file, for messages */
    const void *data;
    uint64_t    shape[2];   /* entries past ndim are 0 */
    uint32_t    ndim;       /* 1 or 2 */
    uint32_t    dtype;      /* TURBO_DTYPE_*: the model's dtype */
    uint64_t    bytes;
} turbo_backend_tensor;

typedef struct turbo_backend_model {
    uint32_t    struct_size;
    uint32_t    family;           /* TURBO_FAMILY_*, which says how tensors is laid out */
    uint32_t    dtype;            /* every tensor's: TURBO_DTYPE_F32, F16 or BF16 */
    uint32_t    layers;
    uint32_t    hidden;
    uint32_t    heads;
    uint32_t    intermediate;
    uint32_t    vocab_size;
    uint32_t    max_positions;
    uint32_t    token_types;
    double      layer_norm_eps;
    uint32_t    tensor_count;     /* BERT: TURBO_BERT_EMBEDDING_TENSORS + layers * TURBO_BERT_LAYER_TENSORS */
    uint32_t    reserved;
    const turbo_backend_tensor *tensors;
} turbo_backend_model;

typedef struct turbo_backend {
    uint32_t    struct_size;
    uint32_t    reserved;
    const char *name;              /* turbo_device_info.backend and the manifest's backends[] */
    const char *runtime_version;   /* the vendor runtime this build linked; "" where there is none */

    int32_t (*device_count)(uint32_t *out, turbo_error *err);
    /* ordinal < device_count. Fills every field but struct_size and
     * backend, which the core writes from name. Called again on every
     * turbo_runtime_device_info, so memory_free is current. */
    int32_t (*device_info)(uint32_t ordinal, turbo_device_info *out, turbo_error *err);
    /* What this backend has built for (task, precision) on the device.
     * status is TURBO_CAP_EXPERIMENTAL or TURBO_CAP_UNSUPPORTED; dtype is
     * the compute dtype the precision resolves to and options_honored is
     * as in turbo_capability, both read only when status is EXPERIMENTAL;
     * reason (NUL-terminated, reason_len bytes) says why it is not
     * EXPERIMENTAL, else is empty. */
    int32_t (*capability)(uint32_t ordinal, uint32_t task, uint32_t precision,
                          uint32_t *status, uint32_t *dtype, uint32_t *options_honored,
                          char *reason, uint32_t reason_len, turbo_error *err);

    /* Contexts and buffers. NULL where the backend has none. */

    /* A context on the device. log (which may be NULL) and log_user_data
     * are the runtime's, for the warnings of this context and its buffers.
     * *out is the backend's own, handed back to buffer_alloc and
     * buffer_import. */
    int32_t (*context_create)(uint32_t ordinal, turbo_log_fn log, void *log_user_data,
                              void **out, turbo_error *err);
    void    (*context_release)(void *ctx);
    /* host receives the memory's host address for HOST, PINNED and SHARED
     * placements, and NULL for DEVICE. *out is the backend's own, handed
     * back to buffer_export and buffer_release. */
    int32_t (*buffer_alloc)(void *ctx, const turbo_buffer_desc *desc,
                            void **out, void **host, turbo_error *err);
    /* As buffer_alloc, over memory the caller owns: nothing is copied and
     * nothing of the caller's is freed on release. */
    int32_t (*buffer_import)(void *ctx, const turbo_buffer_desc *desc, const turbo_native_handle *handle,
                             void **out, void **host, turbo_error *err);
    void    (*buffer_release)(void *buf);
    /* Fills every field of out but struct_size. */
    int32_t (*buffer_export)(void *buf, uint32_t kind, turbo_native_handle *out, turbo_error *err);

    /* Models. NULL where the backend has none. */

    /* A model's weights on the context's device. The core has checked every
     * tensor's name, dtype and shape against the manifest. desc and its
     * tensors array are valid for the call; the bytes each tensor's data
     * points to stay where they are, unchanged, until model_release
     * returns, so a backend on host memory reads them in place rather than
     * copying them. A backend whose device has its own memory copies them
     * there: that copy is the load. A vendor failure is TURBO_E_RUNTIME with
     * the vendor's text. *out is the backend's own, handed back to
     * model_release. */
    int32_t (*model_load)(void *ctx, const turbo_backend_model *desc, void **out, turbo_error *err);
    void    (*model_release)(void *model);
} turbo_backend;

#ifdef __cplusplus
}
#endif

#endif /* TURBO_BACKEND_H */
