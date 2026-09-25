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
 *     a backend fills. The core knows seven sizes: through capability,
 *     through buffer_export, through model_release, through session_run,
 *     through buffer_read, through reserved2, and through
 *     session_create_tuned, which is sizeof(turbo_backend). It refuses
 *     any other. A function added after
 *     the first three may be NULL, and the core calls it only when
 *     struct_size covers it; formats it reads only when struct_size
 *     covers it, and a table that ends before it loads FORMAT_SAFETENSORS
 *     alone.
 *   - A backend that offers context_create offers context_release, one
 *     that offers buffer_alloc or buffer_import offers buffer_release, one
 *     that offers model_load offers model_release, and one that offers
 *     session_create or session_create_tuned offers session_release and
 *     session_run. The core refuses a table that does not.
 *   - The core checks every struct's struct_size, every enumeration and
 *     every shape before it calls, and fills in turbo_buffer_desc.bytes.
 *     It releases every buffer and model before the context it was made
 *     on, every session before the model it was made on, and each handle
 *     once. Release functions return nothing, as turbo.h's do.
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

/* An artifact's format, as docs/bundle.md's artifacts[].format names it.
 * turbo_backend.formats has bit TURBO_FORMAT_BIT(f) set for each format f
 * the backend's model_load takes. The core hands a backend
 * FORMAT_SAFETENSORS and FORMAT_HEF; it never chooses an artifact of the
 * other formats for any backend. */
#define TURBO_FORMAT_SAFETENSORS 1   /* raw weights: tensors, no artifact bytes */
#define TURBO_FORMAT_OPENVINO_IR 2
#define TURBO_FORMAT_HEF         3   /* a Hailo compiled graph: one file, as artifact */
#define TURBO_FORMAT_GGUF        4
#define TURBO_FORMAT_ONNX        5
#define TURBO_FORMAT_BIT(f) (1u << ((f) - 1u))

/* Where an artifact's graph starts and stops (docs/bundle.md,
 * artifacts[].graph_input and graph_output). */
#define TURBO_INPUT_TOKEN_IDS       1   /* ids, mask and types as the rows give them */
#define TURBO_INPUT_EMBEDDINGS      2   /* the word-embedding rows gathered by id, before the
                                           position and token type embeddings are added and
                                           before the embeddings LayerNorm; the graph owns
                                           everything after, and computes token type 0 only */
#define TURBO_OUTPUT_HIDDEN_STATES  1   /* the last layer's hidden states, [batch, seq, hidden];
                                           pooling and normalize are the backend's to add */

/* What a kernel computes in: the classes a precision may run, as bits. A
 * backend's kernel variant computes in exactly one; the core allows a
 * session's precision a set (turbo_backend_tuning.numerics_allowed), and
 * a variant outside the set is never launched, timed or forced. */
#define TURBO_NUMERIC_F32_FMA        1   /* F32 operands, F32 FMAs */
#define TURBO_NUMERIC_TF32           2   /* F32 operands rounded to TF32 on tensor cores, F32 sums */
#define TURBO_NUMERIC_F16_F32ACC     4   /* F16 operands, F32 sums */
#define TURBO_NUMERIC_F16_CHUNKACC   8   /* F16 operands, F16 sums within a k chunk, F32 across chunks */

/* The core's tuning policy and cache for a session, and what the backend
 * chose: passed to session_create_tuned, in and out. Grows at its end
 * under struct_size. */
typedef struct turbo_backend_tuning {
    uint32_t    struct_size;
    /* In. */
    uint32_t    mode;              /* TURBO_AUTOTUNE_OFF, ON or RETUNE; never RUNTIME */
    uint32_t    budget_ms;         /* the time the backend may spend measuring; 0 when OFF */
    uint32_t    numerics_allowed;  /* TURBO_NUMERIC_* bits the precision allows: the core's
                                      table, which only a decision widens */
    const char *cached;            /* the choices string an earlier session reported for this
                                      key, NUL-terminated, or NULL: the incumbent every
                                      candidate must beat by the backend's margin */
    /* Out: filled by the backend before it returns TURBO_OK. */
    uint32_t    tuned;             /* TURBO_TUNED_* */
    uint32_t    tune_ms;
    char        choices[TURBO_CHOICES_LEN];   /* the session's choices, empty for one path */
    /* Out, may be NULL: the timings the choice was made on, for the cache
     * and the record, as "<bin>/<knob>/<variant>=<min ms>" lines separated
     * by "\n", at most timings_len bytes with the NUL; the backend writes
     * what fits and never fails for want of room. */
    char       *timings;
    uint32_t    timings_len;
    uint32_t    numerics_used;     /* numerics_allowed as the backend's own experiment variables
                                      widened it, for the record and the log. A session where it
                                      differs from numerics_allowed is never cached */
} turbo_backend_tuning;

/* One tensor, in the bytes the core read from the weights file and checked
 * against the manifest's hash. Packed row-major, little-endian. */
typedef struct turbo_backend_tensor {
    const char *name;       /* its name in the weights file, for messages */
    const void *data;       /* aligned to at least the dtype's element size */
    uint64_t    shape[2];   /* entries past ndim are 0 */
    uint32_t    ndim;       /* 1 or 2 */
    uint32_t    dtype;      /* TURBO_DTYPE_*: the model's dtype */
    uint64_t    bytes;
} turbo_backend_tensor;

/* The artifact rule 6 of docs/bundle.md chose, as model_load receives it.
 * layers through layer_norm_eps are the manifest's architecture, whatever
 * the format.
 *
 *   FORMAT_SAFETENSORS  graph_input TOKEN_IDS; tensors holds every tensor
 *                       of the encoder; artifact NULL, artifact_bytes 0.
 *   FORMAT_HEF          artifact is the one file's bytes, as hashed. With
 *                       graph_input EMBEDDINGS, tensors holds the
 *                       host_weights artifact's embedding tensors,
 *                       tensor_count TURBO_BERT_EMBEDDING_TENSORS in the
 *                       usual order, for the lookup the backend does on
 *                       the host; with TOKEN_IDS, tensors is NULL and
 *                       tensor_count 0.
 *
 * artifact, like each tensor's data, stays where it is, unchanged, until
 * model_release returns. */
typedef struct turbo_backend_model {
    uint32_t    struct_size;
    uint32_t    family;           /* TURBO_FAMILY_*, which says how tensors is laid out */
    uint32_t    dtype;            /* every tensor's: TURBO_DTYPE_F32, F16 or BF16; 0 when
                                     tensor_count is 0 */
    uint32_t    layers;
    uint32_t    hidden;
    uint32_t    heads;
    uint32_t    intermediate;
    uint32_t    vocab_size;
    uint32_t    max_positions;
    uint32_t    token_types;
    double      layer_norm_eps;
    uint32_t    tensor_count;     /* BERT: TURBO_BERT_EMBEDDING_TENSORS + layers * TURBO_BERT_LAYER_TENSORS
                                     for raw weights; as above for a HEF */
    uint32_t    reserved;
    const turbo_backend_tensor *tensors;
    uint32_t    format;           /* TURBO_FORMAT_*, one turbo_backend.formats lists */
    uint32_t    graph_input;      /* TURBO_INPUT_* */
    uint32_t    graph_output;     /* TURBO_OUTPUT_* */
    uint32_t    compute_dtype;    /* TURBO_DTYPE_* the compilation fixed; 0 where the manifest
                                     fixes none, as for raw weights */
    uint32_t    fixed_seq;        /* the shape compiled in; 0 is dynamic */
    uint32_t    fixed_batch;      /* rows a frame holds; 0 is dynamic. Not a session limit:
                                     the backend runs as many frames as a batch needs */
    const void *artifact;         /* the compiled artifact's bytes; NULL for raw weights */
    uint64_t    artifact_bytes;
} turbo_backend_model;

/* Rows for an embed run, as the core hands them to embed_write, [batch,
 * seq] int32 with row_stride elements between row starts. The core has
 * checked every value: each id is below vocab_size, each type below
 * token_types, each mask entry 0 or 1 with at least one 1 per row, batch
 * and seq within the session's, and the options resolved from
 * turbo_embed_options against the bundle, so none of them is a MODEL
 * value. A backend on a TURBO_INPUT_EMBEDDINGS artifact refuses rows with
 * a type other than 0 with TURBO_E_UNSUPPORTED_OPTION (docs/bundle.md,
 * "Graph inputs"). */
typedef struct turbo_backend_embed_rows {
    uint32_t       struct_size;
    uint32_t       batch;
    uint32_t       seq;
    uint32_t       row_stride;   /* at least seq */
    const int32_t *ids;
    const int32_t *mask;
    const int32_t *types;        /* NULL for all zero */
    uint32_t       pooling;      /* TURBO_POOLING_MEAN, CLS or LAST */
    uint32_t       normalize;    /* TURBO_NORMALIZE_NONE or L2 */
    uint32_t       output_dim;   /* the first output_dim values of each vector, 1 to hidden */
    uint32_t       reserved;
} turbo_backend_embed_rows;

/* What a run produced and what it cost: turbo_result_info's fields that
 * only the backend knows. */
typedef struct turbo_backend_run {
    uint32_t struct_size;
    uint32_t placement;       /* TURBO_PLACE_* of output */
    void    *output;          /* the session's buffer, as buffer_alloc gives one, holding [batch,
                                 output_dim] F32 packed; handed to buffer_export, never to
                                 buffer_release. It holds the vectors until the next embed_write,
                                 and the core writes nothing to the session while a result is held */
    void    *host;            /* output's host address; NULL for TURBO_PLACE_DEVICE */
    uint64_t h2d_bytes;       /* bytes sent to the device for this run, by embed_write too */
    uint64_t d2h_bytes;       /* bytes brought back by the run */
    uint64_t host_allocs;     /* heap allocations the backend made in session_run */
    uint64_t device_allocs;   /* device allocations it made in session_run */
    uint32_t stage[TURBO_STAGE_MAX];   /* TURBO_STAGE_* by the task's stage index */
} turbo_backend_run;

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
     * as in turbo_capability, both read only when status is EXPERIMENTAL.
     * The core honors the options it applies before the backend sees the
     * rows (for embed: truncate, max_tokens and prompt_role) and sets
     * their bits itself; the backend sets the bits of those its run takes;
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

    /* Sessions. NULL where the backend runs no task. */

    /* A session of task on a model: every byte it needs to run max_batch
     * rows of max_seq tokens, allocated here, so that a run allocates
     * nothing. The core has resolved 0 in turbo_session_desc to the
     * model's values, and checked they are within the model's. precision
     * is TURBO_PRECISION_*; *compute_dtype receives the TURBO_DTYPE_* it
     * resolves to on this model. Weights converted to another dtype are
     * the model's, as turbo.h says, made here on first need and shared. A
     * precision the backend cannot compute this model in is
     * TURBO_E_UNSUPPORTED_OPTION with turbo_error.field 3. *out is the
     * backend's own, handed back to the functions below. */
    int32_t (*session_create)(void *model, uint32_t task, uint32_t max_batch, uint32_t max_seq,
                              uint32_t precision, uint32_t *compute_dtype, void **out, turbo_error *err);
    void    (*session_release)(void *session);
    /* The rows and options of the next run, which replace any written
     * before. rows and the arrays it points to are valid for the call: a
     * backend keeps what it needs, sending it to its device if it has one.
     * NULL where the backend does not embed. */
    int32_t (*embed_write)(void *session, const turbo_backend_embed_rows *rows, turbo_error *err);
    /* Run what was last written; the core calls it once per write. out's
     * struct_size is set and stage[TURBO_EMBED_STAGE_TOKENIZE] is filled
     * in; the backend fills in the rest. */
    int32_t (*session_run)(void *session, turbo_backend_run *out, turbo_error *err);

    /* Reads. NULL where every buffer the backend gives has a host address. */

    /* Copy the first bytes of a buffer with no host address (a
     * TURBO_PLACE_DEVICE one: a session's output, say) to dst, host memory
     * the caller owns, and return once they are there. The core calls it
     * for turbo_result_read, after the run that wrote the buffer returned,
     * and counts the bytes in d2h_bytes. */
    int32_t (*buffer_read)(void *buf, void *dst, uint64_t bytes, turbo_error *err);

    /* Formats. */

    /* TURBO_FORMAT_BIT of each TURBO_FORMAT_* model_load takes. 0 is
     * FORMAT_SAFETENSORS alone, as for a table that ends before this. Rule
     * 6 of docs/bundle.md chooses only an artifact whose format is here. */
    uint32_t formats;
    uint32_t reserved2;

    /* Tuned sessions. NULL where the backend has one path. */

    /* session_create with the core's tuning: a backend that offers it is
     * called through it, and session_create is then never called. With
     * tuning->mode ON or RETUNE the backend may spend about budget_ms
     * timing its kernel variants for this session, using the context's
     * log function, and takes the fastest whose numeric class is in
     * numerics_allowed; OFF, it takes its built-in choices. Its own
     * environment variables force choices in every mode, and a forced
     * variant outside numerics_allowed is TURBO_E_UNSUPPORTED_OPTION with
     * turbo_error.field 3. It fills the out fields whatever the mode.
     * tuning is valid for the call. */
    int32_t (*session_create_tuned)(void *model, uint32_t task, uint32_t max_batch, uint32_t max_seq,
                                    uint32_t precision, turbo_backend_tuning *tuning,
                                    uint32_t *compute_dtype, void **out, turbo_error *err);
} turbo_backend;

#ifdef __cplusplus
}
#endif

#endif /* TURBO_BACKEND_H */
