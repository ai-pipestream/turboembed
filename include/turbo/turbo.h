/* SPDX-License-Identifier: Apache-2.0
 *
 * TurboEmbed C interface. Hand-written. This file is the design; the code
 * follows it.
 *
 * First cut: text embedding, tokens in and vectors out, on one device. The
 * other tasks and chunking are added here when they are built, not before,
 * by the rule under "Tasks" below.
 *
 * Conventions
 *   - Every struct the caller fills or receives starts with
 *     uint32_t struct_size = sizeof(struct). The library reads and writes
 *     only fields below that size. A size it does not know is
 *     TURBO_E_INVALID_STRUCT_SIZE.
 *   - Enumerations are uint32_t constants. An unknown value is
 *     TURBO_E_INVALID_ENUM. Nothing is mapped to a default.
 *   - Strings are UTF-8 views (turbo_text), valid for the call. Bad UTF-8 is
 *     TURBO_E_INVALID_UTF8.
 *   - Every function returns a status code. Details go to the caller-owned
 *     turbo_error, which may be NULL. There is no global error state.
 *   - Handles are opaque and reference counted. A child keeps its parent
 *     alive. Release functions accept NULL.
 *   - Runtime, context, model and tokenizer may be used from any thread.
 *     A session and its results have one owner at a time; a second call on
 *     a busy session returns TURBO_E_BUSY.
 *   - An option the backend cannot honor for this model on this device
 *     fails the call with TURBO_E_UNSUPPORTED_OPTION and turbo_error.field
 *     naming the field (1-based). Nothing is ignored, clamped or replaced.
 *   - There is no fake device. Every device listed is real hardware.
 *
 * Tasks
 *   - A bundle has one task. turbo_model_load reads it from the manifest and
 *     turbo_model_info.task reports it. A session and its results belong to
 *     that task. Calling another task's function on it is
 *     TURBO_E_UNSUPPORTED_TASK.
 *   - A task is added by a change to this file alone, made before any of its
 *     code. It adds exactly these, named after the task, and changes nothing
 *     an earlier task uses:
 *
 *       what                 name                        rule
 *       task number          TURBO_TASK_<TASK>           next free number, never reused
 *       options              turbo_<task>_options        struct_size first, then fields
 *                                                        numbered from 1; 0 in every field
 *                                                        is what the bundle says
 *       write                turbo_<task>_write_<input>  one per input shape, each taking
 *                                                        the session and the task's options
 *       read                 turbo_<task>_read_<what>    only when the output is not one
 *                                                        [batch, dim] block, which
 *                                                        turbo_result_read and
 *                                                        turbo_result_buffer already give
 *       stages               TURBO_<TASK>_STAGE_*        in pipeline order from 0, and
 *                                                        TURBO_<TASK>_STAGE_COUNT, at most
 *                                                        TURBO_STAGE_MAX
 *       manifest block       "<task>" in docs/bundle.md  the model facts the task needs
 *       model facts          fields at the end of        0 or empty for a model of
 *                            turbo_model_info            another task; that 0 is not an
 *                                                        option value
 *       tokenizer            turbo_tokenizer_encode_     only when the input is not one
 *                            <input>                     text per row (pairs, say); the
 *                                                        manifest template gets the
 *                                                        matching form
 *
 *   - turbo_session_run, turbo_result_get_info, turbo_result_read and
 *     turbo_result_buffer serve every task. turbo_result_info.stage_count
 *     says how many entries of stage[] the task uses; the rest are
 *     TURBO_STAGE_UNUSED.
 *   - turbo_capability.options_honored and turbo_error.field count the
 *     fields of the task's own options struct.
 *   - A task whose output is not a finished block per run (generation, with
 *     state kept across calls) gets its own handle, designed with it.
 */

#ifndef TURBO_H
#define TURBO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Handles ----------------------------------------------------------- */

typedef struct turbo_runtime   turbo_runtime;   /* the library, its devices */
typedef struct turbo_context   turbo_context;   /* one device, its memory   */
typedef struct turbo_buffer    turbo_buffer;    /* memory on host or device */
typedef struct turbo_model     turbo_model;     /* a bundle loaded on a context */
typedef struct turbo_session   turbo_session;   /* inputs, one run at a time */
typedef struct turbo_result    turbo_result;    /* outputs of one run       */
typedef struct turbo_tokenizer turbo_tokenizer; /* the bundle's tokenizer   */

/* ---- Status codes ------------------------------------------------------ */

#define TURBO_OK                      0

#define TURBO_E_INVALID_ARGUMENT      256  /* turbo_error.field may name it */
#define TURBO_E_INVALID_STRUCT_SIZE   257
#define TURBO_E_INVALID_UTF8          258
#define TURBO_E_INVALID_HANDLE        259
#define TURBO_E_INVALID_SHAPE         260
#define TURBO_E_INVALID_STATE         261  /* e.g. run before write */
#define TURBO_E_INVALID_ENUM          262

#define TURBO_E_UNSUPPORTED           512  /* this device cannot; message says what */
#define TURBO_E_UNSUPPORTED_OPTION    513  /* turbo_error.field names the option */
#define TURBO_E_UNSUPPORTED_TASK      514  /* the model or device does not offer it */

#define TURBO_E_OUT_OF_MEMORY         768
#define TURBO_E_BUSY                  769
#define TURBO_E_CAPACITY              771  /* over the session's batch or sequence */

#define TURBO_E_DEVICE_NOT_FOUND     1024
#define TURBO_E_DEVICE_UNAVAILABLE   1025  /* present, but driver or probe failed */
#define TURBO_E_RUNTIME              1026  /* the vendor runtime failed; message has its text */

#define TURBO_E_BUNDLE_NOT_FOUND     1280
#define TURBO_E_BUNDLE_INVALID       1281
#define TURBO_E_BUNDLE_INTEGRITY     1282  /* a file does not match its hash */
#define TURBO_E_BUNDLE_NO_ARTIFACT   1283  /* nothing in it this device can load */

#define TURBO_E_INTERNAL             1536  /* a bug in the library */
#define TURBO_E_PANIC                1537  /* caught at the language boundary */

/* ---- Constants --------------------------------------------------------- */

#define TURBO_DEVICE_CPU   1
#define TURBO_DEVICE_GPU   2   /* discrete */
#define TURBO_DEVICE_IGPU  3   /* integrated, shares memory with the host */
#define TURBO_DEVICE_NPU   4

#define TURBO_TASK_EMBED   1

/* What a (device, task) cell may say. There is no "planned": a thing that
 * does not run is unsupported. */
#define TURBO_CAP_UNSUPPORTED  0
#define TURBO_CAP_EXPERIMENTAL 1   /* runs; not yet measured against the vendor's best */
#define TURBO_CAP_SUPPORTED    2   /* runs, matches the reference numerically, and the
                                      benchmark record named in the cell exists; the rule
                                      is in docs/benchmarks.md */

#define TURBO_DTYPE_I32   8
#define TURBO_DTYPE_F16  10
#define TURBO_DTYPE_BF16 11
#define TURBO_DTYPE_F32  12

#define TURBO_PLACE_HOST    1   /* pageable host memory */
#define TURBO_PLACE_PINNED  2   /* page-locked host memory the device can read */
#define TURBO_PLACE_DEVICE  3   /* device memory */
#define TURBO_PLACE_SHARED  4   /* one allocation both can address */
/* A CPU has no memory apart from the host's: on a CPU device HOST, PINNED
 * and SHARED are all host memory, and DEVICE is TURBO_E_UNSUPPORTED. */

/* Entries in turbo_result_info.stage; no task has more stages. */
#define TURBO_STAGE_MAX 16

/* Pipeline stages of an embed run, in order. */
#define TURBO_EMBED_STAGE_TOKENIZE  0
#define TURBO_EMBED_STAGE_UPLOAD    1
#define TURBO_EMBED_STAGE_LOOKUP    2   /* word, position and type embedding lookup */
#define TURBO_EMBED_STAGE_ENCODE    3
#define TURBO_EMBED_STAGE_POOL      4
#define TURBO_EMBED_STAGE_NORMALIZE 5
#define TURBO_EMBED_STAGE_DOWNLOAD  6
#define TURBO_EMBED_STAGE_COUNT     7

/* Where a stage ran. */
#define TURBO_STAGE_UNUSED 0   /* the run had no such stage (rows written as tokens are
                                  not tokenized; a CPU uploads nothing), or the index is
                                  at or past stage_count */
#define TURBO_STAGE_HOST   1
#define TURBO_STAGE_DEVICE 2   /* its own kernel or graph node */
#define TURBO_STAGE_FUSED  3   /* inside the neighbouring stage's kernel */

/* Option values. 0 always means "what the bundle says". */

#define TURBO_TRUNCATE_MODEL 0
#define TURBO_TRUNCATE_NONE  1   /* too long is TURBO_E_CAPACITY */
#define TURBO_TRUNCATE_RIGHT 2
#define TURBO_TRUNCATE_LEFT  3

#define TURBO_PROMPT_NONE     0
#define TURBO_PROMPT_QUERY    1  /* the bundle's query prefix */
#define TURBO_PROMPT_DOCUMENT 2  /* the bundle's document prefix */

#define TURBO_NORMALIZE_MODEL 0
#define TURBO_NORMALIZE_NONE  1
#define TURBO_NORMALIZE_L2    2

#define TURBO_POOLING_MODEL 0
#define TURBO_POOLING_MEAN  1   /* the mean over the tokens whose mask is 1 */
#define TURBO_POOLING_CLS   2   /* the row's first column, whatever its mask */
#define TURBO_POOLING_LAST  3   /* the last token whose mask is 1 */

/* How a session computes. The artifact is chosen at load, in manifest
 * order; precision chooses how that artifact computes, never another one. */
#define TURBO_PRECISION_MODEL   0   /* the artifact's compute_dtype if the manifest fixes one,
                                       else the dtype its weights are stored in */
#define TURBO_PRECISION_FASTEST 1   /* the fastest compute dtype this backend has for the
                                       artifact, which may be below the weights' dtype */
#define TURBO_PRECISION_EXACT   2   /* F32 throughout */
/* When a precision computes in a dtype other than the one the weights are
 * stored in, the model keeps one resident copy of its weights per such
 * dtype. The copy is made by the first turbo_session_create that needs it,
 * is shared by every session of that model at that dtype, is released with
 * the model, and is never counted in a result's allocations. The session's
 * compute dtype is fixed at turbo_session_create; turbo_session_get_info
 * reports it before any run. */

/* Native memory handle kinds, for import and export. */
#define TURBO_HANDLE_HOST_PTR   1   /* aux unused */
#define TURBO_HANDLE_CUDA_PTR   2   /* aux = device ordinal */
#define TURBO_HANDLE_CL_MEM     3   /* aux = cl_context */
#define TURBO_HANDLE_ZE_USM     4   /* aux = ze_context_handle_t */
#define TURBO_HANDLE_MTL_BUFFER 5   /* aux = MTLDevice */
#define TURBO_HANDLE_DMABUF_FD  6   /* aux unused */

#define TURBO_ERROR_MESSAGE_LEN 496

/* ---- Small types ------------------------------------------------------- */

/* UTF-8 view. ptr may be NULL only when len is 0. Not NUL-terminated. */
typedef struct turbo_text {
    const char *ptr;
    uint64_t    len;
} turbo_text;

/* Caller-owned. On success code is 0 and message[0] is 0. */
typedef struct turbo_error {
    uint32_t struct_size;
    int32_t  code;
    uint32_t field;     /* 1-based field index for the two errors that name one, else 0 */
    char     message[TURBO_ERROR_MESSAGE_LEN];   /* NUL-terminated UTF-8 */
} turbo_error;

/* level: 0 error, 1 warning, 2 info, 3 debug. message is valid for the call. */
typedef void (*turbo_log_fn)(void *user_data, uint32_t level, turbo_text message);

/* ---- Runtime and devices ----------------------------------------------- */

typedef struct turbo_runtime_desc {
    uint32_t     struct_size;
    uint32_t     reserved;
    turbo_log_fn log;             /* may be NULL */
    void        *log_user_data;
} turbo_runtime_desc;

typedef struct turbo_device_info {
    uint32_t struct_size;
    uint32_t kind;               /* TURBO_DEVICE_* */
    uint32_t ordinal;            /* index within its backend */
    uint32_t unified_memory;     /* 1 if host and device share one memory */
    uint64_t memory_total;       /* bytes, 0 if unknown */
    uint64_t memory_free;        /* bytes at query time, 0 if unknown */
    char     arch[32];           /* the label benchmarks are filed under: rtx4080, b70, m2, ...;
                                  * for a CPU the instruction set, and a CPU record is
                                  * filed under arch and name together */
    char     name[128];          /* what the driver calls it */
    char     vendor[64];
    char     backend[32];        /* cuda, openvino, metal, hailo, ggml */
    char     runtime_version[64];   /* the vendor runtime this build linked */
    char     driver_version[64];    /* empty where there is none */
} turbo_device_info;

/* One cell of the matrix: what (device, task, precision) can do, and the
 * proof. Each precision is its own cell with its own record, so the caller
 * who asks for FASTEST can read what it costs before asking. The numbers are
 * for the bundle the record names; for another bundle, the dtype a precision
 * resolves to is what turbo_session_get_info reports. */
typedef struct turbo_capability {
    uint32_t struct_size;
    uint32_t status;              /* TURBO_CAP_* */
    uint32_t dtype;               /* compute dtype used, TURBO_DTYPE_*; 0 when UNSUPPORTED */
    uint32_t options_honored;     /* bit (i-1) set: field i of the task's options struct is
                                   * honored; 0 when UNSUPPORTED */
    float    cosine_floor;        /* lowest cosine against the fp32 reference in the record, 0 if none */
    float    speed_ratio;         /* our p50 latency over the fastest reference program's, from the
                                   * record, 0 if none */
    char     benchmark[96];       /* file name of the record that backs SUPPORTED, else empty */
    char     reason[160];         /* why it is not SUPPORTED, else empty */
} turbo_capability;

int32_t turbo_runtime_create(const turbo_runtime_desc *desc, turbo_runtime **out, turbo_error *err);
void    turbo_runtime_release(turbo_runtime *rt);

/* The library version and the backends compiled into this build, as a
 * static string: "0.1.0 cuda openvino". */
const char *turbo_version(void);
/* The status constant's name as this file spells it: "TURBO_OK" for 0,
 * "TURBO_E_CAPACITY" for 771, and "TURBO_E_UNKNOWN" for a code this file
 * does not define. The string is static. */
const char *turbo_status_name(int32_t code);

int32_t turbo_runtime_device_count(turbo_runtime *rt, uint32_t *out, turbo_error *err);
int32_t turbo_runtime_device_info(turbo_runtime *rt, uint32_t index, turbo_device_info *out, turbo_error *err);
/* precision is TURBO_PRECISION_*. */
int32_t turbo_runtime_capability(turbo_runtime *rt, uint32_t index, uint32_t task, uint32_t precision,
                                 turbo_capability *out, turbo_error *err);

/* The device that will run task fastest here: the highest capability status
 * at TURBO_PRECISION_MODEL, then the best speed_ratio among equals. Never a
 * CPU. Among devices that still tie, the first in device order. reason
 * receives one line saying why, if non-NULL and reason_len is not 0.
 * TURBO_E_DEVICE_NOT_FOUND when nothing offers the task. */
int32_t turbo_runtime_select(turbo_runtime *rt, uint32_t task, uint32_t *out,
                             char *reason, uint32_t reason_len, turbo_error *err);

/* ---- Context and buffers ----------------------------------------------- */

int32_t turbo_context_create(turbo_runtime *rt, uint32_t device, turbo_context **out, turbo_error *err);
void    turbo_context_release(turbo_context *ctx);
int32_t turbo_context_device(turbo_context *ctx, uint32_t *out, turbo_error *err);

typedef struct turbo_native_handle {
    uint32_t struct_size;
    uint32_t kind;        /* TURBO_HANDLE_* */
    uint64_t handle;      /* pointer, cl_mem, fd or object as an integer */
    uint64_t aux;
    uint64_t offset;      /* bytes into the allocation */
} turbo_native_handle;

/* Packed row-major. */
typedef struct turbo_buffer_desc {
    uint32_t struct_size;
    uint32_t placement;   /* TURBO_PLACE_* */
    uint32_t dtype;       /* TURBO_DTYPE_* */
    uint32_t ndim;        /* 1 or 2 */
    uint64_t shape[2];
    uint64_t bytes;       /* 0 = from shape and dtype */
} turbo_buffer_desc;

int32_t turbo_buffer_alloc(turbo_context *ctx, const turbo_buffer_desc *desc, turbo_buffer **out, turbo_error *err);

/* Wrap memory the caller owns. No copy. The caller keeps it alive and
 * unchanged while the buffer is in use. */
int32_t turbo_buffer_import(turbo_context *ctx, const turbo_buffer_desc *desc,
                            const turbo_native_handle *handle, turbo_buffer **out, turbo_error *err);
void    turbo_buffer_release(turbo_buffer *buf);
int32_t turbo_buffer_get_desc(turbo_buffer *buf, turbo_buffer_desc *out, turbo_error *err);

/* A host pointer for HOST, PINNED and SHARED placements; TURBO_E_UNSUPPORTED for DEVICE. */
int32_t turbo_buffer_host_ptr(turbo_buffer *buf, void **out, turbo_error *err);

/* The device's own handle, for a caller that continues on the device. */
int32_t turbo_buffer_export(turbo_buffer *buf, uint32_t kind, turbo_native_handle *out, turbo_error *err);

/* ---- Model ------------------------------------------------------------- */

/* Entries in turbo_model_info.output_dims. */
#define TURBO_OUTPUT_DIMS_MAX 16

typedef struct turbo_model_info {
    uint32_t struct_size;
    uint32_t task;              /* TURBO_TASK_* */
    uint32_t dim;               /* values per output row */
    uint32_t pooling;           /* embed: TURBO_POOLING_* as the bundle says */
    uint32_t normalize;         /* embed: TURBO_NORMALIZE_* as the bundle says */
    uint32_t max_seq;           /* tokens */
    uint32_t max_batch;         /* rows a session may take */
    uint32_t dtype;             /* the artifact's: its compute_dtype if the manifest fixes one,
                                   else its weights' storage dtype */
    char     model_id[128];
    char     revision[64];
    char     manifest_sha256[72];    /* hex, of manifest.json: the contract this load was made against */
    char     artifact_sha256[72];    /* hex, of the artifact file this device loaded */
    char     tokenizer_sha256[72];   /* hex, of the tokenizer file */
    char     prefix_query[128];      /* embed */
    char     prefix_document[128];   /* embed */
    uint32_t output_dims_count;      /* embed: entries of output_dims in use */
    uint32_t output_dims[TURBO_OUTPUT_DIMS_MAX];   /* embed: the widths besides dim that
                                                      turbo_embed_options.output_dim may
                                                      name, ascending; 0 past the count */
    /* Fields marked embed are 0 or empty for a model of another task. The
     * library also accepts a struct_size that ends before output_dims_count
     * and then writes nothing from there on. */
} turbo_model_info;

/* Load a bundle directory on the context's device. Every file is checked
 * against the manifest's hash before use. A device whose backend loads no
 * models is TURBO_E_UNSUPPORTED before the bundle is read. */
int32_t turbo_model_load(turbo_context *ctx, turbo_text bundle_path, turbo_model **out, turbo_error *err);
void    turbo_model_release(turbo_model *m);
int32_t turbo_model_get_info(turbo_model *m, turbo_model_info *out, turbo_error *err);

/* ---- Tokenizer --------------------------------------------------------- */

typedef struct turbo_tokenizer_info {
    uint32_t struct_size;
    uint32_t vocab_size;
    uint32_t max_seq;
    uint32_t specials_per_sequence;
    int32_t  pad_id;    /* or -1 */
    int32_t  bos_id;
    int32_t  eos_id;
    int32_t  unk_id;
    char     kind[32];  /* wordpiece, bpe, unigram */
    char     sha256[72];            /* hex, of the tokenizer file */
    char     manifest_sha256[72];   /* hex, of manifest.json */
} turbo_tokenizer_info;

/* Fields are numbered from 1 for turbo_error.field. 0 in every field, or
 * opts NULL, is what the bundle says. */
typedef struct turbo_encode_options {
    uint32_t struct_size;
    uint32_t omit_special_tokens;  /* 1: 0 = the bundle's template, 1 = the text's ids alone */
    uint32_t truncate;             /* 2: TURBO_TRUNCATE_* */
    uint32_t max_tokens;           /* 3: including specials; 0 = the bundle's max_seq. No
                                      session limit applies here: rows are cut where told */
    uint32_t prompt_role;          /* 4: TURBO_PROMPT_* */
} turbo_encode_options;

/* The tokenizer the bundle names. Thread-safe. The same ids on every machine. */
int32_t turbo_tokenizer_create(turbo_runtime *rt, turbo_text bundle_path, turbo_tokenizer **out, turbo_error *err);
void    turbo_tokenizer_release(turbo_tokenizer *t);
int32_t turbo_tokenizer_get_info(turbo_tokenizer *t, turbo_tokenizer_info *out, turbo_error *err);

/* Encode count texts into caller-owned [count, row_stride] int32 arrays.
 * Rows are padded with pad_id and mask 0 to row_stride. types may be NULL.
 * lengths, if non-NULL, receives each row's live token count. */
int32_t turbo_tokenizer_encode(turbo_tokenizer *t, const turbo_text *texts, uint32_t count,
                               const turbo_encode_options *opts,
                               int32_t *ids, int32_t *mask, int32_t *types,
                               uint32_t row_stride, uint32_t *lengths, turbo_error *err);

/* Tokens text produces with the prompt role's prefix (TURBO_PROMPT_*), with
 * special tokens and no truncation. */
int32_t turbo_tokenizer_count(turbo_tokenizer *t, turbo_text text, uint32_t prompt_role,
                              uint32_t *out, turbo_error *err);

/* ---- Session and run --------------------------------------------------- */

/* Fields are numbered from 1 for turbo_error.field. A precision the
 * artifact cannot compute in (EXACT on one compiled to a lower dtype, say)
 * fails with TURBO_E_UNSUPPORTED_OPTION naming field 3. */
typedef struct turbo_session_desc {
    uint32_t struct_size;
    uint32_t max_batch;   /* 1: 0 = the model's */
    uint32_t max_seq;     /* 2: 0 = the model's */
    uint32_t precision;   /* 3: TURBO_PRECISION_* */
} turbo_session_desc;

/* Fields are numbered from 1 for turbo_error.field and options_honored.
 * 0 in any field means "what the bundle says". */
typedef struct turbo_embed_options {
    uint32_t struct_size;
    uint32_t truncate;      /* 1: TURBO_TRUNCATE_* */
    uint32_t max_tokens;    /* 2: token budget per row, specials included; 0 = the
                               bundle's embed.max_seq, not the session's. Above the
                               session's max_seq is TURBO_E_CAPACITY */
    uint32_t prompt_role;   /* 3: TURBO_PROMPT_* */
    uint32_t normalize;     /* 4: TURBO_NORMALIZE_* */
    uint32_t pooling;       /* 5: TURBO_POOLING_* */
    uint32_t output_dim;    /* 6: keep only the first output_dim values of each vector, cut
                               before normalize: an L2-normalized vector is unit length at
                               output_dim. Above turbo_model_info.dim is
                               TURBO_E_INVALID_ARGUMENT; neither dim nor one of
                               turbo_model_info.output_dims is TURBO_E_UNSUPPORTED_OPTION */
} turbo_embed_options;

/* Caller-prepared rows, [batch, seq] int32, row_stride elements between row
 * starts (0 = seq). types may be NULL for all zero. The memory may be a
 * buffer the caller imported, in which case nothing is copied on the host
 * on its way to a device with its own memory; a CPU session copies the
 * rows into its own. On a device with its own memory, page-locked or
 * managed rows go straight to the device, and pageable rows are staged
 * through the session's page-locked memory first. Every row has at least one mask entry of 1. The rows
 * are already cut: truncate and prompt_role are for text, and a value
 * other than 0 in either is TURBO_E_INVALID_ARGUMENT naming it. max_tokens
 * is checked, not applied: a row whose tokens through its last mask entry
 * of 1 are more than a max_tokens other than 0 is TURBO_E_CAPACITY. */
typedef struct turbo_token_batch {
    uint32_t       struct_size;
    uint32_t       batch;
    uint32_t       seq;
    uint32_t       row_stride;
    const int32_t *ids;
    const int32_t *mask;
    const int32_t *types;
} turbo_token_batch;

typedef struct turbo_session_info {
    uint32_t struct_size;
    uint32_t max_batch;       /* in effect, after 0 was resolved */
    uint32_t max_seq;
    uint32_t precision;       /* TURBO_PRECISION_* asked for */
    uint32_t compute_dtype;   /* TURBO_DTYPE_* it resolved to */
    uint32_t reserved;
} turbo_session_info;

int32_t turbo_session_create(turbo_model *m, const turbo_session_desc *desc, turbo_session **out, turbo_error *err);
void    turbo_session_release(turbo_session *s);
int32_t turbo_session_get_info(turbo_session *s, turbo_session_info *out, turbo_error *err);

/* Tokenize with the bundle's tokenizer, then write the rows. opts may be NULL. */
int32_t turbo_embed_write_text(turbo_session *s, const turbo_text *texts, uint32_t count,
                               const turbo_embed_options *opts, turbo_error *err);

/* Write rows the caller tokenized. opts may be NULL. */
int32_t turbo_embed_write_tokens(turbo_session *s, const turbo_token_batch *batch,
                                 const turbo_embed_options *opts, turbo_error *err);

/* Run what was written. A write replaces what an earlier write left, and
 * a run takes it, even when the run fails: a run with nothing written
 * since the last run, or after a failed write, is TURBO_E_INVALID_STATE.
 * The result holds the session until it and every buffer from
 * turbo_result_buffer are released; until then a write or run on the
 * session is TURBO_E_BUSY. */
int32_t turbo_session_run(turbo_session *s, turbo_result **out, turbo_error *err);

/* ---- Result ------------------------------------------------------------ */

/* One summary per run: what produced it and what it cost. */
typedef struct turbo_result_info {
    uint32_t struct_size;
    uint32_t task;                             /* TURBO_TASK_* */
    uint32_t batch;
    uint32_t dim;                              /* values per row: the vector width for embed */
    uint32_t dtype;                            /* TURBO_DTYPE_* of the output: F32 in this cut,
                                                  whatever the compute dtype */
    uint32_t compute_dtype;                    /* TURBO_DTYPE_* the session's precision ran in */
    uint32_t placement;                        /* TURBO_PLACE_* where they are now */
    uint32_t device;                           /* runtime device index */
    uint64_t bytes;
    uint64_t h2d_bytes;                        /* every byte that crossed to the device in this run */
    uint64_t d2h_bytes;                        /* every byte that crossed back, including reads */
    uint64_t host_allocs;                      /* heap allocations in turbo_session_run */
    uint64_t device_allocs;                    /* device allocations in turbo_session_run */
    uint32_t stage_count;                      /* the task's TURBO_<TASK>_STAGE_COUNT */
    uint32_t reserved;
    uint32_t stage[TURBO_STAGE_MAX];           /* by TURBO_<TASK>_STAGE_*: TURBO_STAGE_HOST / DEVICE /
                                                  FUSED / UNUSED */
    char     backend[32];
    char     arch[32];
    char     runtime_version[64];
    char     manifest_sha256[72];
    char     artifact_sha256[72];
    char     tokenizer_sha256[72];
} turbo_result_info;

int32_t turbo_result_get_info(turbo_result *r, turbo_result_info *out, turbo_error *err);

/* The output of a run is one block, the vectors: [batch, dim] values of
 * turbo_result_info.dtype, packed row-major, turbo_result_info.bytes long. */

/* Copy the vectors into dst (capacity bytes). Counted in d2h_bytes. */
int32_t turbo_result_read(turbo_result *r, void *dst, uint64_t capacity, uint64_t *written, turbo_error *err);

/* The vectors where they are, without a copy. The buffer keeps the result alive. */
int32_t turbo_result_buffer(turbo_result *r, turbo_buffer **out, turbo_error *err);

void    turbo_result_release(turbo_result *r);

#ifdef __cplusplus
}
#endif

#endif /* TURBO_H */
