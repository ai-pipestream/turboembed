/* SPDX-License-Identifier: Apache-2.0
 *
 * TurboEmbed C interface. Hand-written. This file is the design; the code
 * follows it.
 *
 * First cut: text embedding, tokens in and vectors out, on one device. The
 * other tasks and chunking are added here when they are built, not before.
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
                                      benchmark record named in the cell exists */

#define TURBO_DTYPE_I32   8
#define TURBO_DTYPE_F16  10
#define TURBO_DTYPE_BF16 11
#define TURBO_DTYPE_F32  12

#define TURBO_PLACE_HOST    1   /* pageable host memory */
#define TURBO_PLACE_PINNED  2   /* page-locked host memory the device can read */
#define TURBO_PLACE_DEVICE  3   /* device memory */
#define TURBO_PLACE_SHARED  4   /* one allocation both can address */

/* Pipeline stages of an embed run, in order. */
#define TURBO_STAGE_TOKENIZE  0
#define TURBO_STAGE_UPLOAD    1
#define TURBO_STAGE_ENCODE    2
#define TURBO_STAGE_POOL      3
#define TURBO_STAGE_NORMALIZE 4
#define TURBO_STAGE_DOWNLOAD  5
#define TURBO_STAGE_COUNT     6

/* Where a stage ran. */
#define TURBO_STAGE_UNUSED 0   /* the model does not have this stage */
#define TURBO_STAGE_HOST   1
#define TURBO_STAGE_DEVICE 2   /* its own kernel or graph node */
#define TURBO_STAGE_FUSED  3   /* inside the neighbouring stage's kernel */

/* Option values. 0 always means "what the bundle says". */
#define TURBO_OPT_MODEL 0

#define TURBO_TRUNCATE_MODEL 0
#define TURBO_TRUNCATE_NONE  1   /* too long is an error */
#define TURBO_TRUNCATE_RIGHT 2
#define TURBO_TRUNCATE_LEFT  3

#define TURBO_PROMPT_NONE     0
#define TURBO_PROMPT_QUERY    1  /* the bundle's query prefix */
#define TURBO_PROMPT_DOCUMENT 2  /* the bundle's document prefix */

#define TURBO_NORMALIZE_MODEL 0
#define TURBO_NORMALIZE_NONE  1
#define TURBO_NORMALIZE_L2    2

#define TURBO_POOLING_MODEL 0
#define TURBO_POOLING_MEAN  1
#define TURBO_POOLING_CLS   2
#define TURBO_POOLING_LAST  3

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
    char     arch[32];           /* the label benchmarks are filed under: rtx4080, b70, m2, ... */
    char     name[128];          /* what the driver calls it */
    char     vendor[64];
    char     backend[32];        /* cuda, openvino, metal, hailo, ggml */
    char     runtime_version[64];   /* the vendor runtime this build linked */
    char     driver_version[64];    /* empty where there is none */
} turbo_device_info;

/* One cell of the matrix: what (device, task) can do, and the proof. */
typedef struct turbo_capability {
    uint32_t struct_size;
    uint32_t status;              /* TURBO_CAP_* */
    uint32_t dtype;               /* compute dtype used, TURBO_DTYPE_* */
    uint32_t options_honored;     /* bit (i-1) set: field i of turbo_embed_options is honored */
    float    cosine_floor;        /* lowest cosine against the fp32 reference in the record, 0 if none */
    float    speed_ratio;         /* our p50 latency over the reference's, from the record, 0 if none */
    char     benchmark[96];       /* file name of the record that backs SUPPORTED, else empty */
    char     reason[160];         /* why it is not SUPPORTED, else empty */
} turbo_capability;

int32_t turbo_runtime_create(const turbo_runtime_desc *desc, turbo_runtime **out, turbo_error *err);
void    turbo_runtime_release(turbo_runtime *rt);

/* The library version and the backends compiled into this build, as a
 * static string: "0.1.0 cuda openvino". */
const char *turbo_version(void);
const char *turbo_status_name(int32_t code);

int32_t turbo_runtime_device_count(turbo_runtime *rt, uint32_t *out, turbo_error *err);
int32_t turbo_runtime_device_info(turbo_runtime *rt, uint32_t index, turbo_device_info *out, turbo_error *err);
int32_t turbo_runtime_capability(turbo_runtime *rt, uint32_t index, uint32_t task, turbo_capability *out, turbo_error *err);

/* The device that will run task fastest here: the highest capability status,
 * then the best speed_ratio among equals. Never a CPU. reason receives one
 * line saying why, if non-NULL. TURBO_E_DEVICE_NOT_FOUND when nothing
 * offers the task. */
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

typedef struct turbo_model_info {
    uint32_t struct_size;
    uint32_t task;              /* TURBO_TASK_* */
    uint32_t dim;
    uint32_t pooling;           /* TURBO_POOLING_* as the bundle says */
    uint32_t normalize;         /* TURBO_NORMALIZE_* as the bundle says */
    uint32_t max_seq;           /* tokens */
    uint32_t max_batch;         /* rows a session may take */
    uint32_t dtype;             /* compute dtype in use */
    char     model_id[128];
    char     revision[64];
    char     artifact_sha256[72];    /* hex, of the artifact file this device loaded */
    char     tokenizer_sha256[72];   /* hex, of the tokenizer file */
    char     prefix_query[128];
    char     prefix_document[128];
} turbo_model_info;

/* Load a bundle directory on the context's device. Every file is checked
 * against the manifest's hash before use. */
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
    char     sha256[72];
} turbo_tokenizer_info;

typedef struct turbo_encode_options {
    uint32_t struct_size;
    uint32_t add_special_tokens;   /* 1 = yes (the default when opts is NULL) */
    uint32_t truncate;             /* TURBO_TRUNCATE_* */
    uint32_t max_tokens;           /* including specials; 0 = the bundle's max_seq */
    uint32_t prompt_role;          /* TURBO_PROMPT_* */
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

/* Tokens text produces, with special tokens and no truncation. */
int32_t turbo_tokenizer_count(turbo_tokenizer *t, turbo_text text, uint32_t *out, turbo_error *err);

/* ---- Session and run --------------------------------------------------- */

typedef struct turbo_session_desc {
    uint32_t struct_size;
    uint32_t max_batch;   /* 0 = the model's */
    uint32_t max_seq;     /* 0 = the model's */
} turbo_session_desc;

/* Fields are numbered from 1 for turbo_error.field and options_honored.
 * 0 in any field means "what the bundle says". */
typedef struct turbo_embed_options {
    uint32_t struct_size;
    uint32_t truncate;      /* 1: TURBO_TRUNCATE_* */
    uint32_t max_tokens;    /* 2: token budget per row */
    uint32_t prompt_role;   /* 3: TURBO_PROMPT_* */
    uint32_t normalize;     /* 4: TURBO_NORMALIZE_* */
    uint32_t pooling;       /* 5: TURBO_POOLING_* */
    uint32_t output_dim;    /* 6: keep only the first output_dim values of each vector */
} turbo_embed_options;

/* Caller-prepared rows, [batch, seq] int32, row_stride elements between row
 * starts (0 = seq). types may be NULL for all zero. The memory may be a
 * buffer the caller imported, in which case nothing is copied on the host. */
typedef struct turbo_token_batch {
    uint32_t       struct_size;
    uint32_t       batch;
    uint32_t       seq;
    uint32_t       row_stride;
    const int32_t *ids;
    const int32_t *mask;
    const int32_t *types;
} turbo_token_batch;

int32_t turbo_session_create(turbo_model *m, const turbo_session_desc *desc, turbo_session **out, turbo_error *err);
void    turbo_session_release(turbo_session *s);

/* Tokenize with the bundle's tokenizer, then write the rows. opts may be NULL. */
int32_t turbo_session_write_text(turbo_session *s, const turbo_text *texts, uint32_t count,
                                 const turbo_embed_options *opts, turbo_error *err);

/* Write rows the caller tokenized. opts may be NULL. */
int32_t turbo_session_write_tokens(turbo_session *s, const turbo_token_batch *batch,
                                   const turbo_embed_options *opts, turbo_error *err);

/* Run what was written. The result holds the session until released. */
int32_t turbo_session_run(turbo_session *s, turbo_result **out, turbo_error *err);

/* ---- Result ------------------------------------------------------------ */

/* One summary per run: what produced it and what it cost. */
typedef struct turbo_result_info {
    uint32_t struct_size;
    uint32_t batch;
    uint32_t dim;
    uint32_t dtype;                            /* TURBO_DTYPE_* of the vectors */
    uint32_t placement;                        /* TURBO_PLACE_* where they are now */
    uint32_t device;                           /* runtime device index */
    uint64_t bytes;
    uint32_t stage[TURBO_STAGE_COUNT];         /* TURBO_STAGE_HOST / DEVICE / FUSED / UNUSED */
    uint32_t reserved;
    uint64_t h2d_bytes;                        /* every byte that crossed to the device in this run */
    uint64_t d2h_bytes;                        /* every byte that crossed back, including reads */
    uint64_t host_allocs;                      /* heap allocations on the run path */
    uint64_t device_allocs;                    /* device allocations on the run path */
    char     backend[32];
    char     arch[32];
    char     runtime_version[64];
    char     artifact_sha256[72];
    char     tokenizer_sha256[72];
} turbo_result_info;

int32_t turbo_result_get_info(turbo_result *r, turbo_result_info *out, turbo_error *err);

/* Copy the vectors into dst (capacity bytes). Counted in d2h_bytes. */
int32_t turbo_result_read(turbo_result *r, void *dst, uint64_t capacity, uint64_t *written, turbo_error *err);

/* The vectors where they are, without a copy. The buffer keeps the result alive. */
int32_t turbo_result_buffer(turbo_result *r, turbo_buffer **out, turbo_error *err);

void    turbo_result_release(turbo_result *r);

#ifdef __cplusplus
}
#endif

#endif /* TURBO_H */
