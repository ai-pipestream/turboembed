/* SPDX-License-Identifier: Apache-2.0
 * Additive prepared execution extension. Existing turboembed.h ABI is unchanged.
 * All functions execute in process and never download models.
 * All raw handles: release must not race any other operation on that same
 * handle, including result_read/result_release. Callers must coordinate release
 * and must never use a released handle. Dependent handles retain their parents.
 */
#ifndef TURBOEMBED_PREPARED_H
#define TURBOEMBED_PREPARED_H
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
# if defined(TURBOEMBED_PREPARED_BUILD)
#  define TE_PREPARED_API __declspec(dllexport)
# else
#  define TE_PREPARED_API __declspec(dllimport)
# endif
#else
# define TE_PREPARED_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

#define TE_PREPARED_VERSION 1u
#define TE_OK 0u
#define TE_INVALID_ARGUMENT 1u
#define TE_NOT_FOUND 2u
#define TE_NOT_IMPLEMENTED 3u
#define TE_UNAVAILABLE 4u
#define TE_INTERNAL 5u
#define TE_OUT_OF_MEMORY 6u
#define TE_BUSY 7u
#define TE_ABI_MISMATCH 8u
#define TE_INTEGRITY_ERROR 9u
#define TE_DEVICE_AUTO 0u
#define TE_DEVICE_OPENVINO_GPU 1u
#define TE_DEVICE_OPENVINO_CPU 2u
#define TE_CAP_TEXT 1ull
#define TE_CAP_PREPARED_I32 2ull
#define TE_CAP_OPENCL_RESULT 4ull
#define TE_CAP_HOST_READ 8ull

typedef struct te_context te_context;
typedef struct te_model te_model;
typedef struct te_slot te_slot;
typedef struct te_result te_result;

/* Optional caller-owned error storage. On success code=0 and message is empty.
 * Message is always NUL terminated. No shared last-error buffer is used. */
typedef struct te_error { uint32_t code; char message[508]; } te_error;

/* Every descriptor must be initialized with sizeof(descriptor) and version 1.
 * Unknown sizes, versions and input options are rejected. Reserved output
 * fields are written as zero. Output descriptors are unspecified on failure. */
typedef struct te_context_options {
    uint32_t struct_size, version, device, ordinal;
} te_context_options;
typedef struct te_context_info {
    uint32_t struct_size, version, device, ordinal;
    uint64_t capabilities;
    char device_name[128];
    char runtime_version[128];
    char driver_version[128];
} te_context_info;
typedef struct te_model_info {
    uint32_t struct_size, version, dimension, vocab_size;
    uint32_t max_sequence_length, max_batch_size, normalized, reserved;
    char model_id[128];
    char revision[64];
    char tokenizer_sha256[65];
    char pooling[15];
} te_model_info;
typedef struct te_slot_options {
    uint32_t struct_size, version, batch, sequence_length;
} te_slot_options;
typedef struct te_result_info {
    uint32_t struct_size, version, batch, dimension;
    uint64_t byte_size;
} te_result_info;
typedef struct te_opencl_view {
    uint32_t struct_size, version;
    uintptr_t context, queue, buffer;
    uint64_t byte_size;
    uint32_t batch, dimension;
} te_opencl_view;
typedef struct te_slot_stats {
    uint32_t struct_size, version;
    uint64_t executions, input_write_bytes, output_read_bytes;
    uint64_t owned_input_bytes, owned_output_bytes;
} te_slot_stats;
typedef struct te_text { const char *ptr; uint64_t byte_length; } te_text;

TE_PREPARED_API uint32_t turboembed_prepared_v1_version(void);
TE_PREPARED_API uint32_t turboembed_prepared_v1_context_create(
    const te_context_options *, te_context **out, te_error *);
TE_PREPARED_API uint32_t turboembed_prepared_v1_context_info(
    const te_context *, te_context_info *out, te_error *);
TE_PREPARED_API void turboembed_prepared_v1_context_release(te_context *);

/* bundle_path is a nonempty UTF-8 span without NUL. The bundle is verified and
 * captured in memory before use. Models retain their context independently. */
TE_PREPARED_API uint32_t turboembed_prepared_v1_model_load(
    const te_context *, const char *bundle_path, uint64_t path_length,
    te_model **out, te_error *);
TE_PREPARED_API uint32_t turboembed_prepared_v1_model_info(
    const te_model *, te_model_info *out, te_error *);
TE_PREPARED_API void turboembed_prepared_v1_model_release(te_model *);

/* Slots retain their model and compile a fixed shape at creation. Creation may
 * allocate/compile; execute never changes shape or recompiles. Distinct slots
 * have independent requests/queues. Operations on one slot return BUSY when an
 * operation or result lease is active. Handle release must not race a call on
 * that same handle. Dependent handles/results survive parent-handle release. */
TE_PREPARED_API uint32_t turboembed_prepared_v1_slot_create(
    const te_model *, const te_slot_options *, te_slot **out, te_error *);
TE_PREPARED_API void turboembed_prepared_v1_slot_release(te_slot *);

/* Contiguous row-major i32 input arrays, count == batch * sequence_length.
 * IDs must fit model vocabulary, masks and types must be 0/1. types may be NULL
 * to select all zeros; ids/mask are required. Copies complete before return.
 * A successful write may be reused across executions. A failed write/text call
 * that acquires the slot invalidates its inputs; execute then returns
 * INVALID_ARGUMENT. BUSY leaves an active operation/result unchanged. */
TE_PREPARED_API uint32_t turboembed_prepared_v1_slot_write_tokens(
    te_slot *, const int32_t *ids, const int32_t *mask, const int32_t *types,
    uint64_t count, te_error *);
/* Exactly batch UTF-8 spans. NULL is valid only with length zero. Embedded NUL
 * is supported. Native model tokenization writes preallocated host scratch,
 * then uploads to the same input buffers used by write_tokens/execute. */
TE_PREPARED_API uint32_t turboembed_prepared_v1_slot_write_text(
    te_slot *, const te_text *, uint64_t count, te_error *);
/* Synchronous execution. On success *out leases the slot's reusable output.
 * Release the result before another slot write/execute. Lease bookkeeping is
 * preallocated. Failure clears *out. C callers must not reuse a released handle. */
TE_PREPARED_API uint32_t turboembed_prepared_v1_slot_execute(
    te_slot *, te_result **out, te_error *);
/* Byte counters describe explicit GPU uploads/readbacks by this adapter only.
 * Owned input/output bytes describe the bound tensors (device for GPU, host
 * for CPU). A GPU slot also owns host staging of the same size as its input
 * tensors; that staging is excluded from these counters. These are not process
 * allocation counters and exclude tokenizer scratch and provider internals. */
TE_PREPARED_API uint32_t turboembed_prepared_v1_slot_stats(
    const te_slot *, te_slot_stats *out, te_error *);
TE_PREPARED_API uint32_t turboembed_prepared_v1_result_info(
    const te_result *, te_result_info *out, te_error *);
/* Explicit blocking copy of row-major f32 into caller memory. Capacity is in
 * float elements and must be at least batch * dimension. No implicit readback
 * occurs in execute. Output contents are unspecified on failure. */
TE_PREPARED_API uint32_t turboembed_prepared_v1_result_read(
    const te_result *, float *out, uint64_t capacity, te_error *);
/* GPU only. Borrowed OpenCL RESOURCE HANDLES, never host data pointers.
 * Treat the output buffer as read-only. Use the returned in-order queue/context;
 * do not release the borrowed references to these handles.
 * Results retain owners until release. Release waits for this queue to finish
 * before permitting output reuse. Work on any other queue must finish before
 * release; cross-queue/asynchronous imports are not supported by this version. */
TE_PREPARED_API uint32_t turboembed_prepared_v1_result_opencl(
    const te_result *, te_opencl_view *out, te_error *);
TE_PREPARED_API uint32_t turboembed_prepared_v1_result_release(te_result *, te_error *);

#ifdef __cplusplus
}
#endif
#endif
