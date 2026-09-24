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
 *   - A backend that offers context_create offers context_release, and one
 *     that offers buffer_alloc or buffer_import offers buffer_release. The
 *     core refuses a table that does not.
 *   - The core checks every struct's struct_size, every enumeration and
 *     every shape before it calls, and fills in turbo_buffer_desc.bytes.
 *     It releases every buffer before the context it was made on, and
 *     each handle once. Release functions return nothing, as turbo.h's do.
 */

#ifndef TURBO_BACKEND_H
#define TURBO_BACKEND_H

#include <turbo/turbo.h>

#ifdef __cplusplus
extern "C" {
#endif

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
} turbo_backend;

#ifdef __cplusplus
}
#endif

#endif /* TURBO_BACKEND_H */
