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
} turbo_backend;

#ifdef __cplusplus
}
#endif

#endif /* TURBO_BACKEND_H */
