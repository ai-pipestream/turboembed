// SPDX-License-Identifier: Apache-2.0
//
// The Metal backend: Apple GPUs through Metal, behind
// include/turbo/turbo_backend.h. It lists the devices Metal reports, keeps
// a command queue and the compiled kernels per context, reads a model's
// weights where the core holds them, and runs embed sessions with the BERT
// encoder in kernels.metal.
//
// It runs on Apple silicon, where the GPU and the host share one memory:
// the weights, the rows and the vectors are memory both can address, and
// nothing is copied to reach the device. A GPU with memory of its own is
// listed but runs nothing.
//
// Every function here is called from any thread. A context's queue is
// used under the context's lock, which a run holds from encoding its
// commands until they complete.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <turbo/turbo_backend.h>

/* TURBO_METAL_SDK, the macOS SDK this build compiled against, and
 * TURBO_METAL_KERNELS, kernels.metal's source. */
#include "turbo_metal_build.h"

namespace {

// ---- Errors ------------------------------------------------------------------
//
// Nothing throws across the table: each entry runs its body inside
// guarded(), which turns an exception into a status, in an autorelease
// pool so the Objective-C objects a call makes go with it.

int32_t vrefuse(turbo_error *err, int32_t code, uint32_t field, const char *fmt, va_list ap) {
    if (err) {
        err->code = code;
        err->field = field;
        vsnprintf(err->message, TURBO_ERROR_MESSAGE_LEN, fmt, ap);
    }
    return code;
}

__attribute__((format(printf, 3, 4))) int32_t refuse(turbo_error *err, int32_t code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vrefuse(err, code, 0, fmt, ap);
    va_end(ap);
    return code;
}

__attribute__((format(printf, 4, 5))) int32_t refuse_field(turbo_error *err, int32_t code, uint32_t field,
                                                           const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vrefuse(err, code, field, fmt, ap);
    va_end(ap);
    return code;
}

/* An NSError's text, for a message. */
const char *text(NSError *e) { return e ? e.localizedDescription.UTF8String : "no reason given"; }

#define TRY(expr)                                                                                                      \
    do {                                                                                                               \
        const int32_t rc_ = (expr);                                                                                    \
        if (rc_ != TURBO_OK) return rc_;                                                                               \
    } while (0)

template <typename F> int32_t guarded(turbo_error *err, F f) noexcept {
    @autoreleasepool {
        try {
            return f();
        } catch (const std::bad_alloc &) {
            return refuse(err, TURBO_E_OUT_OF_MEMORY, "host memory for the metal backend");
        } catch (...) {
            return refuse(err, TURBO_E_INTERNAL, "an exception inside the metal backend");
        }
    }
}

// ---- Allocation counts ---------------------------------------------------------
//
// Every allocation the backend makes goes through one of the functions
// below, which count it twice: in the process's total, which the tests
// read through turbo_metal_allocations, and in the calling thread's own
// count, whose change across session_run is what a result reports. A
// Metal buffer is a device allocation; the backend's own structures and
// host buffers are host allocations. The command buffer and encoder a
// run needs are Metal's own objects, which Metal makes for every
// submission and which are not the backend's allocations.

std::atomic<uint64_t> host_total{0};
std::atomic<uint64_t> device_total{0};
thread_local uint64_t host_here = 0;
thread_local uint64_t device_here = 0;

void counted_host() {
    host_total.fetch_add(1, std::memory_order_relaxed);
    host_here++;
}

void counted_device() {
    device_total.fetch_add(1, std::memory_order_relaxed);
    device_here++;
}

template <typename T, typename... A> T *make(A &&...a) {
    T *p = new T(std::forward<A>(a)...);
    counted_host();
    return p;
}

/* Every host buffer starts on a 64-byte boundary, as on the CPU backend. */
constexpr size_t HOST_ALIGN = 64;

size_t round_up(size_t n, size_t a) { return (n + a - 1) / a * a; }

void *host_malloc(size_t n) {
    void *p = aligned_alloc(HOST_ALIGN, round_up(n ? n : 1, HOST_ALIGN));
    if (p) counted_host();
    return p;
}

id<MTLBuffer> new_buffer(id<MTLDevice> d, size_t n, MTLResourceOptions o) {
    id<MTLBuffer> b = [d newBufferWithLength:(n ? n : 1) options:o];
    if (b) counted_device();
    return b;
}

/* A Metal buffer over host memory already mapped, without a copy, or nil
 * when Metal will not take it: it takes whole pages only. */
id<MTLBuffer> wrap(id<MTLDevice> d, void *p, size_t n) {
    const size_t page = (size_t)getpagesize();
    if ((uintptr_t)p % page != 0 || n == 0 || n % page != 0) return nil;
    id<MTLBuffer> b = [d newBufferWithBytesNoCopy:p
                                           length:n
                                          options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                                      deallocator:nil];
    if (b) counted_device();
    return b;
}

/* src into dst's len bytes, cut to fit and NUL-terminated. */
void copy_str(char *dst, size_t len, const char *src) {
    if (len == 0) return;
    const size_t n = strnlen(src, len - 1);
    memcpy(dst, src, n);
    dst[n] = 0;
}

// ---- Devices -------------------------------------------------------------------

/* The devices, probed once and held for the life of the process, so an
 * ordinal names the same device on every call. Metal lists only devices
 * that are present; a Mac without one lists nothing. */
NSArray<id<MTLDevice>> *devices() {
    static NSArray<id<MTLDevice>> *all = [] {
        @autoreleasepool {
            NSArray<id<MTLDevice>> *found = MTLCopyAllDevices();
            return found ? found : @[];
        }
    }();
    return all;
}

id<MTLDevice> device(uint32_t ordinal) {
    NSArray<id<MTLDevice>> *all = devices();
    return ordinal < all.count ? all[ordinal] : nil;
}

int32_t not_listed(turbo_error *err, uint32_t ordinal) {
    return refuse(err, TURBO_E_INVALID_ARGUMENT, "metal device %u: %lu listed", ordinal,
                  (unsigned long)devices().count);
}

/* The label benchmarks are filed under: the chip, lowercased with the
 * spaces dropped ("Apple M2 Pro" is m2pro). A GPU that is not Apple's
 * keeps its whole name, less trademark marks. */
std::string arch_label(const std::string &name) {
    std::string s = name.rfind("Apple ", 0) == 0 ? name.substr(6) : name;
    for (const char *mark : {"(R)", "(TM)"})
        for (size_t at; (at = s.find(mark)) != std::string::npos;) s.erase(at, strlen(mark));
    std::string out;
    for (unsigned char c : s)
        if (std::isalnum(c)) out += (char)std::tolower(c);
    return out;
}

/* Metal has no vendor field; the name leads with it ("Apple M2", "AMD
 * Radeon Pro 5500M", "Intel(R) UHD Graphics 630"). */
std::string vendor(const std::string &name) {
    std::string first = name.substr(0, name.find(' '));
    return first.substr(0, first.find('('));
}

/* What the host has free now: free and inactive pages, as the kernel
 * counts them. 0 if it does not say. */
uint64_t host_free() {
    vm_statistics64_data_t vm;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    mach_port_t host = mach_host_self();
    kern_return_t kr = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &n);
    mach_port_deallocate(mach_task_self(), host);
    if (kr != KERN_SUCCESS) return 0;
    return ((uint64_t)vm.free_count + vm.inactive_count) * vm_kernel_page_size;
}

/* The operating system is the driver: Metal ships with it. */
std::string os_version() {
    NSOperatingSystemVersion v = [NSProcessInfo processInfo].operatingSystemVersion;
    std::string s = "macOS " + std::to_string(v.majorVersion) + "." + std::to_string(v.minorVersion);
    if (v.patchVersion) s += "." + std::to_string(v.patchVersion);
    char build[32] = {0};
    size_t len = sizeof build - 1;
    if (sysctlbyname("kern.osversion", build, &len, nullptr, 0) == 0 && build[0]) s += std::string(" (") + build + ")";
    return s;
}

const char RUNTIME[] = "Metal, macOS SDK " TURBO_METAL_SDK;

int32_t device_count(uint32_t *out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        *out = (uint32_t)devices().count;
        return TURBO_OK;
    });
}

int32_t device_info(uint32_t ordinal, turbo_device_info *out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        id<MTLDevice> d = device(ordinal);
        if (!d) return not_listed(err, ordinal);
        const std::string name = d.name.UTF8String ?: "";
        const bool unified = d.hasUnifiedMemory;
        // What Metal lets this device use without paging, the most a model
        // on it can hold.
        const uint64_t total = d.recommendedMaxWorkingSetSize;
        const uint64_t used = d.currentAllocatedSize;
        const uint64_t left = total > used ? total - used : 0;
        out->kind = unified ? TURBO_DEVICE_IGPU : TURBO_DEVICE_GPU;
        out->ordinal = ordinal;
        out->unified_memory = unified ? 1 : 0;
        out->memory_total = total;
        // With one memory the host's other users count against it too.
        // Metal does not say what other processes hold on a discrete GPU,
        // so there it is unknown.
        out->memory_free = unified ? std::min(left, host_free()) : 0;
        copy_str(out->arch, sizeof out->arch, arch_label(name).c_str());
        copy_str(out->name, sizeof out->name, name.c_str());
        copy_str(out->vendor, sizeof out->vendor, vendor(name).c_str());
        copy_str(out->runtime_version, sizeof out->runtime_version, RUNTIME);
        copy_str(out->driver_version, sizeof out->driver_version, os_version().c_str());
        return TURBO_OK;
    });
}

/* Fields of turbo_embed_options a run honors: normalize (4), pooling (5)
 * and output_dim (6), every value of each. */
constexpr uint32_t EMBED_HONORED = 0x38;

/* Why the device runs no kernels of this backend, or NULL when it runs
 * them: the linear layers need SIMD-group matrices (Apple7, the M1's
 * family, and later), and the design needs one memory. */
const char *cannot_run(id<MTLDevice> d) {
    if (!d.hasUnifiedMemory)
        return "the metal backend runs on Apple silicon, where the GPU shares the host's memory; this GPU has its own";
    if (![d supportsFamily:MTLGPUFamilyApple7])
        return "the metal backend needs SIMD-group matrices, which Apple GPUs have from the M1's family (Apple7) on";
    return nullptr;
}

/* Embed at every precision, in F32: the one dtype the encoder computes
 * in, so FASTEST is F32 too and says so. As on the CPU, a model stored in
 * F16 or BF16 computes in F32 at EXACT and FASTEST from a converted copy,
 * and its session at MODEL is refused. */
int32_t capability(uint32_t ordinal, uint32_t, uint32_t, uint32_t *status, uint32_t *dtype, uint32_t *options_honored,
                   char *reason, uint32_t reason_len, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        id<MTLDevice> d = device(ordinal);
        if (!d) return not_listed(err, ordinal);
        if (const char *why = cannot_run(d)) {
            *status = TURBO_CAP_UNSUPPORTED;
            *dtype = 0;
            *options_honored = 0;
            copy_str(reason, reason_len, why);
            return TURBO_OK;
        }
        *status = TURBO_CAP_EXPERIMENTAL;
        *dtype = TURBO_DTYPE_F32;
        *options_honored = EMBED_HONORED;
        copy_str(reason, reason_len, "");
        return TURBO_OK;
    });
}

// ---- Contexts and buffers ------------------------------------------------------
//
// A context is a command queue on one device and the kernels, compiled
// for it from their source when the context is made, without fast math.
//
// Placements, all over the one memory: DEVICE is a private Metal buffer,
// with no host address; PINNED and SHARED are shared Metal buffers, one
// address for host and device; HOST is 64-byte aligned host memory the
// backend allocates, with no Metal buffer.

constexpr uint32_t LOG_WARNING = 1;
constexpr uint32_t LOG_DEBUG = 3;

enum Kernel { EMBED_LN, ADD_LN, LINEAR, GELU, ATTENTION, POOL, WIDEN_F16, WIDEN_BF16, KERNELS };
const char *const KERNEL_NAMES[KERNELS] = {"embed_layer_norm", "add_layer_norm", "linear", "bias_gelu",
                                           "attention", "pool", "widen_f16", "widen_bf16"};

struct Context {
    uint32_t ordinal = 0;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> kernels[KERNELS] = {};
    turbo_log_fn log = nullptr;
    void *log_user_data = nullptr;
    std::mutex lock;

    __attribute__((format(printf, 3, 4))) void say(uint32_t level, const char *fmt, ...) {
        if (!log) return;
        char m[256];
        va_list ap;
        va_start(ap, fmt);
        const int n = vsnprintf(m, sizeof m, fmt, ap);
        va_end(ap);
        const size_t len = n < 0 ? 0 : (size_t)n < sizeof m ? (size_t)n : sizeof m - 1;
        log(log_user_data, level, turbo_text{m, len});
    }
};

int32_t compile(Context *c, turbo_error *err) {
    MTLCompileOptions *o = [MTLCompileOptions new];
    if (@available(macOS 15.0, *))
        o.mathMode = MTLMathModeSafe;
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        o.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    NSError *e = nil;
    id<MTLLibrary> lib = [c->device newLibraryWithSource:@(TURBO_METAL_KERNELS) options:o error:&e];
    if (!lib) return refuse(err, TURBO_E_RUNTIME, "compiling kernels.metal: %s", text(e));
    for (int k = 0; k < KERNELS; k++) {
        id<MTLFunction> f = [lib newFunctionWithName:@(KERNEL_NAMES[k])];
        if (!f) return refuse(err, TURBO_E_INTERNAL, "kernels.metal has no kernel %s", KERNEL_NAMES[k]);
        c->kernels[k] = [c->device newComputePipelineStateWithFunction:f error:&e];
        if (!c->kernels[k]) return refuse(err, TURBO_E_RUNTIME, "the %s pipeline: %s", KERNEL_NAMES[k], text(e));
    }
    return TURBO_OK;
}

int32_t context_create(uint32_t ordinal, turbo_log_fn log, void *log_user_data, void **out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        id<MTLDevice> d = device(ordinal);
        if (!d) return not_listed(err, ordinal);
        if (const char *why = cannot_run(d)) return refuse(err, TURBO_E_UNSUPPORTED, "%s", why);
        Context *c = make<Context>();
        c->ordinal = ordinal;
        c->device = d;
        c->log = log;
        c->log_user_data = log_user_data;
        c->queue = [d newCommandQueue];
        int32_t rc = c->queue ? TURBO_OK : refuse(err, TURBO_E_RUNTIME, "newCommandQueue gave none");
        if (rc == TURBO_OK) rc = compile(c, err);
        if (rc != TURBO_OK) {
            delete c;
            return rc;
        }
        c->say(LOG_DEBUG, "metal context on device %u (%s): a command queue, the kernels compiled without fast math",
               ordinal, d.name.UTF8String);
        *out = c;
        return TURBO_OK;
    });
}

void context_release(void *ctx) {
    @autoreleasepool {
        try {
            delete static_cast<Context *>(ctx);
        } catch (...) {
        }
    }
}

struct Buffer {
    Context *ctx = nullptr;
    /* nil for HOST memory. */
    id<MTLBuffer> mtl = nil;
    /* Where the bytes start in mtl. */
    uint64_t offset = 0;
    /* The host address, NULL for DEVICE. */
    void *host = nullptr;
    uint32_t placement = 0;
    uint64_t bytes = 0;
    /* Whether release frees host: HOST memory the backend allocated. */
    bool owned = false;
};

const char *placement_name(uint32_t p) {
    switch (p) {
    case TURBO_PLACE_HOST: return "HOST";
    case TURBO_PLACE_PINNED: return "PINNED";
    case TURBO_PLACE_DEVICE: return "DEVICE";
    default: return "SHARED";
    }
}

const char *handle_name(uint32_t k) {
    switch (k) {
    case TURBO_HANDLE_HOST_PTR: return "TURBO_HANDLE_HOST_PTR";
    case TURBO_HANDLE_CUDA_PTR: return "TURBO_HANDLE_CUDA_PTR";
    case TURBO_HANDLE_CL_MEM: return "TURBO_HANDLE_CL_MEM";
    case TURBO_HANDLE_ZE_USM: return "TURBO_HANDLE_ZE_USM";
    case TURBO_HANDLE_MTL_BUFFER: return "TURBO_HANDLE_MTL_BUFFER";
    default: return "TURBO_HANDLE_DMABUF_FD";
    }
}

int32_t give(Buffer *b, void **out, void **host) {
    *host = b->host;
    *out = b;
    return TURBO_OK;
}

int32_t buffer_alloc(void *ctx, const turbo_buffer_desc *desc, void **out, void **host, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Context *c = static_cast<Context *>(ctx);
        const uint64_t bytes = desc->bytes;
        if (bytes > c->device.maxBufferLength)
            return refuse(err, TURBO_E_OUT_OF_MEMORY, "%llu bytes: device %u's largest buffer is %llu bytes",
                          (unsigned long long)bytes, c->ordinal, (unsigned long long)c->device.maxBufferLength);
        Buffer *b = make<Buffer>();
        b->ctx = c;
        b->placement = desc->placement;
        b->bytes = bytes;
        // Not zeroed: turbo.h promises no contents, and the caller writes them.
        if (desc->placement == TURBO_PLACE_HOST) {
            b->host = host_malloc(bytes);
            b->owned = true;
        } else {
            const MTLResourceOptions o = desc->placement == TURBO_PLACE_DEVICE ? MTLResourceStorageModePrivate
                                                                               : MTLResourceStorageModeShared;
            b->mtl = new_buffer(c->device, bytes, o);
            if (b->mtl && desc->placement != TURBO_PLACE_DEVICE) b->host = b->mtl.contents;
        }
        if (!b->host && !b->mtl) {
            delete b;
            return refuse(err, TURBO_E_OUT_OF_MEMORY, "%llu bytes of %s memory", (unsigned long long)bytes,
                          placement_name(desc->placement));
        }
        return give(b, out, host);
    });
}

/* The caller's memory, wrapped: nothing is copied, nothing is freed. A
 * TURBO_HANDLE_MTL_BUFFER names a Metal buffer on this context's device,
 * private for DEVICE and shared otherwise; a TURBO_HANDLE_HOST_PTR names
 * host memory, which for PINNED and SHARED must be whole pages Metal can
 * map. */
int32_t buffer_import(void *ctx, const turbo_buffer_desc *desc, const turbo_native_handle *h, void **out, void **host,
                      turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Context *c = static_cast<Context *>(ctx);
        const uint32_t pl = desc->placement;
        if (h->kind != TURBO_HANDLE_MTL_BUFFER && h->kind != TURBO_HANDLE_HOST_PTR)
            return refuse(err, TURBO_E_UNSUPPORTED,
                          "kind: %s: the metal backend imports TURBO_HANDLE_MTL_BUFFER and TURBO_HANDLE_HOST_PTR",
                          handle_name(h->kind));
        if (h->handle == 0) return refuse(err, TURBO_E_INVALID_ARGUMENT, "handle: NULL");
        Buffer b;
        b.ctx = c;
        b.placement = pl;
        b.bytes = desc->bytes;
        if (h->kind == TURBO_HANDLE_MTL_BUFFER) {
            id<MTLBuffer> m = (__bridge id<MTLBuffer>)(void *)(uintptr_t)h->handle;
            if (h->aux != (uint64_t)(uintptr_t)(__bridge void *)c->device)
                return refuse(err, TURBO_E_INVALID_ARGUMENT, "aux: not the MTLDevice of this context (device %u)",
                              c->ordinal);
            if (m.device != c->device)
                return refuse(err, TURBO_E_INVALID_ARGUMENT, "handle: a buffer of another device than device %u",
                              c->ordinal);
            if (h->offset > m.length || desc->bytes > m.length - h->offset)
                return refuse(err, TURBO_E_INVALID_ARGUMENT, "offset %llu + %llu bytes is past the buffer's %lu",
                              (unsigned long long)h->offset, (unsigned long long)desc->bytes,
                              (unsigned long)m.length);
            const bool priv = m.storageMode == MTLStorageModePrivate;
            if (pl == TURBO_PLACE_HOST)
                return refuse(err, TURBO_E_INVALID_ARGUMENT,
                              "placement: HOST: a Metal buffer is PINNED, SHARED or DEVICE memory");
            if ((pl == TURBO_PLACE_DEVICE) != priv)
                return refuse(err, TURBO_E_INVALID_ARGUMENT,
                              "placement: %s: the buffer's storage is %s; DEVICE is a private buffer and PINNED and "
                              "SHARED a shared one",
                              placement_name(pl), priv ? "private" : "not private");
            b.mtl = m;
            b.offset = h->offset;
            b.host = priv ? nullptr : static_cast<char *>(m.contents) + h->offset;
        } else {
            const uint64_t end = h->handle + h->offset;
            if (end < h->handle || end + desc->bytes < end)
                return refuse(err, TURBO_E_INVALID_ARGUMENT,
                              "handle %#llx + offset %llu + %llu bytes is past the address space",
                              (unsigned long long)h->handle, (unsigned long long)h->offset,
                              (unsigned long long)desc->bytes);
            void *p = reinterpret_cast<void *>((uintptr_t)end);
            if (pl == TURBO_PLACE_DEVICE)
                return refuse(err, TURBO_E_INVALID_ARGUMENT,
                              "placement: DEVICE: a TURBO_HANDLE_HOST_PTR is host memory; import a private Metal "
                              "buffer as TURBO_HANDLE_MTL_BUFFER");
            if (pl != TURBO_PLACE_HOST) {
                b.mtl = wrap(c->device, p, desc->bytes);
                if (!b.mtl)
                    return refuse(err, TURBO_E_INVALID_ARGUMENT,
                                  "placement: %s: Metal maps whole pages only, and %p + %llu bytes is not on %d-byte "
                                  "boundaries (import it as HOST)",
                                  placement_name(pl), p, (unsigned long long)desc->bytes, getpagesize());
            }
            b.host = p;
        }
        Buffer *nb = make<Buffer>(std::move(b));
        return give(nb, out, host);
    });
}

void buffer_release(void *buf) {
    @autoreleasepool {
        try {
            Buffer *b = static_cast<Buffer *>(buf);
            if (b->owned) free(b->host);
            delete b;
        } catch (...) {
        }
    }
}

/* The buffer's own memory, no copy: its Metal buffer, and for any but
 * DEVICE its host address. */
int32_t buffer_export(void *buf, uint32_t kind, turbo_native_handle *out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        const Buffer *b = static_cast<const Buffer *>(buf);
        if (kind == TURBO_HANDLE_MTL_BUFFER) {
            if (!b->mtl)
                return refuse(err, TURBO_E_UNSUPPORTED,
                              "kind: TURBO_HANDLE_MTL_BUFFER: a HOST buffer is not a Metal buffer; export it as "
                              "TURBO_HANDLE_HOST_PTR");
            out->kind = kind;
            out->handle = (uint64_t)(uintptr_t)(__bridge void *)b->mtl;
            out->aux = (uint64_t)(uintptr_t)(__bridge void *)b->ctx->device;
            out->offset = b->offset;
            return TURBO_OK;
        }
        if (kind == TURBO_HANDLE_HOST_PTR) {
            if (!b->host)
                return refuse(err, TURBO_E_UNSUPPORTED,
                              "kind: TURBO_HANDLE_HOST_PTR: a DEVICE buffer has no host address; export it as "
                              "TURBO_HANDLE_MTL_BUFFER");
            out->kind = kind;
            out->handle = (uint64_t)(uintptr_t)b->host;
            out->aux = 0;
            out->offset = 0;
            return TURBO_OK;
        }
        return refuse(err, TURBO_E_UNSUPPORTED,
                      "kind: %s: the metal backend exports TURBO_HANDLE_MTL_BUFFER and TURBO_HANDLE_HOST_PTR",
                      handle_name(kind));
    });
}

/* A DEVICE buffer's first bytes to the caller's host memory: a blit into a
 * shared buffer, then a copy out of it. */
int32_t buffer_read(void *buf, void *dst, uint64_t bytes, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        const Buffer *b = static_cast<const Buffer *>(buf);
        if (bytes > b->bytes)
            return refuse(err, TURBO_E_INVALID_ARGUMENT, "%llu bytes from a buffer of %llu", (unsigned long long)bytes,
                          (unsigned long long)b->bytes);
        if (b->host) {
            memcpy(dst, b->host, bytes);
            return TURBO_OK;
        }
        Context *c = b->ctx;
        id<MTLBuffer> staging = new_buffer(c->device, bytes, MTLResourceStorageModeShared);
        if (!staging) return refuse(err, TURBO_E_OUT_OF_MEMORY, "%llu bytes to read through", (unsigned long long)bytes);
        std::lock_guard<std::mutex> g(c->lock);
        id<MTLCommandBuffer> cb = [c->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:b->mtl sourceOffset:b->offset toBuffer:staging destinationOffset:0 size:bytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted)
            return refuse(err, TURBO_E_RUNTIME, "reading the buffer: %s", text(cb.error));
        memcpy(dst, staging.contents, bytes);
        return TURBO_OK;
    });
}

// ---- Models --------------------------------------------------------------------
//
// The device reads the weights where the core holds them: the pages under
// every tensor are wrapped as one shared Metal buffer, without a copy, and
// each tensor is an offset into it. Where Metal will not wrap them, they
// are copied into a shared buffer once, and the log says so. A model
// stored in F16 or BF16 gets a second buffer, its weights widened to F32
// on the device, made by the first session that computes in F32 and
// shared by every later one, as turbo.h's precision rules say.

/* TURBO_BERT_* in turbo_backend.h. */
enum : int {
    WORD = TURBO_BERT_WORD_EMBEDDINGS,
    POSITION = TURBO_BERT_POSITION_EMBEDDINGS,
    TOKEN_TYPE = TURBO_BERT_TOKEN_TYPE_EMBEDDINGS,
    EMB_LN_W = TURBO_BERT_EMBEDDINGS_LN_WEIGHT,
    EMB_LN_B = TURBO_BERT_EMBEDDINGS_LN_BIAS,
};

/* A tensor: its Metal buffer and where in it the tensor starts. */
struct Ref {
    id<MTLBuffer> buf = nil;
    uint64_t at = 0;
};

struct Model {
    Context *ctx = nullptr;
    turbo_backend_model desc{};
    /* The weights as stored, and whether that is the core's own memory. */
    id<MTLBuffer> stored = nil;
    bool in_place = false;
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> counts;
    /* Every tensor in F32: into stored for an F32 model, else into widened
     * once a session made it. */
    std::mutex widen_lock;
    id<MTLBuffer> widened = nil;
    std::vector<Ref> f32;
};

const char *dtype_name(uint32_t d) { return d == TURBO_DTYPE_F32 ? "F32" : d == TURBO_DTYPE_F16 ? "F16" : "BF16"; }

int32_t model_load(void *ctx, const turbo_backend_model *desc, void **out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Context *c = static_cast<Context *>(ctx);
        if (desc->family != TURBO_FAMILY_BERT)
            return refuse(err, TURBO_E_UNSUPPORTED, "family %u: the metal backend holds BERT encoders", desc->family);
        if (desc->heads == 0 || desc->hidden % desc->heads != 0)
            return refuse(err, TURBO_E_UNSUPPORTED, "hidden %u is not a multiple of heads %u", desc->hidden,
                          desc->heads);
        const size_t elem = desc->dtype == TURBO_DTYPE_F32 ? 4 : 2;
        // The pages the tensors lie on, first to last.
        const uintptr_t page = (uintptr_t)getpagesize();
        uintptr_t lo = UINTPTR_MAX, hi = 0;
        for (uint32_t i = 0; i < desc->tensor_count; i++) {
            const uintptr_t p = (uintptr_t)desc->tensors[i].data;
            lo = std::min(lo, p);
            hi = std::max(hi, p + (uintptr_t)desc->tensors[i].bytes);
        }
        lo = lo / page * page;
        hi = (hi + page - 1) / page * page;

        Model *m = make<Model>();
        m->ctx = c;
        m->desc = *desc;
        m->desc.tensors = nullptr;
        m->stored = wrap(c->device, reinterpret_cast<void *>(lo), hi - lo);
        m->in_place = m->stored != nil;
        if (m->in_place) {
            for (uint32_t i = 0; i < desc->tensor_count; i++)
                m->offsets.push_back((uintptr_t)desc->tensors[i].data - lo);
        } else {
            size_t total = 0;
            for (uint32_t i = 0; i < desc->tensor_count; i++) {
                m->offsets.push_back(total);
                total += round_up(desc->tensors[i].bytes, 256);
            }
            m->stored = new_buffer(c->device, total, MTLResourceStorageModeShared);
            if (!m->stored) {
                delete m;
                return refuse(err, TURBO_E_OUT_OF_MEMORY, "%zu bytes for the weights", total);
            }
            char *base = static_cast<char *>(m->stored.contents);
            for (uint32_t i = 0; i < desc->tensor_count; i++)
                memcpy(base + m->offsets[i], desc->tensors[i].data, desc->tensors[i].bytes);
            c->say(LOG_WARNING, "metal device %u: the weights are not on whole pages Metal can map; copied %zu bytes",
                   c->ordinal, total);
        }
        for (uint32_t i = 0; i < desc->tensor_count; i++) m->counts.push_back(desc->tensors[i].bytes / elem);
        if (desc->dtype == TURBO_DTYPE_F32)
            for (uint32_t i = 0; i < desc->tensor_count; i++) m->f32.push_back(Ref{m->stored, m->offsets[i]});
        c->say(LOG_DEBUG, "metal device %u: a BERT of %u layers, %s weights %s", c->ordinal, desc->layers,
               dtype_name(desc->dtype), m->in_place ? "read in place" : "copied");
        *out = m;
        return TURBO_OK;
    });
}

void model_release(void *model) {
    @autoreleasepool {
        try {
            delete static_cast<Model *>(model);
        } catch (...) {
        }
    }
}

/* Wait for a command buffer, and say what went wrong if it failed. */
int32_t finish(id<MTLCommandBuffer> cb, turbo_error *err, const char *what) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        NSError *e = cb.error;
        const bool oom = e && e.code == MTLCommandBufferErrorOutOfMemory;
        return refuse(err, oom ? TURBO_E_OUT_OF_MEMORY : TURBO_E_RUNTIME, "%s: %s", what, text(e));
    }
    return TURBO_OK;
}

/* The model's weights in F32, made on first need. */
int32_t f32_weights(Model *m, turbo_error *err) {
    std::lock_guard<std::mutex> g(m->widen_lock);
    if (!m->f32.empty()) return TURBO_OK;
    Context *c = m->ctx;
    size_t total = 0;
    std::vector<uint64_t> at;
    for (uint64_t n : m->counts) {
        at.push_back(total);
        total += round_up(n * 4, 256);
    }
    id<MTLBuffer> wide = new_buffer(c->device, total, MTLResourceStorageModePrivate);
    if (!wide) return refuse(err, TURBO_E_OUT_OF_MEMORY, "%zu bytes for the weights in F32", total);
    {
        std::lock_guard<std::mutex> cg(c->lock);
        id<MTLCommandBuffer> cb = [c->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:c->kernels[m->desc.dtype == TURBO_DTYPE_F16 ? WIDEN_F16 : WIDEN_BF16]];
        for (size_t i = 0; i < m->counts.size(); i++) {
            const uint32_t n = (uint32_t)m->counts[i];
            [enc setBuffer:m->stored offset:m->offsets[i] atIndex:0];
            [enc setBuffer:wide offset:at[i] atIndex:1];
            [enc setBytes:&n length:sizeof n atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
        TRY(finish(cb, err, "widening the weights"));
    }
    m->widened = wide;
    for (size_t i = 0; i < m->counts.size(); i++) m->f32.push_back(Ref{wide, at[i]});
    return TURBO_OK;
}

// ---- Sessions ------------------------------------------------------------------
//
// A session is private scratch for its largest batch, shared memory for
// the rows, and the shared buffer its vectors are written to, all
// allocated here; a run allocates nothing. The rows are computed over the
// written batch's full [batch, seq] grid: attention skips masked keys and
// no pooling reads a masked token, so the padding a row carries changes no
// output, as on the CPU, which leaves it out.
//
// embed_write copies the rows into the session's shared memory, where the
// GPU reads them: with one memory nothing crosses to a device, so
// h2d_bytes and d2h_bytes are 0, as on the CPU. The vectors are left in
// the session's shared buffer, which the host reads where they are.

/* Kernel parameter blocks, as kernels.metal declares them. */
struct RowParams {
    uint32_t seq, hidden;
    float eps;
    uint32_t has_types;
};
struct LinearParams {
    uint32_t m, n, k;
};
struct GeluParams {
    uint32_t n, width;
};
struct AttentionParams {
    uint32_t seq, hidden, head_dim;
    float scale;
};
struct PoolParams {
    uint32_t seq, hidden, output_dim, pooling, l2;
};

constexpr uint32_t BLOCK = 128;
constexpr uint32_t TILE = 32;

size_t attention_bytes(uint32_t seq, uint32_t head_dim) {
    const uint32_t part = head_dim <= BLOCK ? (BLOCK / head_dim) * head_dim : 0;
    // Threadgroup memory is given in multiples of 16 bytes.
    return round_up(sizeof(float) * ((size_t)head_dim + seq + part), 16);
}

struct Session {
    Model *model = nullptr;
    Context *ctx = nullptr;
    uint32_t max_batch = 0, max_seq = 0;
    /* [3, max_batch * max_seq] int32: ids, mask, types. */
    id<MTLBuffer> rows = nil;
    id<MTLBuffer> scratch = nil;
    uint64_t x = 0, q = 0, k = 0, v = 0, att = 0, tmp = 0, ffn = 0;
    /* [max_batch, hidden] F32, handed out as the output. */
    Buffer output;
    bool written = false;
    bool has_types = false;
    uint32_t batch = 0, seq = 0, pooling = 0, normalize = 0, output_dim = 0;
};

int32_t session_create(void *model, uint32_t task, uint32_t max_batch, uint32_t max_seq, uint32_t precision,
                       uint32_t *compute_dtype, void **out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Model *m = static_cast<Model *>(model);
        Context *c = m->ctx;
        const turbo_backend_model &d = m->desc;
        if (task != TURBO_TASK_EMBED)
            return refuse(err, TURBO_E_UNSUPPORTED_TASK, "task %u: the metal backend runs embed", task);
        if (precision == TURBO_PRECISION_MODEL && d.dtype != TURBO_DTYPE_F32)
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 3,
                                "precision: MODEL computes in the weights' %s, and the metal backend computes in F32 "
                                "only; EXACT and FASTEST compute this model in F32",
                                dtype_name(d.dtype));
        if (max_seq > d.max_positions)
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 2, "max_seq %u is over the model's %u positions",
                                max_seq, d.max_positions);
        const size_t shared = attention_bytes(max_seq, d.hidden / d.heads);
        const size_t most = c->device.maxThreadgroupMemoryLength - c->kernels[ATTENTION].staticThreadgroupMemoryLength;
        if (shared > most)
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 2,
                                "max_seq %u: attention needs %zu bytes of threadgroup memory, and device %u gives it "
                                "%zu",
                                max_seq, shared, c->ordinal, most);
        const size_t tokens = (size_t)max_batch * max_seq;
        const size_t widest = std::max(d.intermediate, d.hidden);
        if (tokens * widest * 4 > c->device.maxBufferLength || tokens > UINT32_MAX / widest)
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 1,
                                "%u rows of %u tokens is more than one of device %u's buffers holds", max_batch,
                                max_seq, c->ordinal);
        TRY(f32_weights(m, err));

        const size_t wide = round_up(tokens * d.hidden * 4, 256);
        const size_t ffn = round_up(tokens * d.intermediate * 4, 256);
        Session *s = make<Session>();
        s->model = m;
        s->ctx = c;
        s->max_batch = max_batch;
        s->max_seq = max_seq;
        s->rows = new_buffer(c->device, 3 * tokens * 4, MTLResourceStorageModeShared);
        s->scratch = new_buffer(c->device, 6 * wide + ffn, MTLResourceStorageModePrivate);
        const uint64_t vectors = (uint64_t)max_batch * d.hidden * 4;
        id<MTLBuffer> output = new_buffer(c->device, vectors, MTLResourceStorageModeShared);
        if (!s->rows || !s->scratch || !output) {
            delete s;
            return refuse(err, TURBO_E_OUT_OF_MEMORY, "%zu bytes for the session", 12 * tokens + 6 * wide + ffn);
        }
        s->x = 0;
        s->q = wide;
        s->k = 2 * wide;
        s->v = 3 * wide;
        s->att = 4 * wide;
        s->tmp = 5 * wide;
        s->ffn = 6 * wide;
        s->output.ctx = c;
        s->output.mtl = output;
        s->output.host = output.contents;
        s->output.placement = TURBO_PLACE_SHARED;
        s->output.bytes = vectors;
        s->output.owned = false;
        *compute_dtype = TURBO_DTYPE_F32;
        *out = s;
        return TURBO_OK;
    });
}

void session_release(void *session) {
    @autoreleasepool {
        try {
            delete static_cast<Session *>(session);
        } catch (...) {
        }
    }
}

int32_t embed_write(void *session, const turbo_backend_embed_rows *r, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Session &s = *static_cast<Session *>(session);
        const size_t tokens = (size_t)s.max_batch * s.max_seq;
        int32_t *dst = static_cast<int32_t *>(s.rows.contents);
        const int32_t *src[3] = {r->ids, r->mask, r->types};
        for (int a = 0; a < 3; a++) {
            if (!src[a]) continue;
            for (uint32_t row = 0; row < r->batch; row++)
                memcpy(dst + a * tokens + (size_t)row * r->seq, src[a] + (size_t)row * r->row_stride,
                       (size_t)r->seq * 4);
        }
        s.has_types = r->types != nullptr;
        s.batch = r->batch;
        s.seq = r->seq;
        s.pooling = r->pooling;
        s.normalize = r->normalize;
        s.output_dim = r->output_dim;
        s.written = true;
        return TURBO_OK;
    });
}

/* The encoder over the written rows, into one compute pass. Its
 * dispatches run in order, each seeing what the one before wrote. */
void encode(Session &s, id<MTLComputeCommandEncoder> enc) {
    const turbo_backend_model &d = s.model->desc;
    const std::vector<Ref> &w = s.model->f32;
    Context *c = s.ctx;
    const uint32_t batch = s.batch, seq = s.seq, tokens = batch * seq;
    const uint32_t h = d.hidden, inter = d.intermediate, hd = d.hidden / d.heads;
    const size_t max_tokens = (size_t)s.max_batch * s.max_seq;
    id<MTLBuffer> sc = s.scratch;
    auto bind = [&](const Ref &r, NSUInteger i) { [enc setBuffer:r.buf offset:r.at atIndex:i]; };
    auto layer = [&](uint32_t l, int r) -> const Ref & {
        return w[TURBO_BERT_EMBEDDING_TENSORS + l * TURBO_BERT_LAYER_TENSORS + r];
    };
    auto rows = [&](Kernel k) {
        [enc setComputePipelineState:c->kernels[k]];
    };
    auto per_token = [&]() {
        [enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(BLOCK, 1, 1)];
    };
    auto linear = [&](uint64_t x, uint32_t n_in, const Ref &wt, uint32_t n_out, uint64_t y) {
        const LinearParams p{tokens, n_out, n_in};
        [enc setComputePipelineState:c->kernels[LINEAR]];
        [enc setBuffer:sc offset:x atIndex:0];
        bind(wt, 1);
        [enc setBuffer:sc offset:y atIndex:2];
        [enc setBytes:&p length:sizeof p atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((n_out + TILE - 1) / TILE, (tokens + TILE - 1) / TILE, 1)
            threadsPerThreadgroup:MTLSizeMake(BLOCK, 1, 1)];
    };
    const RowParams rp{seq, h, (float)d.layer_norm_eps, s.has_types ? 1u : 0u};
    auto add_ln = [&](uint64_t y, const Ref &bias, const Ref &ln_w, const Ref &ln_b) {
        rows(ADD_LN);
        [enc setBuffer:sc offset:s.x atIndex:0];
        [enc setBuffer:sc offset:y atIndex:1];
        bind(bias, 2);
        bind(ln_w, 3);
        bind(ln_b, 4);
        [enc setBytes:&rp length:sizeof rp atIndex:5];
        per_token();
    };

    rows(EMBED_LN);
    [enc setBuffer:s.rows offset:0 atIndex:0];
    [enc setBuffer:s.rows offset:2 * max_tokens * 4 atIndex:1];
    bind(w[WORD], 2);
    bind(w[POSITION], 3);
    bind(w[TOKEN_TYPE], 4);
    bind(w[EMB_LN_W], 5);
    bind(w[EMB_LN_B], 6);
    [enc setBuffer:sc offset:s.x atIndex:7];
    [enc setBytes:&rp length:sizeof rp atIndex:8];
    per_token();

    const AttentionParams ap{seq, h, hd, 1.0f / sqrtf((float)hd)};
    const GeluParams gp{tokens * inter, inter};
    for (uint32_t l = 0; l < d.layers; l++) {
        linear(s.x, h, layer(l, TURBO_BERT_Q_WEIGHT), h, s.q);
        linear(s.x, h, layer(l, TURBO_BERT_K_WEIGHT), h, s.k);
        linear(s.x, h, layer(l, TURBO_BERT_V_WEIGHT), h, s.v);

        [enc setComputePipelineState:c->kernels[ATTENTION]];
        [enc setBuffer:sc offset:s.q atIndex:0];
        [enc setBuffer:sc offset:s.k atIndex:1];
        [enc setBuffer:sc offset:s.v atIndex:2];
        bind(layer(l, TURBO_BERT_Q_BIAS), 3);
        bind(layer(l, TURBO_BERT_K_BIAS), 4);
        bind(layer(l, TURBO_BERT_V_BIAS), 5);
        [enc setBuffer:s.rows offset:max_tokens * 4 atIndex:6];
        [enc setBuffer:sc offset:s.att atIndex:7];
        [enc setBytes:&ap length:sizeof ap atIndex:8];
        [enc setThreadgroupMemoryLength:attention_bytes(seq, hd) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(seq, d.heads, batch) threadsPerThreadgroup:MTLSizeMake(BLOCK, 1, 1)];

        linear(s.att, h, layer(l, TURBO_BERT_ATTN_OUT_WEIGHT), h, s.tmp);
        add_ln(s.tmp, layer(l, TURBO_BERT_ATTN_OUT_BIAS), layer(l, TURBO_BERT_ATTN_LN_WEIGHT),
               layer(l, TURBO_BERT_ATTN_LN_BIAS));
        linear(s.x, h, layer(l, TURBO_BERT_FFN_IN_WEIGHT), inter, s.ffn);

        [enc setComputePipelineState:c->kernels[GELU]];
        [enc setBuffer:sc offset:s.ffn atIndex:0];
        bind(layer(l, TURBO_BERT_FFN_IN_BIAS), 1);
        [enc setBytes:&gp length:sizeof gp atIndex:2];
        [enc dispatchThreads:MTLSizeMake(gp.n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        linear(s.ffn, inter, layer(l, TURBO_BERT_FFN_OUT_WEIGHT), h, s.tmp);
        add_ln(s.tmp, layer(l, TURBO_BERT_FFN_OUT_BIAS), layer(l, TURBO_BERT_FFN_LN_WEIGHT),
               layer(l, TURBO_BERT_FFN_LN_BIAS));
    }

    const PoolParams pp{seq, h, s.output_dim, s.pooling, s.normalize == TURBO_NORMALIZE_L2 ? 1u : 0u};
    rows(POOL);
    [enc setBuffer:sc offset:s.x atIndex:0];
    [enc setBuffer:s.rows offset:max_tokens * 4 atIndex:1];
    [enc setBuffer:s.output.mtl offset:0 atIndex:2];
    [enc setBytes:&pp length:sizeof pp atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(BLOCK, 1, 1)];
}

int32_t session_run(void *session, turbo_backend_run *out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Session &s = *static_cast<Session *>(session);
        if (!s.written)
            return refuse(err, TURBO_E_INVALID_STATE, "the metal session has no rows written since its last run");
        s.written = false;
        const uint64_t host0 = host_here, device0 = device_here;
        {
            std::lock_guard<std::mutex> g(s.ctx->lock);
            id<MTLCommandBuffer> cb = [s.ctx->queue commandBufferWithUnretainedReferences];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode(s, enc);
            [enc endEncoding];
            TRY(finish(cb, err, "the run"));
        }
        out->placement = TURBO_PLACE_SHARED;
        out->output = &s.output;
        out->host = s.output.host;
        out->h2d_bytes = 0;
        out->d2h_bytes = 0;
        out->host_allocs = host_here - host0;
        out->device_allocs = device_here - device0;
        uint32_t *st = out->stage;
        // The rows were copied into the session's shared memory by
        // embed_write, on the host.
        st[TURBO_EMBED_STAGE_UPLOAD] = TURBO_STAGE_HOST;
        st[TURBO_EMBED_STAGE_LOOKUP] = TURBO_STAGE_DEVICE;
        st[TURBO_EMBED_STAGE_ENCODE] = TURBO_STAGE_DEVICE;
        st[TURBO_EMBED_STAGE_POOL] = TURBO_STAGE_DEVICE;
        // Normalization is the last step of the pooling kernel.
        st[TURBO_EMBED_STAGE_NORMALIZE] = s.normalize == TURBO_NORMALIZE_L2 ? TURBO_STAGE_FUSED : TURBO_STAGE_UNUSED;
        st[TURBO_EMBED_STAGE_DOWNLOAD] = TURBO_STAGE_UNUSED;
        return TURBO_OK;
    });
}

} // namespace

extern "C" {

extern const turbo_backend turbo_metal_backend;

const turbo_backend turbo_metal_backend = {
    sizeof(turbo_backend),
    0,
    "metal",
    RUNTIME,
    device_count,
    device_info,
    capability,
    context_create,
    context_release,
    buffer_alloc,
    buffer_import,
    buffer_release,
    buffer_export,
    model_load,
    model_release,
    session_create,
    session_release,
    embed_write,
    session_run,
    buffer_read,
};

/* Every allocation this backend has made in the process, host and device,
 * for the tests that hold a warm run to none. */
void turbo_metal_allocations(uint64_t *host, uint64_t *device) {
    *host = host_total.load(std::memory_order_relaxed);
    *device = device_total.load(std::memory_order_relaxed);
}

/* Whether the model reads its weights where the core holds them. */
int turbo_metal_in_place(void *model) { return static_cast<Model *>(model)->in_place ? 1 : 0; }

/* The F32 copy of an F16 or BF16 model's weights, once a session made it;
 * NULL before, and for an F32 model, which computes from its own. */
const void *turbo_metal_widened(void *model) {
    try {
        Model *m = static_cast<Model *>(model);
        std::lock_guard<std::mutex> g(m->widen_lock);
        return (__bridge const void *)m->widened;
    } catch (...) {
        return nullptr;
    }
}

} // extern "C"
