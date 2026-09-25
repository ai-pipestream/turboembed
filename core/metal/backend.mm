// SPDX-License-Identifier: Apache-2.0
//
// The Metal backend: Apple GPUs through Metal, behind
// include/turbo/turbo_backend.h. It lists the devices Metal reports, keeps
// the compiled kernels per device, reads a model's weights where the core
// holds them, and runs embed sessions with the BERT encoder in
// kernels.metal, each session on a command queue of its own.
//
// It runs on Apple silicon, where the GPU and the host share one memory:
// the weights, the rows and the vectors are memory both can address, and
// nothing is copied to reach the device. A GPU with memory of its own is
// listed but runs nothing.
//
// Every function here is called from any thread. A context's own queue,
// for widening weights and reading DEVICE buffers, is used under the
// context's lock, held until the commands complete. A session's queue is
// its own, and a session has one owner at a time.

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
id<MTLBuffer> wrap(id<MTLDevice> d, void *p, size_t n, MTLResourceOptions extra = 0) {
    const size_t page = (size_t)getpagesize();
    if ((uintptr_t)p % page != 0 || n == 0 || n % page != 0) return nil;
    id<MTLBuffer> b = [d newBufferWithBytesNoCopy:p
                                           length:n
                                          options:MTLResourceStorageModeShared | extra
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
    if (@available(macOS 14.0, *)) {
    } else {
        return "the metal backend's kernels need Metal 3.1, which macOS has from 14 on";
    }
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

enum Kernel { EMBED_LN, ADD_LN, GEMM, ATTENTION, ATTENTION_NARROW, POOL, WIDEN_F16, WIDEN_BF16, KERNELS };
const char *const KERNEL_NAMES[KERNELS] = {"embed_layer_norm", "add_layer_norm", "gemm", "attention",
                                           "attention_narrow", "pool", "widen_f16", "widen_bf16"};

/* The kernels compiled for one device. */
struct Kernels {
    id<MTLComputePipelineState> k[KERNELS] = {};
};

struct Context {
    uint32_t ordinal = 0;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    const Kernels *kernels = nullptr;
    /* A shared buffer buffer_read copies DEVICE memory through, grown to
     * the largest read so far; used under lock. */
    id<MTLBuffer> staging = nil;
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

/* Compile kernels.metal for a device into k, without fast math. */
int32_t compile(id<MTLDevice> d, Kernels &k, turbo_error *err) {
    MTLCompileOptions *o = [MTLCompileOptions new];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *))
        o.mathMode = MTLMathModeSafe;
    else
#endif
    {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        o.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    NSError *e = nil;
    id<MTLLibrary> lib = [d newLibraryWithSource:@(TURBO_METAL_KERNELS) options:o error:&e];
    if (!lib) return refuse(err, TURBO_E_RUNTIME, "compiling kernels.metal: %s", text(e));
    for (int i = 0; i < KERNELS; i++) {
        id<MTLFunction> f = [lib newFunctionWithName:@(KERNEL_NAMES[i])];
        if (!f) return refuse(err, TURBO_E_INTERNAL, "kernels.metal has no kernel %s", KERNEL_NAMES[i]);
        k.k[i] = [d newComputePipelineStateWithFunction:f error:&e];
        if (!k.k[i]) return refuse(err, TURBO_E_RUNTIME, "the %s pipeline: %s", KERNEL_NAMES[i], text(e));
    }
    return TURBO_OK;
}

/* The kernels for a listed device, compiled by the first context made on
 * it and held for the life of the process, as the device list is. A
 * failed compile is tried again by the next context. */
int32_t kernels_for(uint32_t ordinal, const Kernels **out, turbo_error *err) {
    // Never destroyed, so a context made while the process exits finds
    // them whole.
    struct Cache {
        std::mutex lock;
        std::vector<Kernels *> compiled = std::vector<Kernels *>(devices().count, nullptr);
    };
    static Cache &cache = *new Cache;
    std::lock_guard<std::mutex> g(cache.lock);
    std::vector<Kernels *> &compiled = cache.compiled;
    if (!compiled[ordinal]) {
        Kernels *k = make<Kernels>();
        const int32_t rc = compile(device(ordinal), *k, err);
        if (rc != TURBO_OK) {
            delete k;
            return rc;
        }
        compiled[ordinal] = k;
    }
    *out = compiled[ordinal];
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
        if (rc == TURBO_OK) rc = kernels_for(ordinal, &c->kernels, err);
        // kernels.metal's reductions and matrices take SIMD groups of 32.
        if (rc == TURBO_OK && c->kernels->k[GEMM].threadExecutionWidth != 32)
            rc = refuse(err, TURBO_E_UNSUPPORTED, "device %u runs SIMD groups of %lu threads; the kernels need 32",
                        ordinal, (unsigned long)c->kernels->k[GEMM].threadExecutionWidth);
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
        if (desc->placement != TURBO_PLACE_HOST && bytes > c->device.maxBufferLength)
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
int32_t finish(id<MTLCommandBuffer> cb, turbo_error *err, const char *what);

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
        std::lock_guard<std::mutex> g(c->lock);
        if (!c->staging || c->staging.length < bytes) {
            c->staging = nil;
            c->staging = new_buffer(c->device, bytes, MTLResourceStorageModeShared);
            if (!c->staging)
                return refuse(err, TURBO_E_OUT_OF_MEMORY, "%llu bytes to read through", (unsigned long long)bytes);
        }
        id<MTLBuffer> staging = c->staging;
        id<MTLCommandBuffer> cb = [c->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:b->mtl sourceOffset:b->offset toBuffer:staging destinationOffset:0 size:bytes];
        [blit endEncoding];
        TRY(finish(cb, err, "reading the buffer"));
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

/* Weights are written once, before any session reads them, then only
 * read, by any number of sessions at once: Metal need not track them,
 * and tracking them would order sessions that share a model one after
 * another. */
constexpr MTLResourceOptions WEIGHTS = MTLResourceHazardTrackingModeUntracked;

/* A tensor: its Metal buffer and where in it the tensor starts. */
struct Ref {
    id<MTLBuffer> buf = nil;
    uint64_t at = 0;
};

struct Model {
    Context *ctx = nullptr;
    turbo_backend_model desc{};
    /* Each tensor as stored, and whether that is the core's own memory. */
    std::vector<Ref> stored;
    bool in_place = false;
    std::vector<uint64_t> counts;
    /* Every tensor in F32: stored for an F32 model, else into widened once
     * a session made it. */
    std::mutex widen_lock;
    id<MTLBuffer> widened = nil;
    std::vector<Ref> f32;
};

const char *dtype_name(uint32_t d) { return d == TURBO_DTYPE_F32 ? "F32" : d == TURBO_DTYPE_F16 ? "F16" : "BF16"; }

/* The tensors where the core holds them, in runs: tensors that share a
 * page lie in one allocation (a weights file's, packed), and each run is
 * mapped as one shared buffer over the pages it lies on. A run Metal will
 * not map is copied into a shared buffer of its own. Returns the bytes
 * copied, or -1 when the host could not give them. */
int64_t map_weights(Model *m, const turbo_backend_model *desc) {
    const uintptr_t page = (uintptr_t)getpagesize();
    const uint32_t n = desc->tensor_count;
    std::vector<uint32_t> order(n);
    for (uint32_t i = 0; i < n; i++) order[i] = i;
    auto start = [&](uint32_t i) { return (uintptr_t)desc->tensors[i].data; };
    auto stop = [&](uint32_t i) { return start(i) + (uintptr_t)desc->tensors[i].bytes; };
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return start(a) < start(b); });
    m->stored.assign(n, Ref{});
    int64_t copied = 0;
    for (uint32_t k = 0; k < n;) {
        const uintptr_t lo = start(order[k]) / page * page;
        uintptr_t hi = stop(order[k]);
        uint32_t e = k + 1;
        while (e < n && start(order[e]) < (hi + page - 1) / page * page) hi = std::max(hi, stop(order[e++]));
        hi = (hi + page - 1) / page * page;
        if (id<MTLBuffer> b = wrap(m->ctx->device, reinterpret_cast<void *>(lo), hi - lo, WEIGHTS)) {
            for (; k < e; k++) m->stored[order[k]] = Ref{b, start(order[k]) - lo};
            continue;
        }
        size_t total = 0;
        for (uint32_t i = k; i < e; i++) total += round_up(desc->tensors[order[i]].bytes, 256);
        id<MTLBuffer> copy = new_buffer(m->ctx->device, total, MTLResourceStorageModeShared | WEIGHTS);
        if (!copy) return -1;
        char *base = static_cast<char *>(copy.contents);
        for (uint64_t at = 0; k < e; k++) {
            const turbo_backend_tensor &t = desc->tensors[order[k]];
            memcpy(base + at, t.data, t.bytes);
            m->stored[order[k]] = Ref{copy, at};
            at += round_up(t.bytes, 256);
        }
        // A run of empty tensors still took a buffer: count it as copied.
        copied += (int64_t)std::max<size_t>(total, 1);
    }
    return copied;
}

int32_t model_load(void *ctx, const turbo_backend_model *desc, void **out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Context *c = static_cast<Context *>(ctx);
        if (desc->family != TURBO_FAMILY_BERT)
            return refuse(err, TURBO_E_UNSUPPORTED, "family %u: the metal backend holds BERT encoders", desc->family);
        if (desc->heads == 0 || desc->hidden % desc->heads != 0)
            return refuse(err, TURBO_E_UNSUPPORTED, "hidden %u is not a multiple of heads %u", desc->hidden,
                          desc->heads);
        // The kernels work in 8 x 8 matrices.
        if (desc->hidden % 8 || desc->intermediate % 8)
            return refuse(err, TURBO_E_UNSUPPORTED,
                          "hidden %u and intermediate %u: the metal backend needs each a multiple of 8", desc->hidden,
                          desc->intermediate);
        const size_t elem = desc->dtype == TURBO_DTYPE_F32 ? 4 : 2;
        Model *m = make<Model>();
        m->ctx = c;
        m->desc = *desc;
        m->desc.tensors = nullptr;
        const int64_t copied = map_weights(m, desc);
        if (copied < 0) {
            delete m;
            return refuse(err, TURBO_E_OUT_OF_MEMORY, "host memory for a copy of the weights");
        }
        m->in_place = copied == 0;
        if (copied)
            c->say(LOG_WARNING, "metal device %u: Metal did not map some pages the weights are on; copied %lld bytes",
                   c->ordinal, (long long)copied);
        for (uint32_t i = 0; i < desc->tensor_count; i++) m->counts.push_back(desc->tensors[i].bytes / elem);
        if (desc->dtype == TURBO_DTYPE_F32) m->f32 = m->stored;
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
        const bool oom =
            e && [e.domain isEqualToString:MTLCommandBufferErrorDomain] && e.code == MTLCommandBufferErrorOutOfMemory;
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
    id<MTLBuffer> wide = new_buffer(c->device, total, MTLResourceStorageModePrivate | WEIGHTS);
    if (!wide) return refuse(err, TURBO_E_OUT_OF_MEMORY, "%zu bytes for the weights in F32", total);
    {
        std::lock_guard<std::mutex> cg(c->lock);
        id<MTLCommandBuffer> cb = [c->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:c->kernels->k[m->desc.dtype == TURBO_DTYPE_F16 ? WIDEN_F16 : WIDEN_BF16]];
        for (size_t i = 0; i < m->counts.size(); i++) {
            const uint32_t n = (uint32_t)m->counts[i];
            [enc setBuffer:m->stored[i].buf offset:m->stored[i].at atIndex:0];
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
// allocated here; a run allocates nothing.
//
// The rows are packed, as on the CPU: embed_write copies each row's
// columns up to its last live token one after another into the session's
// shared memory, where the GPU reads them, with each token's column for
// its position, each row's start and length, and the attention's blocks
// of 8 queries. Padding past a row's last live token is never computed: no
// output depends on it. The packed tokens are rounded up to a multiple of
// 32 for the matrix kernels, and those extra tokens are real lookups
// (token 0 at column 0) whose results no output reads. With one memory
// nothing crosses to a device, so h2d_bytes and d2h_bytes are 0. The
// vectors are left in the session's shared buffer, which the host reads
// where they are.

/* Kernel parameter blocks, as kernels.metal declares them. */
struct RowParams {
    uint32_t hidden;
    float eps;
};
struct GemmParams {
    uint32_t m, n, k, epilogue, splits, kchunk;
};
struct AttentionParams {
    uint32_t hidden, head_dim;
    float scale;
};
struct PoolParams {
    uint32_t hidden, output_dim, pooling, l2;
};

enum : uint32_t { EPILOGUE_NONE = 0, EPILOGUE_BIAS = 1, EPILOGUE_BIAS_GELU = 2 };

/* Threadgroups a linear layer's dispatch should have to keep the GPU's
 * cores busy: several per core on the largest Apple GPUs. */
constexpr uint32_t SPREAD = 128;

/* The packed tokens a batch of `tokens` can take, rounded up to a
 * multiple of 32, with room for attention's last chunk of 32 keys to read
 * past the last token. */
size_t capacity(size_t tokens) { return round_up(tokens + 31, 32); }

/* Whether attention runs on SIMD-group matrices for this head width. */
bool wide_heads(uint32_t head_dim) { return head_dim % 8 == 0 && head_dim <= 64; }

/* Attention's threadgroup memory: for the matrix kernel, 32 queries and a
 * chunk of 32 keys' K and V, rows padded by 4 floats, and four groups' 8 x
 * 32 scores; for the narrow one, one query's scores against seq keys. */
size_t attention_bytes(uint32_t head_dim, uint32_t seq) {
    const size_t floats = wide_heads(head_dim) ? 3 * 32 * (head_dim + 4) + 4 * 8 * 32 : round_up(seq, 8);
    return round_up(sizeof(float) * floats, 16);
}

struct Session {
    Model *model = nullptr;
    Context *ctx = nullptr;
    uint32_t max_batch = 0, max_seq = 0, cap = 0;
    /* Shared, int32 each: ids, types, mask and pos [cap]; rows [max_batch]
     * of (start, length); blocks [cap / 32 + max_batch] of (row, block of
     * 32 queries). */
    id<MTLBuffer> rows = nil;
    uint64_t ids = 0, types = 0, mask = 0, pos = 0, starts = 0, blocks = 0;
    id<MTLBuffer> scratch = nil;
    uint64_t x = 0, q = 0, k = 0, v = 0, att = 0, tmp = 0, ffn = 0;
    /* The session's own queue: sessions run at the same time, each on the
     * GPU cores the others leave free, and a session has one owner at a
     * time, so its queue needs no lock. */
    id<MTLCommandQueue> queue = nil;
    /* [max_batch, hidden] F32, handed out as the output. */
    Buffer output;
    bool written = false;
    /* What the last write left: the rows, the tokens packed and rounded
     * up, the attention blocks, and the options. */
    uint32_t batch = 0, tokens = 0, padded = 0, nblocks = 0, longest = 0;
    uint32_t pooling = 0, normalize = 0, output_dim = 0;
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
        const size_t shared = attention_bytes(d.hidden / d.heads, max_seq);
        const Kernel att = wide_heads(d.hidden / d.heads) ? ATTENTION : ATTENTION_NARROW;
        const size_t most = c->device.maxThreadgroupMemoryLength - c->kernels->k[att].staticThreadgroupMemoryLength;
        if (shared > most)
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 2,
                                "max_seq %u: attention needs %zu bytes of threadgroup memory, and device %u gives it "
                                "%zu",
                                max_seq, shared, c->ordinal, most);
        const size_t cap = capacity((size_t)max_batch * max_seq);
        const size_t widest = std::max(d.intermediate, d.hidden);
        const size_t wide = round_up(cap * d.hidden * 4, 256);
        const size_t ffn = round_up(cap * d.intermediate * 4, 256);
        const size_t nblk = cap / 32 + max_batch;
        const size_t ints = round_up(cap * 4, 256), pairs = round_up(max_batch * 8, 256),
                     block_pairs = round_up(nblk * 8, 256);
        if (cap > UINT32_MAX / widest || 6 * wide + ffn > c->device.maxBufferLength)
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 1,
                                "%u rows of %u tokens is more than one of device %u's buffers holds", max_batch,
                                max_seq, c->ordinal);
        TRY(f32_weights(m, err));

        Session *s = make<Session>();
        s->model = m;
        s->ctx = c;
        s->max_batch = max_batch;
        s->max_seq = max_seq;
        s->cap = (uint32_t)cap;
        s->rows = new_buffer(c->device, 4 * ints + pairs + block_pairs, MTLResourceStorageModeShared);
        s->scratch = new_buffer(c->device, 6 * wide + ffn, MTLResourceStorageModePrivate);
        s->queue = [c->device newCommandQueue];
        const uint64_t vectors = (uint64_t)max_batch * d.hidden * 4;
        id<MTLBuffer> output = new_buffer(c->device, vectors, MTLResourceStorageModeShared);
        if (!s->rows || !s->scratch || !output || !s->queue) {
            delete s;
            return refuse(err, TURBO_E_OUT_OF_MEMORY, "%llu bytes for the session", (unsigned long long)(
                          4 * ints + pairs + block_pairs + 6 * wide + ffn + vectors));
        }
        s->ids = 0;
        s->types = ints;
        s->mask = 2 * ints;
        s->pos = 3 * ints;
        s->starts = 4 * ints;
        s->blocks = 4 * ints + pairs;
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

/* The rows, packed into the session: see above. */
int32_t embed_write(void *session, const turbo_backend_embed_rows *r, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Session &s = *static_cast<Session *>(session);
        char *base = static_cast<char *>(s.rows.contents);
        int32_t *ids = reinterpret_cast<int32_t *>(base + s.ids);
        int32_t *types = reinterpret_cast<int32_t *>(base + s.types);
        int32_t *mask = reinterpret_cast<int32_t *>(base + s.mask);
        int32_t *pos = reinterpret_cast<int32_t *>(base + s.pos);
        uint32_t *starts = reinterpret_cast<uint32_t *>(base + s.starts);
        uint32_t *blocks = reinterpret_cast<uint32_t *>(base + s.blocks);
        uint32_t t = 0, nb = 0, longest = 0;
        for (uint32_t row = 0; row < r->batch; row++) {
            const size_t at = (size_t)row * r->row_stride;
            const int32_t *m = r->mask + at;
            uint32_t len = r->seq;
            while (m[len - 1] == 0) len--;   // the core guarantees a live token
            memcpy(ids + t, r->ids + at, (size_t)len * 4);
            memcpy(mask + t, m, (size_t)len * 4);
            if (r->types)
                memcpy(types + t, r->types + at, (size_t)len * 4);
            else
                memset(types + t, 0, (size_t)len * 4);
            for (uint32_t p = 0; p < len; p++) pos[t + p] = (int32_t)p;
            starts[2 * row] = t;
            starts[2 * row + 1] = len;
            for (uint32_t b = 0; b * 32 < len; b++) {
                blocks[2 * nb] = row;
                blocks[2 * nb + 1] = b;
                nb++;
            }
            t += len;
            longest = std::max(longest, len);
        }
        const uint32_t padded = (uint32_t)capacity(t);
        memset(ids + t, 0, (size_t)(padded - t) * 4);
        memset(types + t, 0, (size_t)(padded - t) * 4);
        memset(mask + t, 0, (size_t)(padded - t) * 4);
        memset(pos + t, 0, (size_t)(padded - t) * 4);
        s.batch = r->batch;
        s.tokens = t;
        s.padded = padded;
        s.nblocks = nb;
        s.longest = longest;
        s.pooling = r->pooling;
        s.normalize = r->normalize;
        s.output_dim = r->output_dim;
        s.written = true;
        return TURBO_OK;
    });
}

/* The encoder over the packed tokens, into one compute pass. Its
 * dispatches run in order, each seeing what the one before wrote. */
void encode(Session &s, id<MTLComputeCommandEncoder> enc) {
    const turbo_backend_model &d = s.model->desc;
    const std::vector<Ref> &w = s.model->f32;
    const Kernels &kn = *s.ctx->kernels;
    const uint32_t tokens = s.padded;
    const uint32_t h = d.hidden, inter = d.intermediate, hd = d.hidden / d.heads;
    id<MTLBuffer> sc = s.scratch, rows = s.rows;
    auto bind = [&](const Ref &r, NSUInteger i) { [enc setBuffer:r.buf offset:r.at atIndex:i]; };
    auto layer = [&](uint32_t l, int r) -> const Ref & {
        return w[TURBO_BERT_EMBEDDING_TENSORS + l * TURBO_BERT_LAYER_TENSORS + r];
    };
    auto per_token = [&]() {
        [enc dispatchThreadgroups:MTLSizeMake(tokens / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    };
    // Up to three layers of one shape, over the same input, in one dispatch.
    // Up to three layers of one shape, over the same input, in one
    // dispatch. A single layer with no epilogue whose tiles would leave
    // most of the GPU idle has its k split, so more threadgroups share the
    // work; the splits land side by side in its output, which holds
    // capacity rows, and add_ln sums them. Returns the splits.
    auto gemm = [&](uint64_t x, uint32_t n_in, uint32_t n_out, uint32_t epilogue,
                    std::initializer_list<std::pair<const Ref *, uint64_t>> outs,
                    std::initializer_list<const Ref *> biases) -> uint32_t {
        const uint32_t tiles = (n_out + 63) / 64 * (tokens / 32);
        uint32_t splits = 1;
        if (outs.size() == 1 && epilogue == EPILOGUE_NONE && tiles < SPREAD) {
            splits = std::min({(SPREAD + tiles - 1) / tiles, n_in / 64, s.cap / tokens, 8u});
            splits = std::max(splits, 1u);
        }
        const uint32_t kchunk = (uint32_t)round_up((n_in + splits - 1) / splits, 16);
        splits = (n_in + kchunk - 1) / kchunk;
        const GemmParams p{tokens, n_out, n_in, epilogue, splits, kchunk};
        [enc setComputePipelineState:kn.k[GEMM]];
        [enc setBuffer:sc offset:x atIndex:0];
        NSUInteger i = 0;
        for (const auto &o : outs) {
            bind(*o.first, 1 + i);
            [enc setBuffer:sc offset:o.second atIndex:7 + i];
            i++;
        }
        const NSUInteger n = i;
        i = 0;
        for (const Ref *b : biases) bind(*b, 4 + i++);
        // Unused bindings name something valid; the kernel never reads them.
        for (; i < 3; i++) bind(w[0], 4 + i);
        for (NSUInteger j = n; j < 3; j++) {
            bind(w[0], 1 + j);
            [enc setBuffer:sc offset:0 atIndex:7 + j];
        }
        [enc setBytes:&p length:sizeof p atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake((n_out + 63) / 64, tokens / 32, n * splits)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return splits;
    };
    const RowParams rp{h, (float)d.layer_norm_eps};
    auto add_ln = [&](uint64_t y, uint32_t splits, const Ref &bias, const Ref &ln_w, const Ref &ln_b) {
        [enc setComputePipelineState:kn.k[ADD_LN]];
        [enc setBuffer:sc offset:s.x atIndex:0];
        [enc setBuffer:sc offset:y atIndex:1];
        bind(bias, 2);
        bind(ln_w, 3);
        bind(ln_b, 4);
        [enc setBytes:&rp length:sizeof rp atIndex:5];
        [enc setBytes:&splits length:sizeof splits atIndex:6];
        [enc setBytes:&tokens length:sizeof tokens atIndex:7];
        per_token();
    };

    [enc setComputePipelineState:kn.k[EMBED_LN]];
    [enc setBuffer:rows offset:s.ids atIndex:0];
    [enc setBuffer:rows offset:s.types atIndex:1];
    [enc setBuffer:rows offset:s.pos atIndex:2];
    bind(w[WORD], 3);
    bind(w[POSITION], 4);
    bind(w[TOKEN_TYPE], 5);
    bind(w[EMB_LN_W], 6);
    bind(w[EMB_LN_B], 7);
    [enc setBuffer:sc offset:s.x atIndex:8];
    [enc setBytes:&rp length:sizeof rp atIndex:9];
    per_token();

    const AttentionParams ap{h, hd, 1.0f / sqrtf((float)hd)};
    for (uint32_t l = 0; l < d.layers; l++) {
        gemm(s.x, h, h, EPILOGUE_BIAS,
             {{&layer(l, TURBO_BERT_Q_WEIGHT), s.q}, {&layer(l, TURBO_BERT_K_WEIGHT), s.k},
              {&layer(l, TURBO_BERT_V_WEIGHT), s.v}},
             {&layer(l, TURBO_BERT_Q_BIAS), &layer(l, TURBO_BERT_K_BIAS), &layer(l, TURBO_BERT_V_BIAS)});

        const bool wide = wide_heads(hd);
        [enc setComputePipelineState:kn.k[wide ? ATTENTION : ATTENTION_NARROW]];
        [enc setBuffer:sc offset:s.q atIndex:0];
        [enc setBuffer:sc offset:s.k atIndex:1];
        [enc setBuffer:sc offset:s.v atIndex:2];
        [enc setBuffer:rows offset:s.mask atIndex:3];
        [enc setBuffer:rows offset:s.starts atIndex:4];
        [enc setBuffer:rows offset:s.blocks atIndex:5];
        [enc setBuffer:sc offset:s.att atIndex:6];
        [enc setBytes:&ap length:sizeof ap atIndex:7];
        [enc setThreadgroupMemoryLength:attention_bytes(hd, s.longest) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(s.nblocks, d.heads, 1) threadsPerThreadgroup:MTLSizeMake(wide ? 128 : 32, 1, 1)];

        uint32_t splits = gemm(s.att, h, h, EPILOGUE_NONE, {{&layer(l, TURBO_BERT_ATTN_OUT_WEIGHT), s.tmp}}, {});
        add_ln(s.tmp, splits, layer(l, TURBO_BERT_ATTN_OUT_BIAS), layer(l, TURBO_BERT_ATTN_LN_WEIGHT),
               layer(l, TURBO_BERT_ATTN_LN_BIAS));
        gemm(s.x, h, inter, EPILOGUE_BIAS_GELU, {{&layer(l, TURBO_BERT_FFN_IN_WEIGHT), s.ffn}},
             {&layer(l, TURBO_BERT_FFN_IN_BIAS)});
        splits = gemm(s.ffn, inter, h, EPILOGUE_NONE, {{&layer(l, TURBO_BERT_FFN_OUT_WEIGHT), s.tmp}}, {});
        add_ln(s.tmp, splits, layer(l, TURBO_BERT_FFN_OUT_BIAS), layer(l, TURBO_BERT_FFN_LN_WEIGHT),
               layer(l, TURBO_BERT_FFN_LN_BIAS));
    }

    const PoolParams pp{h, s.output_dim, s.pooling, s.normalize == TURBO_NORMALIZE_L2 ? 1u : 0u};
    [enc setComputePipelineState:kn.k[POOL]];
    [enc setBuffer:sc offset:s.x atIndex:0];
    [enc setBuffer:rows offset:s.mask atIndex:1];
    [enc setBuffer:rows offset:s.starts atIndex:2];
    [enc setBuffer:s.output.mtl offset:0 atIndex:3];
    [enc setBytes:&pp length:sizeof pp atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(s.batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

int32_t session_run(void *session, turbo_backend_run *out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Session &s = *static_cast<Session *>(session);
        if (!s.written)
            return refuse(err, TURBO_E_INVALID_STATE, "the metal session has no rows written since its last run");
        s.written = false;
        const uint64_t host0 = host_here, device0 = device_here;
        {
            id<MTLCommandBuffer> cb = [s.queue commandBufferWithUnretainedReferences];
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
        // With one memory there is no upload: embed_write copied the rows
        // into the session, as on the CPU.
        st[TURBO_EMBED_STAGE_UPLOAD] = TURBO_STAGE_UNUSED;
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
