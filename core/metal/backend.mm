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
#include <chrono>
#include <cctype>
#include <cstdarg>
#include <cstddef>
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

/* Embed at every precision: in F32 at EXACT and MODEL, and at FASTEST
 * with the linear layers in F16. As on the CPU, a model stored in F16 or
 * BF16 computes from a converted copy at EXACT and FASTEST, and its
 * session at MODEL is refused. */
int32_t capability(uint32_t ordinal, uint32_t, uint32_t precision, uint32_t *status, uint32_t *dtype,
                   uint32_t *options_honored,
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
        *dtype = precision == TURBO_PRECISION_FASTEST ? TURBO_DTYPE_F16 : TURBO_DTYPE_F32;
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
constexpr uint32_t LOG_INFO = 2;
constexpr uint32_t LOG_DEBUG = 3;

/* The F16 linear layers come in fours, for x and y in F32 or F16: ff,
 * fh, hf, hh, in that order (kernels.metal). */
enum Kernel {
    EMBED_LN, ADD_LN, GEMM, GEMM_SMALL, ATTENTION, ATTENTION_NARROW, POOL, POOL_RANGES, WIDEN_F16, WIDEN_BF16,
    NARROW_F16, ATTENTION_H, ATTENTION_HR, GEMM_H32, GEMM_H64 = GEMM_H32 + 4, GEMM_H128 = GEMM_H64 + 4,
    GEMM_H32N32 = GEMM_H128 + 4, GEMM_SMALL_H = GEMM_H32N32 + 4, ABS_MAX_F16 = GEMM_SMALL_H + 4, KERNELS
};
const char *const KERNEL_NAMES[KERNELS] = {
    "embed_layer_norm", "add_layer_norm",  "gemm",            "gemm_small",      "attention",       "attention_narrow",
    "pool",             "pool_ranges",     "widen_f16",       "widen_bf16",      "narrow_f16",      "attention_h",
    "attention_hr",
    "gemm_h32_ff",      "gemm_h32_fh",     "gemm_h32_hf",     "gemm_h32_hh",     "gemm_h64_ff",     "gemm_h64_fh",
    "gemm_h64_hf",      "gemm_h64_hh",     "gemm_h128_ff",    "gemm_h128_fh",    "gemm_h128_hf",    "gemm_h128_hh",
    "gemm_h32n32_ff",   "gemm_h32n32_fh",  "gemm_h32n32_hf",  "gemm_h32n32_hh",
    "gemm_small_h_ff",  "gemm_small_h_fh", "gemm_small_h_hf", "gemm_small_h_hh", "abs_max_f16"};

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
        for (int kk = GEMM; kk < KERNELS; kk++) {
            if (kk == ATTENTION_NARROW || (kk >= POOL && kk < ATTENTION_H)) continue;
            if (rc == TURBO_OK && c->kernels->k[kk].maxTotalThreadsPerThreadgroup < 128)
                rc = refuse(err, TURBO_E_UNSUPPORTED, "device %u runs the %s kernel in threadgroups of %lu; it needs 128",
                            ordinal, KERNEL_NAMES[kk], (unsigned long)c->kernels->k[kk].maxTotalThreadsPerThreadgroup);
        }
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
    /* The linear layers' weights in F16, for FASTEST: stored for an F16
     * model, else into narrowed once a FASTEST session made it; the other
     * tensors' entries are empty. */
    id<MTLBuffer> narrowed = nil;
    std::vector<Ref> f16;
    /* Whether a session tried to make f16; it stays empty when a weight
     * is past F16's range, and FASTEST then computes in F32. */
    bool narrow_tried = false;
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
        if (desc->family != TURBO_FAMILY_BERT && desc->family != TURBO_FAMILY_ROBERTA)
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

/* Whether tensor i is a linear layer's weight: Q, K, V, the attention
 * output and the feed-forward block's two. */
bool linear_weight(uint32_t i) {
    if (i < TURBO_BERT_EMBEDDING_TENSORS) return false;
    const uint32_t r = (i - TURBO_BERT_EMBEDDING_TENSORS) % TURBO_BERT_LAYER_TENSORS;
    return r == TURBO_BERT_Q_WEIGHT || r == TURBO_BERT_K_WEIGHT || r == TURBO_BERT_V_WEIGHT ||
           r == TURBO_BERT_ATTN_OUT_WEIGHT || r == TURBO_BERT_FFN_IN_WEIGHT || r == TURBO_BERT_FFN_OUT_WEIGHT;
}

/* The linear layers' weights in F16, made on first need from the F32
 * ones, which f32_weights has made. An F16 model's are its own where each
 * starts 4-byte aligned, as the kernels read them 16 bytes at a time. A
 * finite F32 or BF16 weight past F16's range (65504) leaves f16 empty, with
 * a warning, and the model's sessions run no F16 kernels. */
int32_t f16_weights(Model *m, turbo_error *err) {
    std::lock_guard<std::mutex> g(m->widen_lock);
    if (m->narrow_tried) return TURBO_OK;
    const size_t n = m->counts.size();
    bool aligned = m->desc.dtype == TURBO_DTYPE_F16;
    for (size_t i = 0; i < n && aligned; i++)
        aligned = !linear_weight((uint32_t)i) || m->stored[i].at % 4 == 0;
    if (aligned) {
        m->f16.assign(n, Ref{});
        for (size_t i = 0; i < n; i++)
            if (linear_weight((uint32_t)i)) m->f16[i] = m->stored[i];
        m->narrow_tried = true;
        return TURBO_OK;
    }
    Context *c = m->ctx;
    size_t total = 0;
    std::vector<uint64_t> at(n, 0);
    for (size_t i = 0; i < n; i++) {
        if (!linear_weight((uint32_t)i)) continue;
        at[i] = total;
        total += round_up(m->counts[i] * 2, 256);
    }
    id<MTLBuffer> half = new_buffer(c->device, std::max<size_t>(total, 1), MTLResourceStorageModePrivate | WEIGHTS);
    id<MTLBuffer> flag = new_buffer(c->device, sizeof(uint32_t), MTLResourceStorageModeShared);
    if (!half || !flag)
        return refuse(err, TURBO_E_OUT_OF_MEMORY, "%zu bytes for the linear layers' weights in F16", total);
    *static_cast<uint32_t *>(flag.contents) = 0;
    {
        std::lock_guard<std::mutex> cg(c->lock);
        id<MTLCommandBuffer> cb = [c->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:c->kernels->k[NARROW_F16]];
        for (size_t i = 0; i < n; i++) {
            if (!linear_weight((uint32_t)i)) continue;
            const uint32_t count = (uint32_t)m->counts[i];
            [enc setBuffer:m->f32[i].buf offset:m->f32[i].at atIndex:0];
            [enc setBuffer:half offset:at[i] atIndex:1];
            [enc setBytes:&count length:sizeof count atIndex:2];
            [enc setBuffer:flag offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
        TRY(finish(cb, err, "narrowing the weights"));
    }
    m->narrow_tried = true;
    if (*static_cast<const uint32_t *>(flag.contents) != 0) {
        c->say(LOG_WARNING,
               "metal device %u: a linear layer's weight of this %s model is past F16's range; its sessions run "
               "no F16 kernels",
               c->ordinal, dtype_name(m->desc.dtype));
        return TURBO_OK;
    }
    m->narrowed = half;
    m->f16.assign(n, Ref{});
    for (size_t i = 0; i < n; i++)
        if (linear_weight((uint32_t)i)) m->f16[i] = Ref{half, at[i]};
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
// of 32 queries. Padding past a row's last live token is never computed: no
// output depends on it. The packed tokens are rounded up to a multiple of
// 32 for the matrix kernels, and those extra tokens are real lookups
// (token 0 at column 0) whose results no output reads. With one memory
// nothing crosses to a device, so h2d_bytes and d2h_bytes are 0. The
// vectors are left in the session's shared buffer, which the host reads
// where they are.
//
// embed_write_ranges packs the rows the same way and copies its ranges
// into a region of the same shared memory after the blocks, sized for the
// session's max_ranges when it is made. The run then pools one vector per
// range instead of one per row, so the output buffer holds
// max(max_batch, max_ranges) vectors of hidden F32.

/* Kernel parameter blocks, as kernels.metal declares them. */
struct RowParams {
    uint32_t hidden;
    float eps;
    uint32_t half_copy;
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
/* pool_ranges reads each range as three uints, as turbo.h lays it out. */
static_assert(sizeof(turbo_token_range) == 12, "turbo_token_range is three uint32");

enum : uint32_t { EPILOGUE_NONE = 0, EPILOGUE_BIAS = 1, EPILOGUE_BIAS_GELU = 2 };

/* Threadgroups a linear layer's dispatch should have to keep the GPU's
 * cores busy: several per core on the largest Apple GPUs. */
constexpr uint32_t SPREAD = 128;

/* Up to this many packed tokens, the linear layers run the small kernel,
 * which reads the weights with more threadgroups in flight; above it the
 * tiled kernel, which reuses them more, is faster. In F16 the tiled
 * kernel is the faster from fewer tokens on (on the M2, 96 and up). */
constexpr uint32_t SMALL_TOKENS = 128, SMALL_TOKENS_F16 = 64;

/* linear=auto's tile: h32n32's 32 x 32 up to this many packed tokens,
 * where more threadgroups fill the GPU, and 64 x 64 above it (on the
 * M2). */
constexpr uint32_t AUTO_H32_TOKENS = 1024;

/* The packed tokens a batch of `tokens` can take, rounded up to a
 * multiple of 32. */
size_t capacity(size_t tokens) { return round_up(tokens, 32); }

/* Which kernels run a session's linear layers: F32 at EXACT and MODEL;
 * at FASTEST F16 operands with F32 sums, past SMALL_TOKENS_F16 in tiles of
 * 32 x 64, 64 x 64 or 128 x 64 (gemm_h), or 32 x 32 (gemm_hn), or auto:
 * the tile each run's tokens favor (docs/metal.md, Kernel choices). */
enum Linear : uint32_t { LINEAR_F32, LINEAR_H32, LINEAR_H64, LINEAR_H128, LINEAR_H32N32, LINEAR_AUTO, LINEAR_COUNT };
const char *const LINEAR_NAMES[] = {"f32", "h32", "h64", "h128", "h32n32", "auto"};
/* FASTEST's choice when nothing forces or tunes it. */
constexpr Linear LINEAR_DEFAULT = LINEAR_AUTO;

/* Which kernel runs FASTEST's attention, for heads the matrix kernel
 * takes: staged passes each group's scores and probabilities through
 * threadgroup memory, registers keeps them in the SIMD-group matrices'
 * registers, and auto runs registers past ATTN_REGISTERS_TOKENS packed
 * tokens and staged up to them (docs/metal.md, Kernel choices). */
enum Attn : uint32_t { ATTN_STAGED, ATTN_REGISTERS, ATTN_AUTO, ATTN_COUNT };
const char *const ATTN_NAMES[] = {"staged", "registers", "auto"};
/* FASTEST's choice when nothing forces it. */
constexpr Attn ATTN_DEFAULT = ATTN_AUTO;

/* attn=auto's line: up to this many packed tokens staged is the faster
 * on the M2 (by 3 to 4% at 2 and 4 rows of 128), and registers past it
 * (by 0.2 to 0.7% from 8 to 32 rows of 128, 1.6% at 32 of 256). */
constexpr uint32_t ATTN_REGISTERS_TOKENS = 512;

/* Whether the context's device runs a choice's kernels: h128's tile takes
 * threadgroups of 256. */
bool runs_linear(const Context *c, Linear l) {
    if (l != LINEAR_H128) return true;
    for (int i = 0; i < 4; i++)
        if (c->kernels->k[GEMM_H128 + i].maxTotalThreadsPerThreadgroup < 256) return false;
    return true;
}

/* Whether attention runs on SIMD-group matrices for this head width. */
bool wide_heads(uint32_t head_dim) { return head_dim % 8 == 0 && head_dim <= 64; }

/* Attention's threadgroup memory: for the matrix kernel, 32 queries and a
 * chunk of 32 keys' K and V, rows padded by 4 floats, and four groups' 8 x
 * 32 scores; for the narrow one, one query's scores against seq keys. */
size_t attention_bytes(uint32_t head_dim, uint32_t seq) {
    const size_t floats = wide_heads(head_dim) ? 3 * 32 * (head_dim + 4) + 4 * 8 * 32 : round_up(seq, 8);
    return round_up(sizeof(float) * floats, 16);
}

/* The F16 attention's: a chunk's K transposed, [head_dim][32 + 8], and V,
 * [32][head_dim + 8], in F16; for the staged kernel, four groups' 8 x 32
 * scores in F32 and probabilities in F16; and the chunk's 32 key biases
 * in F32. */
size_t attention_h_bytes(uint32_t head_dim, bool registers) {
    return round_up(2 * (head_dim * (32 + 8) + 32 * (head_dim + 8)) + (registers ? 0 : 6 * 4 * 8 * 32) + 4 * 32, 16);
}

struct Session {
    Model *model = nullptr;
    Context *ctx = nullptr;
    uint32_t max_batch = 0, max_seq = 0, cap = 0;
    /* The most ranges one embed_write_ranges holds. */
    uint32_t max_ranges = 0;
    /* Shared, int32 each: ids, types, mask and pos [cap]; rows [max_batch]
     * of (start, length); blocks [cap / 32 + max_batch] of (row, block of
     * 32 queries); ranges [max_ranges] of (row, start, end), as
     * turbo_token_range lays them out. */
    id<MTLBuffer> rows = nil;
    uint64_t ids = 0, types = 0, mask = 0, pos = 0, starts = 0, blocks = 0, ranges = 0;
    id<MTLBuffer> scratch = nil;
    /* xh: x in F16, for FASTEST's linear layers, after ffn. */
    uint64_t x = 0, q = 0, k = 0, v = 0, att = 0, tmp = 0, ffn = 0, xh = 0;
    /* The session's own queue: sessions run at the same time, each on the
     * GPU cores the others leave free, and a session has one owner at a
     * time, so its queue needs no lock. */
    id<MTLCommandQueue> queue = nil;
    /* [max(max_batch, max_ranges), hidden] F32, handed out as the output:
     * a vector per row after embed_write, per range after
     * embed_write_ranges, each output_dim values, packed. */
    Buffer output;
    bool written = false;
    /* What the last write left: the rows, the tokens packed and rounded
     * up, the attention blocks, the ranges (0 after embed_write), and the
     * options. */
    uint32_t batch = 0, tokens = 0, padded = 0, nblocks = 0, longest = 0, nranges = 0;
    uint32_t pooling = 0, normalize = 0, output_dim = 0;
    Linear linear = LINEAR_F32;
    /* Whether TURBO_METAL_CHOICES named the linear layers' kernels. */
    bool forced = false;
    Attn attn = ATTN_DEFAULT;
    /* Whether it named attention's. */
    bool attn_forced = false;
    /* For the tests (turbo_metal_track_peaks): a shared buffer of
     * PEAK_SLOTS uints, the largest magnitude of each F16 tensor a run
     * stores, as F32 bits; nil unless tracking was on when it was made. */
    id<MTLBuffer> peaks = nil;
};

/* The F16 tensors a FASTEST run stores between kernels, as
 * turbo_metal_peaks orders them: Q, K, V, the attention context, GELU's
 * output and the residual stream's F16 copy. */
constexpr int PEAK_SLOTS = 6;

/* Whether new sessions track their peaks, and the last tracked run's,
 * for the tests. */
struct TestPeaks {
    std::mutex lock;
    bool on = false;
    float last[PEAK_SLOTS] = {};
};
TestPeaks &test_peaks() {
    static TestPeaks &t = *new TestPeaks;
    return t;
}

/* TURBO_METAL_CHOICES, read when a session is made: items separated by
 * ';', "linear=<f32|h32|h64|h128|auto>" forcing the linear layers' kernels into
 * *linear, "attn=<staged|registers|auto>" forcing attention's into *attn,
 * and "forced=" as a session reports it, whatever it names, read and left
 * alone. Each is -1 when nothing names it. */
/* A line the tests set in place of the variable, through
 * turbo_metal_use_choices. */
struct TestChoices {
    std::mutex lock;
    bool set = false;
    std::string line;
};
TestChoices &test_choices() {
    static TestChoices &t = *new TestChoices;
    return t;
}

int32_t forced_choices(int *linear, int *attn, turbo_error *err) {
    *linear = -1;
    *attn = -1;
    std::string line;
    {
        TestChoices &t = test_choices();
        std::lock_guard<std::mutex> g(t.lock);
        if (t.set) {
            line = t.line;
        } else {
            const char *env = getenv("TURBO_METAL_CHOICES");
            if (!env) return TURBO_OK;
            line = env;
        }
    }
    for (size_t at = 0; at <= line.size();) {
        size_t end = line.find(';', at);
        if (end == std::string::npos) end = line.size();
        const std::string item = line.substr(at, end - at);
        at = end + 1;
        if (item.empty() || item.rfind("forced=", 0) == 0) continue;
        int found = -1, found_attn = -1;
        for (int l = LINEAR_F32; l < (int)LINEAR_COUNT; l++)
            if (item == std::string("linear=") + LINEAR_NAMES[l]) found = l;
        for (int a = ATTN_STAGED; a < (int)ATTN_COUNT; a++)
            if (item == std::string("attn=") + ATTN_NAMES[a]) found_attn = a;
        if (found < 0 && found_attn < 0)
            return refuse(err, TURBO_E_INVALID_ARGUMENT,
                          "TURBO_METAL_CHOICES item \"%s\": the metal backend takes linear=f32, h32, h64, h128, h32n32 or "
                          "auto, and attn=staged, registers or auto",
                          item.c_str());
        if (found >= 0) *linear = found;
        if (found_attn >= 0) *attn = found_attn;
    }
    return TURBO_OK;
}

/* The numeric classes the core allows each precision (core/src/tuning.rs),
 * for a session made without a tuning struct to say. */
uint32_t numerics_for(uint32_t precision) {
    return precision == TURBO_PRECISION_FASTEST ? TURBO_NUMERIC_F16_F32ACC | TURBO_NUMERIC_F16_CHUNKACC
                                                : TURBO_NUMERIC_F32_FMA;
}

/* A session for session_create and session_open: max_ranges is
 * max_batch for the first, the description's for the second; numerics
 * the TURBO_NUMERIC_* classes the precision allows. The linear layers run
 * in F16 where F16 operands with F32 sums are allowed, else in F32, and
 * a kernel outside the classes is never launched. */
int32_t open_session(void *model, uint32_t task, uint32_t max_batch, uint32_t max_seq, uint32_t precision,
                     uint32_t max_ranges, uint32_t numerics, uint32_t *compute_dtype, void **out,
                     turbo_error *err) {
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
        if ((uint64_t)max_seq + d.position_offset > d.max_positions)
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 2,
                                "max_seq %u from position %u is over the model's %u positions", max_seq,
                                d.position_offset, d.max_positions);
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
                     block_pairs = round_up(nblk * 8, 256),
                     triples = round_up((size_t)max_ranges * sizeof(turbo_token_range), 256);
        const bool half = (numerics & TURBO_NUMERIC_F16_F32ACC) != 0;
        const size_t xh = half ? round_up(cap * d.hidden * 2, 256) : 0;
        if (cap > UINT32_MAX / widest || 6 * wide + ffn + xh > c->device.maxBufferLength)
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 1,
                                "%u rows of %u tokens is more than one of device %u's buffers holds", max_batch,
                                max_seq, c->ordinal);
        // A kernel the environment forces must compute in a class the
        // precision allows.
        int forced = -1, forced_attn = -1;
        TRY(forced_choices(&forced, &forced_attn, err));
        const uint32_t needs = forced == (int)LINEAR_F32 ? TURBO_NUMERIC_F32_FMA : TURBO_NUMERIC_F16_F32ACC;
        if (forced >= 0 && !(numerics & needs))
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 3,
                                "TURBO_METAL_CHOICES linear=%s: its kernels compute in %s, which precision %u "
                                "does not allow",
                                LINEAR_NAMES[forced], forced == (int)LINEAR_F32 ? "F32" : "F16 with F32 sums",
                                precision);
        if (forced > (int)LINEAR_F32 && !runs_linear(c, (Linear)forced))
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 3,
                                "TURBO_METAL_CHOICES linear=%s: device %u runs its kernels in threadgroups of fewer "
                                "than 256",
                                LINEAR_NAMES[forced], c->ordinal);
        TRY(f32_weights(m, err));
        // The linear layers' kernels: the forced ones, else F16 where
        // allowed, else F32 where allowed. A model with a weight past
        // F16's range has no F16 kernels; a precision that allows only F16
        // is refused for it.
        const bool f32 = (numerics & TURBO_NUMERIC_F32_FMA) != 0;
        if (forced < 0 && !half && !f32)
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 3,
                                "precision %u allows none of the classes the metal backend's linear layers compute "
                                "in (F32 FMAs, F16 with F32 sums)",
                                precision);
        Linear linear = forced >= 0 ? (Linear)forced : half ? LINEAR_DEFAULT : LINEAR_F32;
        if (linear != LINEAR_F32) {
            TRY(f16_weights(m, err));
            if (m->f16.empty()) {
                if (forced >= 0 || !f32)
                    return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 3,
                                        "precision %u: a linear layer's weight of this model is past F16's range "
                                        "(65504), and the F32 kernels it needs are outside the precision's numerics",
                                        precision);
                linear = LINEAR_F32;
            }
        }
        // Attention's kernels are F16 ones, and run where the linear
        // layers' do and the heads are ones the matrix kernel takes.
        if (forced_attn >= 0 && !(linear != LINEAR_F32 && wide_heads(d.hidden / d.heads)))
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 3,
                                "TURBO_METAL_CHOICES attn=%s: its kernels compute in F16 with F32 sums, for heads "
                                "of a multiple of 8 up to 64, and this session's attention runs in F32 (heads of %u)",
                                ATTN_NAMES[forced_attn], d.hidden / d.heads);

        Session *s = make<Session>();
        s->model = m;
        s->ctx = c;
        s->max_batch = max_batch;
        s->max_seq = max_seq;
        s->cap = (uint32_t)cap;
        s->max_ranges = max_ranges;
        s->rows = new_buffer(c->device, 4 * ints + pairs + block_pairs + triples, MTLResourceStorageModeShared);
        s->scratch = new_buffer(c->device, 6 * wide + ffn + xh, MTLResourceStorageModePrivate);
        s->queue = [c->device newCommandQueue];
        {
            TestPeaks &tp = test_peaks();
            std::lock_guard<std::mutex> g(tp.lock);
            if (tp.on) s->peaks = new_buffer(c->device, PEAK_SLOTS * sizeof(uint32_t), MTLResourceStorageModeShared);
        }
        // The row path and the range path write their vectors to this one
        // buffer, so it holds the larger of the two counts. max_ranges is
        // at most max_batch * max_seq, so this is no more than the scratch
        // checked above.
        const uint64_t vectors = (uint64_t)std::max(max_batch, max_ranges) * d.hidden * 4;
        id<MTLBuffer> output = new_buffer(c->device, vectors, MTLResourceStorageModeShared);
        if (!s->rows || !s->scratch || !output || !s->queue) {
            delete s;
            return refuse(err, TURBO_E_OUT_OF_MEMORY, "%llu bytes for the session", (unsigned long long)(
                          4 * ints + pairs + block_pairs + triples + 6 * wide + ffn + xh + vectors));
        }
        s->ids = 0;
        s->types = ints;
        s->mask = 2 * ints;
        s->pos = 3 * ints;
        s->starts = 4 * ints;
        s->blocks = 4 * ints + pairs;
        s->ranges = 4 * ints + pairs + block_pairs;
        s->x = 0;
        s->q = wide;
        s->k = 2 * wide;
        s->v = 3 * wide;
        s->att = 4 * wide;
        s->tmp = 5 * wide;
        s->ffn = 6 * wide;
        s->xh = 6 * wide + ffn;
        s->output.ctx = c;
        s->output.mtl = output;
        s->output.host = output.contents;
        s->output.placement = TURBO_PLACE_SHARED;
        s->output.bytes = vectors;
        s->output.owned = false;
        s->linear = linear;
        s->forced = forced >= 0;
        s->attn = forced_attn >= 0 ? (Attn)forced_attn : ATTN_DEFAULT;
        s->attn_forced = forced_attn >= 0;
        *compute_dtype = linear != LINEAR_F32 ? TURBO_DTYPE_F16 : TURBO_DTYPE_F32;
        *out = s;
        return TURBO_OK;
    });
}

int32_t session_create(void *model, uint32_t task, uint32_t max_batch, uint32_t max_seq, uint32_t precision,
                       uint32_t *compute_dtype, void **out, turbo_error *err) {
    return open_session(model, task, max_batch, max_seq, precision, max_batch, numerics_for(precision),
                        compute_dtype, out, err);
}

/* The session's whole description, then its linear layers' kernels
 * chosen as docs/metal.md says (Kernel choices) and the choice reported
 * in tuning. Its device is a GPU, so cpu_threads is 0 alone. */
void session_release(void *session);
int32_t tune(Session &s, turbo_backend_tuning &t, turbo_error *err);

int32_t session_open(void *model, const turbo_backend_session_desc *desc, turbo_backend_tuning *tuning,
                     uint32_t *compute_dtype, void **out, turbo_error *err) {
    (void)tuning;
    const size_t need = offsetof(turbo_backend_session_desc, max_ranges) + sizeof(uint32_t);
    if (desc->struct_size < need)
        return refuse(err, TURBO_E_INVALID_STRUCT_SIZE,
                      "turbo_backend_session_desc of %u bytes: the metal backend reads it through max_ranges, %zu "
                      "bytes",
                      desc->struct_size, need);
    if (desc->cpu_threads != 0)
        return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 6,
                            "cpu_threads %u: the metal backend's device is not a CPU; cpu_threads must be 0",
                            desc->cpu_threads);
    const uint32_t numerics = tuning ? tuning->numerics_allowed : numerics_for(desc->precision);
    const int32_t rc = open_session(model, desc->task, desc->max_batch, desc->max_seq, desc->precision,
                                    desc->max_ranges, numerics, compute_dtype, out, err);
    if (rc != TURBO_OK || !tuning) return rc;
    const int32_t tuned = guarded(err, [&]() -> int32_t { return tune(*static_cast<Session *>(*out), *tuning, err); });
    if (tuned != TURBO_OK) {
        session_release(*out);
        *out = nullptr;
    }
    return tuned;
}

void session_release(void *session) {
    @autoreleasepool {
        try {
            delete static_cast<Session *>(session);
        } catch (...) {
        }
    }
}

/* The rows, packed into the session, and the options: see above. Leaves
 * the ranges for the caller to set. */
void pack(Session &s, const turbo_backend_embed_rows *r) {
    char *base = static_cast<char *>(s.rows.contents);
    int32_t *ids = reinterpret_cast<int32_t *>(base + s.ids);
    int32_t *types = reinterpret_cast<int32_t *>(base + s.types);
    int32_t *mask = reinterpret_cast<int32_t *>(base + s.mask);
    int32_t *pos = reinterpret_cast<int32_t *>(base + s.pos);
    uint32_t *starts = reinterpret_cast<uint32_t *>(base + s.starts);
    uint32_t *blocks = reinterpret_cast<uint32_t *>(base + s.blocks);
    uint32_t t = 0, nb = 0, longest = 0;
    const int32_t pos0 = (int32_t)s.model->desc.position_offset;
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
        for (uint32_t p = 0; p < len; p++) pos[t + p] = (int32_t)p + pos0;
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
}

int32_t embed_write(void *session, const turbo_backend_embed_rows *r, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Session &s = *static_cast<Session *>(session);
        pack(s, r);
        s.nranges = 0;
        return TURBO_OK;
    });
}

/* The rows as embed_write packs them, and the ranges copied after the
 * blocks, where pool_ranges reads them. The core has checked each range
 * and count against max_ranges, and set pooling to MEAN. */
int32_t embed_write_ranges(void *session, const turbo_backend_embed_rows *r, const turbo_token_range *ranges,
                           uint32_t count, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Session &s = *static_cast<Session *>(session);
        if (count == 0 || count > s.max_ranges)
            return refuse(err, TURBO_E_INTERNAL, "%u ranges for a metal session of max_ranges %u", count,
                          s.max_ranges);
        pack(s, r);
        memcpy(static_cast<char *>(s.rows.contents) + s.ranges, ranges, (size_t)count * sizeof(turbo_token_range));
        s.nranges = count;
        s.pooling = TURBO_POOLING_MEAN;
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
    // At FASTEST the weights are F16, and the feed-forward block's middle,
    // GELU's output, is F16 too: x_half and y_half say which side is.
    const bool f16 = s.linear != LINEAR_F32;
    // At FASTEST with heads the matrix kernel takes, Q, K, V and the
    // context are F16 too.
    const bool f16_att = f16 && wide_heads(hd);
    const bool registers =
        f16_att && (s.attn == ATTN_REGISTERS || (s.attn == ATTN_AUTO && tokens > ATTN_REGISTERS_TOKENS));
    const std::vector<Ref> &lw = f16 ? s.model->f16 : w;
    auto layer = [&](uint32_t l, int r) -> const Ref & {
        return w[TURBO_BERT_EMBEDDING_TENSORS + l * TURBO_BERT_LAYER_TENSORS + r];
    };
    auto weight = [&](uint32_t l, int r) -> const Ref * {
        return &lw[TURBO_BERT_EMBEDDING_TENSORS + l * TURBO_BERT_LAYER_TENSORS + r];
    };
    auto per_token = [&]() {
        [enc dispatchThreadgroups:MTLSizeMake(tokens / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    };
    // Up to three layers of one shape, over the same input, in one
    // dispatch. A single layer with no epilogue whose tiles would leave
    // most of the GPU idle has its k split, so more threadgroups share the
    // work; the splits land side by side in its output, which holds
    // capacity rows, and add_ln sums them. Returns the splits.
    auto gemm = [&](uint64_t x, uint32_t n_in, uint32_t n_out, uint32_t epilogue,
                    std::initializer_list<std::pair<const Ref *, uint64_t>> outs,
                    std::initializer_list<const Ref *> biases, bool x_half = false, bool y_half = false) -> uint32_t {
        // Few tokens: the small kernel, over all of k.
        const bool small = tokens <= (f16 ? SMALL_TOKENS_F16 : SMALL_TOKENS) && n_out % 16 == 0;
        const Linear tile =
            s.linear != LINEAR_AUTO ? s.linear : tokens <= AUTO_H32_TOKENS ? LINEAR_H32N32 : LINEAR_H64;
        const uint32_t bm = tile == LINEAR_H128 ? 128 : tile == LINEAR_H64 ? 64 : 32, bk = f16 ? 32 : 16;
        const uint32_t bn = f16 && tile == LINEAR_H32N32 ? 32 : 64;
        const uint32_t tiles = (n_out + bn - 1) / bn * ((tokens + bm - 1) / bm);
        uint32_t splits = 1;
        if (!small && outs.size() == 1 && epilogue == EPILOGUE_NONE && tiles < SPREAD) {
            splits = std::min({(SPREAD + tiles - 1) / tiles, n_in / 64, s.cap / tokens, 8u});
            splits = std::max(splits, 1u);
        }
        const uint32_t kchunk = (uint32_t)round_up((n_in + splits - 1) / splits, bk);
        splits = (n_in + kchunk - 1) / kchunk;
        const GemmParams p{tokens, n_out, n_in, epilogue, splits, kchunk};
        int kernel = small ? GEMM_SMALL : GEMM;
        if (f16) {
            const int types = (x_half ? 2 : 0) + (y_half ? 1 : 0);
            kernel = (small                     ? GEMM_SMALL_H
                      : tile == LINEAR_H128 ? GEMM_H128
                      : tile == LINEAR_H32N32 ? GEMM_H32N32
                      : tile == LINEAR_H64  ? GEMM_H64
                                            : GEMM_H32) +
                     types;
        }
        [enc setComputePipelineState:kn.k[kernel]];
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
        if (small)
            [enc dispatchThreadgroups:MTLSizeMake(n_out / 16, tokens / 32, n) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        else
            [enc dispatchThreadgroups:MTLSizeMake((n_out + bn - 1) / bn, (tokens + bm - 1) / bm, n * splits)
                threadsPerThreadgroup:MTLSizeMake(bm == 128 ? 256 : 128, 1, 1)];
        return splits;
    };
    const RowParams rp{h, (float)d.layer_norm_eps, f16 ? 1u : 0u};
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
        [enc setBuffer:sc offset:f16 ? s.xh : s.x atIndex:8];
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
    [enc setBuffer:sc offset:f16 ? s.xh : s.x atIndex:10];
    per_token();

    // For the tests: each F16 tensor's largest magnitude, into its slot.
    auto peak = [&](uint64_t at, uint32_t width, int slot) {
        if (!s.peaks) return;
        const uint32_t n = tokens * width;
        [enc setComputePipelineState:kn.k[ABS_MAX_F16]];
        [enc setBuffer:sc offset:at atIndex:0];
        [enc setBytes:&n length:sizeof n atIndex:1];
        [enc setBuffer:s.peaks offset:slot * sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    const AttentionParams ap{h, hd, 1.0f / sqrtf((float)hd)};
    for (uint32_t l = 0; l < d.layers; l++) {
        gemm(f16 ? s.xh : s.x, h, h, EPILOGUE_BIAS,
             {{weight(l, TURBO_BERT_Q_WEIGHT), s.q}, {weight(l, TURBO_BERT_K_WEIGHT), s.k},
              {weight(l, TURBO_BERT_V_WEIGHT), s.v}},
             {&layer(l, TURBO_BERT_Q_BIAS), &layer(l, TURBO_BERT_K_BIAS), &layer(l, TURBO_BERT_V_BIAS)}, f16,
             f16_att);

        if (f16) peak(s.xh, h, 5);
        if (f16_att) {
            peak(s.q, h, 0);
            peak(s.k, h, 1);
            peak(s.v, h, 2);
        }
        const bool wide = wide_heads(hd);
        [enc setComputePipelineState:kn.k[registers ? ATTENTION_HR
                                          : f16_att ? ATTENTION_H
                                          : wide    ? ATTENTION
                                                    : ATTENTION_NARROW]];
        [enc setBuffer:sc offset:s.q atIndex:0];
        [enc setBuffer:sc offset:s.k atIndex:1];
        [enc setBuffer:sc offset:s.v atIndex:2];
        [enc setBuffer:rows offset:s.mask atIndex:3];
        [enc setBuffer:rows offset:s.starts atIndex:4];
        [enc setBuffer:rows offset:s.blocks atIndex:5];
        [enc setBuffer:sc offset:s.att atIndex:6];
        [enc setBytes:&ap length:sizeof ap atIndex:7];
        [enc setThreadgroupMemoryLength:f16_att ? attention_h_bytes(hd, registers) : attention_bytes(hd, s.longest)
                                atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(s.nblocks, d.heads, 1) threadsPerThreadgroup:MTLSizeMake(wide ? 128 : 32, 1, 1)];

        if (f16_att) peak(s.att, h, 3);
        uint32_t splits =
            gemm(s.att, h, h, EPILOGUE_NONE, {{weight(l, TURBO_BERT_ATTN_OUT_WEIGHT), s.tmp}}, {}, f16_att);
        add_ln(s.tmp, splits, layer(l, TURBO_BERT_ATTN_OUT_BIAS), layer(l, TURBO_BERT_ATTN_LN_WEIGHT),
               layer(l, TURBO_BERT_ATTN_LN_BIAS));
        gemm(f16 ? s.xh : s.x, h, inter, EPILOGUE_BIAS_GELU, {{weight(l, TURBO_BERT_FFN_IN_WEIGHT), s.ffn}},
             {&layer(l, TURBO_BERT_FFN_IN_BIAS)}, f16, f16);
        if (f16) {
            peak(s.xh, h, 5);
            peak(s.ffn, inter, 4);
        }
        splits = gemm(s.ffn, inter, h, EPILOGUE_NONE, {{weight(l, TURBO_BERT_FFN_OUT_WEIGHT), s.tmp}}, {}, f16);
        add_ln(s.tmp, splits, layer(l, TURBO_BERT_FFN_OUT_BIAS), layer(l, TURBO_BERT_FFN_LN_WEIGHT),
               layer(l, TURBO_BERT_FFN_LN_BIAS));
    }

    const PoolParams pp{h, s.output_dim, s.pooling, s.normalize == TURBO_NORMALIZE_L2 ? 1u : 0u};
    if (s.nranges > 0) {
        // One SIMD group per range, after embed_write_ranges.
        [enc setComputePipelineState:kn.k[POOL_RANGES]];
        [enc setBuffer:sc offset:s.x atIndex:0];
        [enc setBuffer:rows offset:s.mask atIndex:1];
        [enc setBuffer:rows offset:s.starts atIndex:2];
        [enc setBuffer:rows offset:s.ranges atIndex:3];
        [enc setBuffer:s.output.mtl offset:0 atIndex:4];
        [enc setBytes:&pp length:sizeof pp atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(s.nranges, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        return;
    }
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
            if (s.peaks) memset(s.peaks.contents, 0, PEAK_SLOTS * sizeof(uint32_t));
            encode(s, enc);
            [enc endEncoding];
            TRY(finish(cb, err, "the run"));
        }
        if (s.peaks) {
            TestPeaks &tp = test_peaks();
            std::lock_guard<std::mutex> g(tp.lock);
            memcpy(tp.last, s.peaks.contents, sizeof tp.last);
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


// ---- Kernel choices --------------------------------------------------------------
//
// At FASTEST a session's linear layers run in F16 tiles of 32 x 64 (h32),
// 64 x 64 (h64), 128 x 64 (h128) or 32 x 32 (h32n32) past
// SMALL_TOKENS_F16, or in the tile each run's tokens favor (auto); EXACT and MODEL have one path, F32. A
// tuned session times each when it is made, on its own memory, at each
// token bin it can hold, and keeps the incumbent unless another is faster
// at every one; docs/metal.md says how (Autotuning).

/* The packed tokens the tuner times each choice at, one per token bin up
 * to the session's capacity, and the capacity when it lies past the last;
 * at most TUNE_TOKENS in all. */
constexpr uint32_t TUNE_BIN_TOKENS[] = {256, 1024, 4096, 8192};
constexpr uint32_t TUNE_TOKENS = 8192;
/* Timed runs of each variant, after an untimed one. */
constexpr size_t TUNE_RUNS = 5;
/* A variant replaces the incumbent only when this much faster. */
constexpr double TUNE_MARGIN = 0.05;
/* Times whose median is more than this far above their least are apart:
 * the clocks still coming up, or a GPU shared or throttling. */
constexpr double TUNE_APART = 0.25;
/* How many times a variant whose times are apart is timed in all. */
constexpr int TUNE_TRIES = 3;

void write_choices(char *dst, const std::string &text) {
    const size_t n = std::min(text.size(), (size_t)TURBO_CHOICES_LEN - 1);
    memcpy(dst, text.data(), n);
    memset(dst + n, 0, TURBO_CHOICES_LEN - n);
}

/* A FASTEST session's line: its linear layers', then, where attention
 * runs in F16, attention's, then what the environment forced. */
std::string choice_line(const Session &s) {
    const turbo_backend_model &d = s.model->desc;
    std::string line = std::string("linear=") + LINEAR_NAMES[s.linear];
    if (s.linear != LINEAR_F32 && wide_heads(d.hidden / d.heads)) line += std::string(";attn=") + ATTN_NAMES[s.attn];
    line += ";forced=";
    if (s.forced) line += "linear";
    if (s.attn_forced) line += s.forced ? ",attn" : "attn";
    return line;
}

/* The linear layers a choices line names, or -1. */
int parse_linear(const char *line) {
    for (int l = LINEAR_F32; l < (int)LINEAR_COUNT; l++) {
        const std::string item = std::string("linear=") + LINEAR_NAMES[l];
        const char *at = strstr(line, item.c_str());
        if (at && (at == line || at[-1] == ';') && (at[item.size()] == ';' || at[item.size()] == 0)) return l;
    }
    return -1;
}

/* One run of the encoder over what the session holds, waited for: its
 * time from the host in ms. */
int32_t timed_run(Session &s, double *ms, turbo_error *err) {
    const auto t0 = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> cb = [s.queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    encode(s, enc);
    [enc endEncoding];
    TRY(finish(cb, err, "a tuning run"));
    *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return TURBO_OK;
}

int32_t tune(Session &s, turbo_backend_tuning &t, turbo_error *err) {
    t.tuned = TURBO_TUNED_DEFAULT;
    t.tune_ms = 0;
    t.numerics_used = t.numerics_allowed;
    Context *c = s.ctx;
    if (s.forced || s.attn_forced) {
        write_choices(t.choices, choice_line(s));
        t.tuned = TURBO_TUNED_FORCED;
        return TURBO_OK;
    }
    // One path: nothing to choose, nothing reported.
    if (s.linear == LINEAR_F32) {
        write_choices(t.choices, "");
        return TURBO_OK;
    }
    const int cached = t.cached ? parse_linear(t.cached) : -1;
    const bool usable = cached > (int)LINEAR_F32 && runs_linear(c, (Linear)cached);
    if (t.mode == TURBO_AUTOTUNE_ON && usable) {
        s.linear = (Linear)cached;
        write_choices(t.choices, choice_line(s));
        t.tuned = TURBO_TUNED_CACHE;
        return TURBO_OK;
    }
    if (t.mode != TURBO_AUTOTUNE_ON && t.mode != TURBO_AUTOTUNE_RETUNE) {
        write_choices(t.choices, choice_line(s));
        return TURBO_OK;
    }
    // The incumbent first, then every other F16 tiling the device runs.
    const Linear incumbent = usable ? (Linear)cached : s.linear;
    std::vector<Linear> order{incumbent};
    for (int l = LINEAR_H32; l < (int)LINEAR_COUNT; l++)
        if (l != (int)incumbent && runs_linear(c, (Linear)l)) order.push_back((Linear)l);

    // The token counts to time at: each bin's in TUNE_BIN_TOKENS the session
    // can hold, and its capacity (up to TUNE_TOKENS) when that lies past
    // the last of them.
    const auto started = std::chrono::steady_clock::now();
    const uint32_t capacity = (uint32_t)std::min<uint64_t>((uint64_t)s.max_batch * s.max_seq, TUNE_TOKENS);
    std::vector<uint32_t> counts;
    for (uint32_t n : TUNE_BIN_TOKENS)
        if (n <= capacity) counts.push_back(n);
    if (counts.empty() || counts.back() < capacity) counts.push_back(capacity);
    static const char *const BINS[] = {"le256", "le1k", "le4k", "le16k", "gt16k"};

    const double budget = t.budget_ms;
    double spent = 0;
    // least[b][v]: choice v's least time at count b; timed[b][v] whether
    // it was timed there.
    std::vector<std::vector<double>> least(counts.size(), std::vector<double>(order.size(), 0));
    std::vector<std::vector<bool>> timed(counts.size(), std::vector<bool>(order.size(), false));
    std::vector<uint32_t> bins(counts.size(), 0);
    // A challenger stays in only while it beats the incumbent by the margin
    // at every count timed so far, since it must at all of them to win:
    // the larger counts, the costly ones, time the incumbent and those
    // left, and none left settles the choice.
    std::vector<bool> alive(order.size(), true);
    size_t left = order.size() - 1;
    for (size_t b = 0; b < counts.size() && spent < budget && left > 0; b++) {
        // Rows of up to max_seq tokens, ids 1, as many as make counts[b].
        const uint32_t len = std::min(s.max_seq, counts[b]);
        const uint32_t batch = std::clamp(counts[b] / len, 1u, s.max_batch);
        std::vector<int32_t> ones((size_t)batch * len, 1);
        turbo_backend_embed_rows rows{};
        rows.struct_size = sizeof rows;
        rows.batch = batch;
        rows.seq = len;
        rows.row_stride = len;
        rows.ids = ones.data();
        rows.mask = ones.data();
        rows.pooling = TURBO_POOLING_MEAN;
        rows.normalize = TURBO_NORMALIZE_NONE;
        rows.output_dim = s.model->desc.hidden;
        pack(s, &rows);
        s.nranges = 0;
        s.written = false;
        for (uint32_t edge : {256u, 1024u, 4096u, 16384u}) bins[b] += s.tokens > edge;
        // Timed runs per choice here: as many as the budget left shares
        // out over the counts and choices left, by the incumbent's untimed
        // run, two at least and TUNE_RUNS at most.
        size_t runs = TUNE_RUNS;
        for (size_t v = 0; v < order.size(); v++) {
            if (spent >= budget) break;
            if (v > 0 && !alive[v]) continue;
            s.linear = order[v];
            double ms = 0;
            TRY(timed_run(s, &ms, err));   // untimed: the clocks come up
            if (v == 0) {
                const double share = (budget - spent) / (double)(counts.size() - b) / (double)(left + 1);
                runs = std::clamp<size_t>((size_t)(share / std::max(ms, 1e-3)), 2, TUNE_RUNS);
            }
            bool apart = true;
            for (int tries = 0; tries < TUNE_TRIES && apart; tries++) {
                std::vector<double> times;
                while (times.size() < runs && spent < budget) {
                    TRY(timed_run(s, &ms, err));
                    spent += ms;
                    times.push_back(ms);
                }
                if (times.empty()) break;
                std::sort(times.begin(), times.end());
                least[b][v] = times[0];
                timed[b][v] = true;
                apart = times.size() > 1 && times[times.size() / 2] > times[0] * (1.0 + TUNE_APART);
            }
            if (apart && timed[b][v]) {
                s.linear = incumbent;
                c->say(LOG_INFO,
                       "metal device %u: not tuned: linear=%s's times at %s stayed more than %.0f%% apart, so the GPU "
                       "is shared or throttling; the next session measures again",
                       c->ordinal, LINEAR_NAMES[order[v]], BINS[bins[b]], TUNE_APART * 100);
                write_choices(t.choices, choice_line(s));
                return TURBO_OK;
            }
        }
        for (size_t v = 1; v < order.size(); v++) {
            if (!alive[v]) continue;
            alive[v] = timed[b][0] && timed[b][v] && least[b][v] < least[b][0] * (1.0 - TUNE_MARGIN);
            if (!alive[v]) left--;
        }
    }
    if (!timed[0][0]) return refuse(err, TURBO_E_INVALID_STATE, "metal: the tuner timed no incumbent");
    // A challenger replaces the incumbent only when both were timed at
    // every count and it beat the incumbent by the margin at each; of
    // those, the one least in all. A count past the budget leaves the
    // incumbent in place, and so does a challenger dropped at a count.
    size_t best = 0;
    double best_total = 0;
    for (size_t v = 1; v < order.size(); v++) {
        bool wins = true;
        double total = 0;
        for (size_t b = 0; b < counts.size() && wins; b++) {
            wins = timed[b][0] && timed[b][v] && least[b][v] < least[b][0] * (1.0 - TUNE_MARGIN);
            total += least[b][v];
        }
        if (wins && (best == 0 || total < best_total)) {
            best = v;
            best_total = total;
        }
    }
    s.linear = order[best];
    // Why the larger counts were not timed, if they were not: every
    // challenger dropped out, or the budget ran out.
    bool cut = false;
    for (size_t b = 0; b < counts.size(); b++) cut = cut || !timed[b][0];
    const char *why = !cut ? "" : left == 0 ? "; the rest settled, no challenger left" : "; the rest past the budget";
    const uint32_t tune_ms = (uint32_t)std::min<double>(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(), UINT32_MAX);
    std::string lines, said;
    for (size_t b = 0; b < counts.size(); b++) {
        if (!timed[b][0]) continue;
        said += std::string(said.empty() ? "" : "; ") + BINS[bins[b]] + ":";
        for (size_t v = 0; v < order.size(); v++) {
            if (!timed[b][v]) continue;
            char one[96];
            snprintf(one, sizeof one, "%s%s/linear/%s=%.4f", lines.empty() ? "" : "\n", BINS[bins[b]],
                     LINEAR_NAMES[order[v]], least[b][v]);
            lines += one;
            snprintf(one, sizeof one, " %s %.3f ms", LINEAR_NAMES[order[v]], least[b][v]);
            said += one;
        }
    }
    if (t.timings && t.timings_len > 0) {
        const size_t n = std::min(lines.size(), (size_t)t.timings_len - 1);
        memcpy(t.timings, lines.data(), n);
        t.timings[n] = 0;
    }
    c->say(LOG_INFO, "metal device %u: linear layers chosen in %u ms: %s (a run at %s%s)", c->ordinal, tune_ms,
           LINEAR_NAMES[s.linear], said.c_str(), why);
    write_choices(t.choices, choice_line(s));
    t.tune_ms = tune_ms;
    t.tuned = TURBO_TUNED_MEASURED;
    return TURBO_OK;
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
    TURBO_FORMAT_BIT(TURBO_FORMAT_SAFETENSORS),
    0,
    nullptr,
    session_open,
    embed_write_ranges,
    0,
    0,
};

/* Every allocation this backend has made in the process, host and device,
 * for the tests that hold a warm run to none. */
void turbo_metal_allocations(uint64_t *host, uint64_t *device) {
    *host = host_total.load(std::memory_order_relaxed);
    *device = device_total.load(std::memory_order_relaxed);
}

/* For the tests: whether sessions made after this track the largest
 * magnitude of each F16 tensor their runs store, and the last tracked
 * run's, PEAK_SLOTS floats in the order PEAK_SLOTS says. */
void turbo_metal_track_peaks(int on) {
    try {
        TestPeaks &t = test_peaks();
        std::lock_guard<std::mutex> g(t.lock);
        t.on = on != 0;
    } catch (...) {
    }
}

void turbo_metal_peaks(float *out) {
    try {
        TestPeaks &t = test_peaks();
        std::lock_guard<std::mutex> g(t.lock);
        memcpy(out, t.last, sizeof t.last);
    } catch (...) {
    }
}

/* A choices line read in place of TURBO_METAL_CHOICES by the sessions
 * made after it, or NULL to read the variable again: for the tests. */
void turbo_metal_use_choices(const char *line) {
    try {
        TestChoices &t = test_choices();
        std::lock_guard<std::mutex> g(t.lock);
        t.set = line != nullptr;
        t.line = line ? line : "";
    } catch (...) {
    }
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
