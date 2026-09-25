/* SPDX-License-Identifier: Apache-2.0
 *
 * The CUDA backend: NVIDIA GPUs through the CUDA runtime and cuBLAS,
 * behind include/turbo/turbo_backend.h. It lists the devices the driver
 * reports, keeps a stream and a cuBLAS handle per context, holds models in
 * device memory, and runs embed sessions with the BERT encoder in
 * kernels.cu and cuBLAS's GEMMs: single precision for an F32 session,
 * F16 inputs with F32 accumulation on the tensor cores for an F16 one.
 *
 * Every function here is called from any thread. A context's stream, its
 * cuBLAS handle and the handle's workspace are used under the context's
 * lock, which a run holds from its first launch to the stream's
 * synchronization. The calling thread's current device is set for the
 * call and put back after it, so a caller that uses CUDA itself finds its
 * device where it left it.
 */

#include <turbo/turbo_backend.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <strings.h>
#include <utility>
#include <vector>

#include "kernels.h"

/* TURBO_CUDA_ARCHS: the architectures build.rs compiled for, as
 * TURBO_CUDA_ARCH gave them. */
#include "turbo_cuda_build.h"

using namespace turbo_cuda;

namespace {

// ---- Errors ------------------------------------------------------------------
//
// Nothing throws across the table: each entry runs its body inside
// guarded(), which turns an exception into a status.

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

/* A runtime failure, with the runtime's name and text for it. An error
 * the runtime keeps for the next call is cleared, unless it is sticky. */
int32_t cuda_failed(turbo_error *err, cudaError_t e, const char *what) {
    (void)cudaGetLastError();
    const int32_t code = e == cudaErrorMemoryAllocation ? TURBO_E_OUT_OF_MEMORY : TURBO_E_RUNTIME;
    return refuse(err, code, "%s: %s: %s", what, cudaGetErrorName(e), cudaGetErrorString(e));
}

int32_t cublas_failed(turbo_error *err, cublasStatus_t s, const char *what) {
    const int32_t code = s == CUBLAS_STATUS_ALLOC_FAILED ? TURBO_E_OUT_OF_MEMORY : TURBO_E_RUNTIME;
    return refuse(err, code, "%s: %s: %s", what, cublasGetStatusName(s), cublasGetStatusString(s));
}

#define TRY_CUDA(expr, what)                                                                                           \
    do {                                                                                                               \
        const cudaError_t e_ = (expr);                                                                                 \
        if (e_ != cudaSuccess) return cuda_failed(err, e_, what);                                                      \
    } while (0)

#define TRY_CUBLAS(expr, what)                                                                                         \
    do {                                                                                                               \
        const cublasStatus_t s_ = (expr);                                                                              \
        if (s_ != CUBLAS_STATUS_SUCCESS) return cublas_failed(err, s_, what);                                          \
    } while (0)

#define TRY(expr)                                                                                                      \
    do {                                                                                                               \
        const int32_t rc_ = (expr);                                                                                    \
        if (rc_ != TURBO_OK) return rc_;                                                                               \
    } while (0)

template <typename F> int32_t guarded(turbo_error *err, F f) noexcept {
    try {
        return f();
    } catch (const std::bad_alloc &) {
        return refuse(err, TURBO_E_OUT_OF_MEMORY, "host memory for the cuda backend");
    } catch (...) {
        return refuse(err, TURBO_E_INTERNAL, "an exception inside the cuda backend");
    }
}

// ---- Allocation counts ---------------------------------------------------------
//
// Every allocation the backend makes goes through one of the functions
// below, which count it twice: in the process's total, which the tests
// read through turbo_cuda_allocations, and in the calling thread's own
// count, whose change across session_run is what a result reports. Device
// memory, page-locked host memory and managed memory are the driver's and
// count as device allocations; the backend's own structures and pageable
// buffers are host allocations.

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

cudaError_t device_malloc(void **p, size_t n) {
    const cudaError_t e = cudaMalloc(p, n);
    if (e == cudaSuccess) counted_device();
    return e;
}

cudaError_t pinned_malloc(void **p, size_t n) {
    const cudaError_t e = cudaMallocHost(p, n);
    if (e == cudaSuccess) counted_device();
    return e;
}

cudaError_t managed_malloc(void **p, size_t n) {
    const cudaError_t e = cudaMallocManaged(p, n, cudaMemAttachGlobal);
    if (e == cudaSuccess) counted_device();
    return e;
}

/* Every host buffer starts on a 64-byte boundary, as on the CPU backend. */
constexpr size_t HOST_ALIGN = 64;
/* Tensors and scratch in one device allocation each start on 256 bytes,
 * the alignment cudaMalloc gives. */
constexpr size_t DEVICE_ALIGN = 256;

size_t round_up(size_t n, size_t a) { return (n + a - 1) / a * a; }

/* src into dst's len bytes, cut to fit and NUL-terminated. */
void copy_str(char *dst, size_t len, const char *src) {
    const size_t n = strnlen(src, len - 1);
    memcpy(dst, src, n);
    dst[n] = 0;
}

void *host_malloc(size_t n) {
    void *p = aligned_alloc(HOST_ALIGN, round_up(n, HOST_ALIGN));
    if (p) counted_host();
    return p;
}

// ---- Devices -------------------------------------------------------------------

/* The calling thread's current device set to one ordinal while this lives. */
class DeviceScope {
  public:
    explicit DeviceScope(int ordinal) {
        if (cudaGetDevice(&prev_) != cudaSuccess) {
            (void)cudaGetLastError();
            prev_ = -1;
        }
        status_ = prev_ == ordinal ? cudaSuccess : cudaSetDevice(ordinal);
        changed_ = prev_ != ordinal && status_ == cudaSuccess;
    }
    ~DeviceScope() {
        if (changed_ && prev_ >= 0) (void)cudaSetDevice(prev_);
    }
    DeviceScope(const DeviceScope &) = delete;
    DeviceScope &operator=(const DeviceScope &) = delete;
    cudaError_t status() const { return status_; }

  private:
    int prev_ = -1;
    cudaError_t status_ = cudaSuccess;
    bool changed_ = false;
};

#define ON_DEVICE(ordinal)                                                                                             \
    DeviceScope scope_(ordinal);                                                                                       \
    TRY_CUDA(scope_.status(), "cudaSetDevice")

/* CUDA versions are major * 1000 + minor * 10. */
void write_version(char *dst, size_t len, int v) { snprintf(dst, len, "%d.%d", v / 1000, (v % 1000) / 10); }

/* The runtime this build's headers are, as a static string: the soname
 * it links is libcudart.so.<major>. */
struct VersionText {
    char s[16];
};

constexpr VersionText version_text(int v) {
    VersionText t{};
    int i = 0;
    const int parts[2] = {v / 1000, (v % 1000) / 10};
    for (int k = 0; k < 2; k++) {
        const int n = parts[k];
        if (n >= 100) t.s[i++] = (char)('0' + n / 100);
        if (n >= 10) t.s[i++] = (char)('0' + n / 10 % 10);
        t.s[i++] = (char)('0' + n % 10);
        if (k == 0) t.s[i++] = '.';
    }
    t.s[i] = 0;
    return t;
}

constexpr VersionText LINKED_RUNTIME = version_text(CUDART_VERSION);

/* The label benchmark records for a device are filed under (turbo.h's
 * turbo_device_info.arch), from the name the driver gives it: the words
 * of the name, split at spaces and hyphens, without the vendor's and
 * brand's words (NVIDIA, GeForce, Tesla, Quadro, GPU, Generation), up to
 * the first word that gives a memory size or a form factor (80GB, PCIe,
 * SXM4, HBM3, NVL), in lower case with only letters and digits:
 * "NVIDIA GeForce RTX 4080" is rtx4080, "NVIDIA A100-SXM4-80GB" is a100,
 * "NVIDIA GeForce RTX 4070 Ti SUPER" is rtx4070tisuper,
 * "NVIDIA RTX 6000 Ada Generation" is rtx6000ada, "Tesla T4" is t4. */

bool word_is(const char *w, size_t n, const char *lit) { return strlen(lit) == n && strncasecmp(w, lit, n) == 0; }

/* A memory size (80GB, 16G) or a form factor: where a model name ends. */
bool ends_name(const char *w, size_t n) {
    if (n >= 2 && isdigit((unsigned char)w[0])) {
        if (tolower((unsigned char)w[n - 1]) == 'b' && tolower((unsigned char)w[n - 2]) == 'g') return true;
        if (tolower((unsigned char)w[n - 1]) == 'g') return true;
    }
    return strncasecmp(w, "pcie", 4) == 0 || strncasecmp(w, "sxm", 3) == 0 || strncasecmp(w, "hbm", 3) == 0 ||
           word_is(w, n, "nvl");
}

void arch_label(const char *name, char *out, size_t len) {
    if (len == 0) return;
    size_t o = 0;
    const char *p = name;
    while (*p) {
        while (*p == ' ' || *p == '-') p++;
        const char *w = p;
        while (*p && *p != ' ' && *p != '-') p++;
        const size_t n = (size_t)(p - w);
        if (n == 0) break;
        if (word_is(w, n, "nvidia") || word_is(w, n, "geforce") || word_is(w, n, "tesla") || word_is(w, n, "quadro") ||
            word_is(w, n, "gpu") || word_is(w, n, "generation"))
            continue;
        if (ends_name(w, n)) break;
        for (size_t i = 0; i < n && o + 1 < len; i++) {
            const unsigned char c = (unsigned char)w[i];
            if (isalnum(c)) out[o++] = (char)tolower(c);
        }
    }
    out[o] = 0;
}

/* The kernel module's version from /proc, where Linux has one: the first
 * word of its line that starts with digits and a dot. */
bool kernel_module_version(char *out, size_t len) {
    FILE *f = fopen("/proc/driver/nvidia/version", "r");
    if (!f) return false;
    char line[256];
    bool found = false;
    if (fgets(line, sizeof line, f)) {
        char *save = nullptr;
        for (char *w = strtok_r(line, " \t\n", &save); w; w = strtok_r(nullptr, " \t\n", &save)) {
            if (isdigit((unsigned char)w[0]) && strchr(w, '.')) {
                copy_str(out, len, w);
                found = true;
                break;
            }
        }
    }
    fclose(f);
    return found;
}

int32_t device_count(uint32_t *out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        int n = 0;
        const cudaError_t e = cudaGetDeviceCount(&n);
        if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver) {
            (void)cudaGetLastError();
            int driver = 0;
            if (e == cudaErrorInsufficientDriver && cudaDriverGetVersion(&driver) == cudaSuccess && driver > 0) {
                char d[16], r[16];
                write_version(d, sizeof d, driver);
                write_version(r, sizeof r, CUDART_VERSION);
                return refuse(err, TURBO_E_DEVICE_UNAVAILABLE,
                              "the driver runs CUDA %s, older than the %s runtime this build linked", d, r);
            }
            // No driver, or no device: nothing to list.
            *out = 0;
            return TURBO_OK;
        }
        if (e != cudaSuccess) {
            (void)cudaGetLastError();
            return refuse(err, TURBO_E_DEVICE_UNAVAILABLE, "cudaGetDeviceCount: %s: %s", cudaGetErrorName(e),
                          cudaGetErrorString(e));
        }
        *out = (uint32_t)n;
        return TURBO_OK;
    });
}

int32_t device_info(uint32_t ordinal, turbo_device_info *out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        cudaDeviceProp p;
        TRY_CUDA(cudaGetDeviceProperties(&p, (int)ordinal), "cudaGetDeviceProperties");
        size_t free_bytes = 0, total_bytes = 0;
        {
            ON_DEVICE((int)ordinal);
            TRY_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
        }
        out->kind = p.integrated ? TURBO_DEVICE_IGPU : TURBO_DEVICE_GPU;
        out->ordinal = ordinal;
        out->unified_memory = p.integrated ? 1 : 0;
        out->memory_total = total_bytes;
        out->memory_free = free_bytes;
        arch_label(p.name, out->arch, sizeof out->arch);
        copy_str(out->name, sizeof out->name, p.name);
        copy_str(out->vendor, sizeof out->vendor, "NVIDIA");
        int runtime = 0, driver = 0;
        TRY_CUDA(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
        TRY_CUDA(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
        write_version(out->runtime_version, sizeof out->runtime_version, runtime);
        char module[32], cuda[16];
        write_version(cuda, sizeof cuda, driver);
        if (kernel_module_version(module, sizeof module))
            snprintf(out->driver_version, sizeof out->driver_version, "%s, CUDA %s", module, cuda);
        else
            snprintf(out->driver_version, sizeof out->driver_version, "CUDA %s", cuda);
        return TURBO_OK;
    });
}

/* Fields of turbo_embed_options a run honors: normalize (4), pooling (5)
 * and output_dim (6), every value of each. */
constexpr uint32_t EMBED_HONORED = 0x38;

/* Embed at every precision: MODEL and EXACT in F32, FASTEST in F16 (F16
 * GEMM inputs, F32 accumulation, and everything else in F32). As on the
 * CPU, a model stored in F16 or BF16 computes in F32 at EXACT from a
 * converted copy, and its session at MODEL is refused. A model with a
 * GEMM weight past F16's range computes in F32 at FASTEST too, and
 * turbo_session_get_info says so for that model. A device the build has
 * no code for runs nothing. */
int32_t capability(uint32_t ordinal, uint32_t, uint32_t precision, uint32_t *status, uint32_t *dtype,
                   uint32_t *options_honored, char *reason, uint32_t reason_len, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        ON_DEVICE((int)ordinal);
        const cudaError_t e = kernels_run_here();
        if (e == cudaErrorNoKernelImageForDevice || e == cudaErrorInvalidDeviceFunction ||
            e == cudaErrorUnsupportedPtxVersion) {
            (void)cudaGetLastError();
            cudaDeviceProp p;
            TRY_CUDA(cudaGetDeviceProperties(&p, (int)ordinal), "cudaGetDeviceProperties");
            *status = TURBO_CAP_UNSUPPORTED;
            *dtype = 0;
            *options_honored = 0;
            if (reason_len)
                snprintf(reason, reason_len, "built for sm %s (TURBO_CUDA_ARCH); this device is sm %d%d: %s",
                         TURBO_CUDA_ARCHS, p.major, p.minor, cudaGetErrorName(e));
            return TURBO_OK;
        }
        TRY_CUDA(e, "cudaFuncGetAttributes");
        *status = TURBO_CAP_EXPERIMENTAL;
        *dtype = precision == TURBO_PRECISION_FASTEST ? TURBO_DTYPE_F16 : TURBO_DTYPE_F32;
        *options_honored = EMBED_HONORED;
        if (reason_len) reason[0] = 0;
        return TURBO_OK;
    });
}

// ---- Contexts and buffers ------------------------------------------------------
//
// A context is a stream and a cuBLAS handle on one device. The handle
// computes on the stream with TF32 off (CUBLAS_DEFAULT_MATH: TF32 would
// round an F32 GEMM's inputs to 10 bits of mantissa), so an F32 GEMM is
// F32; an F16 session's GEMMs take F16 inputs and accumulate in F32
// (CUBLAS_COMPUTE_32F), which the default math mode runs on the tensor
// cores. The handle has a fixed workspace of its own, so a GEMM never
// allocates.
//
// Placements: DEVICE is cudaMalloc memory, with no host address; PINNED is
// page-locked host memory (cudaMallocHost); HOST is pageable, 64-byte
// aligned host memory; SHARED is managed memory (cudaMallocManaged), one
// address for both.

/* cuBLAS's own default on the newest devices, and more than an F32 GEMM
 * on any of them asks for. */
constexpr size_t CUBLAS_WORKSPACE = 32u << 20;

struct Context {
    int ordinal = 0;
    turbo_log_fn log = nullptr;
    void *log_user_data = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    void *workspace = nullptr;
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

constexpr uint32_t LOG_WARNING = 1;
constexpr uint32_t LOG_DEBUG = 3;

void release_context(Context *c) {
    DeviceScope scope(c->ordinal);
    if (c->blas) cublasDestroy(c->blas);
    if (c->stream) cudaStreamDestroy(c->stream);
    if (c->workspace) cudaFree(c->workspace);
    delete c;
}

int32_t context_create(uint32_t ordinal, turbo_log_fn log, void *log_user_data, void **out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        ON_DEVICE((int)ordinal);
        Context *c = make<Context>();
        c->ordinal = (int)ordinal;
        c->log = log;
        c->log_user_data = log_user_data;
        const int32_t rc = [&]() -> int32_t {
            cudaDeviceProp p;
            TRY_CUDA(cudaGetDeviceProperties(&p, (int)ordinal), "cudaGetDeviceProperties");
            TRY_CUDA(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
            TRY_CUDA(device_malloc(&c->workspace, CUBLAS_WORKSPACE), "the cuBLAS workspace");
            TRY_CUBLAS(cublasCreate(&c->blas), "cublasCreate");
            // cublasSetStream puts the handle back on cuBLAS's own
            // workspace pool, so the workspace is set after it.
            TRY_CUBLAS(cublasSetStream(c->blas, c->stream), "cublasSetStream");
            TRY_CUBLAS(cublasSetWorkspace(c->blas, c->workspace, CUBLAS_WORKSPACE), "cublasSetWorkspace");
            TRY_CUBLAS(cublasSetMathMode(c->blas, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
            TRY_CUBLAS(cublasSetPointerMode(c->blas, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
            c->say(
                LOG_DEBUG,
                "cuda context on device %u (%s, sm %d%d): a stream, cuBLAS in F32 without TF32, %zu MiB of workspace",
                ordinal, p.name, p.major, p.minor, CUBLAS_WORKSPACE >> 20);
            return TURBO_OK;
        }();
        if (rc != TURBO_OK) {
            release_context(c);
            return rc;
        }
        *out = c;
        return TURBO_OK;
    });
}

void context_release(void *ctx) {
    try {
        release_context(static_cast<Context *>(ctx));
    } catch (...) {
    }
}

struct Buffer {
    Context *ctx = nullptr;
    void *ptr = nullptr;
    uint32_t placement = 0;
    uint64_t bytes = 0;
    /* Whether release frees ptr: not for memory the caller owns, nor for a
     * session's output, which the session frees. */
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
    *host = b->placement == TURBO_PLACE_DEVICE ? nullptr : b->ptr;
    *out = b;
    return TURBO_OK;
}

int32_t buffer_alloc(void *ctx, const turbo_buffer_desc *desc, void **out, void **host, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Context *c = static_cast<Context *>(ctx);
        const uint64_t bytes = desc->bytes;
        if (bytes > SIZE_MAX - HOST_ALIGN)
            return refuse(err, TURBO_E_OUT_OF_MEMORY, "%llu bytes is more than the host can address",
                          (unsigned long long)bytes);
        ON_DEVICE(c->ordinal);
        void *p = nullptr;
        char what[64];
        snprintf(what, sizeof what, "%llu bytes of %s memory", (unsigned long long)bytes,
                 placement_name(desc->placement));
        // Not zeroed: turbo.h promises no contents, and the caller writes them.
        switch (desc->placement) {
        case TURBO_PLACE_DEVICE: TRY_CUDA(device_malloc(&p, bytes), what); break;
        case TURBO_PLACE_PINNED: TRY_CUDA(pinned_malloc(&p, bytes), what); break;
        case TURBO_PLACE_SHARED: TRY_CUDA(managed_malloc(&p, bytes), what); break;
        default:
            p = host_malloc(bytes);
            if (!p) return refuse(err, TURBO_E_OUT_OF_MEMORY, "%s", what);
        }
        Buffer *b = make<Buffer>();
        b->ctx = c;
        b->ptr = p;
        b->placement = desc->placement;
        b->bytes = bytes;
        b->owned = true;
        return give(b, out, host);
    });
}

/* What the driver says an address is, or cudaMemoryTypeUnregistered for
 * memory it does not know. */
cudaPointerAttributes attributes(const void *p) {
    cudaPointerAttributes a;
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
        (void)cudaGetLastError();
        memset(&a, 0, sizeof a);
        a.type = cudaMemoryTypeUnregistered;
        a.device = -1;
    }
    return a;
}

const char *memory_type_name(cudaMemoryType t) {
    switch (t) {
    case cudaMemoryTypeHost: return "page-locked host memory";
    case cudaMemoryTypeDevice: return "device memory";
    case cudaMemoryTypeManaged: return "managed memory";
    default: return "memory the driver does not know";
    }
}

/* The caller's memory, wrapped: nothing is copied, nothing is freed. A
 * CUDA_PTR names memory on this context's device, DEVICE or SHARED; a
 * HOST_PTR names host memory, which for PINNED is page-locked and for
 * SHARED managed. */
int32_t buffer_import(void *ctx, const turbo_buffer_desc *desc, const turbo_native_handle *h, void **out, void **host,
                      turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Context *c = static_cast<Context *>(ctx);
        if (h->kind != TURBO_HANDLE_CUDA_PTR && h->kind != TURBO_HANDLE_HOST_PTR)
            return refuse(err, TURBO_E_UNSUPPORTED,
                          "kind: %s: the cuda backend imports TURBO_HANDLE_CUDA_PTR and TURBO_HANDLE_HOST_PTR",
                          handle_name(h->kind));
        if (h->handle == 0) return refuse(err, TURBO_E_INVALID_ARGUMENT, "handle: a NULL pointer");
        const uint64_t end = h->handle + h->offset;
        if (end < h->handle || end + desc->bytes < end || end + desc->bytes > (uint64_t)UINTPTR_MAX)
            return refuse(
                err, TURBO_E_INVALID_ARGUMENT, "handle %#llx + offset %llu + %llu bytes is past the address space",
                (unsigned long long)h->handle, (unsigned long long)h->offset, (unsigned long long)desc->bytes);
        void *p = reinterpret_cast<void *>((uintptr_t)end);
        const uint32_t pl = desc->placement;
        ON_DEVICE(c->ordinal);
        const cudaPointerAttributes a = attributes(p);
        if (h->kind == TURBO_HANDLE_CUDA_PTR) {
            if (h->aux != (uint64_t)c->ordinal)
                return refuse(err, TURBO_E_INVALID_ARGUMENT, "aux: device %llu, and this context is on device %d",
                              (unsigned long long)h->aux, c->ordinal);
            if (pl != TURBO_PLACE_DEVICE && pl != TURBO_PLACE_SHARED)
                return refuse(err, TURBO_E_INVALID_ARGUMENT,
                              "placement: %s: a TURBO_HANDLE_CUDA_PTR is DEVICE or SHARED memory", placement_name(pl));
            const bool fits = pl == TURBO_PLACE_SHARED
                                  ? a.type == cudaMemoryTypeManaged
                                  : a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged;
            if (!fits)
                return refuse(err, TURBO_E_INVALID_ARGUMENT, "handle: the pointer is %s, not %s memory",
                              memory_type_name(a.type), placement_name(pl));
            if (a.type == cudaMemoryTypeDevice && a.device != c->ordinal)
                return refuse(err, TURBO_E_INVALID_ARGUMENT,
                              "handle: memory of device %d, and this context is on device %d", a.device, c->ordinal);
        } else {
            if (pl == TURBO_PLACE_DEVICE)
                return refuse(err, TURBO_E_INVALID_ARGUMENT,
                              "placement: DEVICE: a TURBO_HANDLE_HOST_PTR is host memory; import device memory as "
                              "TURBO_HANDLE_CUDA_PTR");
            if (pl == TURBO_PLACE_PINNED && a.type != cudaMemoryTypeHost)
                return refuse(
                    err, TURBO_E_INVALID_ARGUMENT,
                    "placement: PINNED: the pointer is %s, not page-locked (cudaHostRegister it, or import it as HOST)",
                    memory_type_name(a.type));
            if (pl == TURBO_PLACE_SHARED && a.type != cudaMemoryTypeManaged)
                return refuse(err, TURBO_E_INVALID_ARGUMENT, "placement: SHARED: the pointer is %s, not managed memory",
                              memory_type_name(a.type));
            if (a.type == cudaMemoryTypeDevice)
                return refuse(err, TURBO_E_INVALID_ARGUMENT, "handle: the pointer is device memory, not host memory");
        }
        Buffer *b = make<Buffer>();
        b->ctx = c;
        b->ptr = p;
        b->placement = pl;
        b->bytes = desc->bytes;
        b->owned = false;
        return give(b, out, host);
    });
}

void buffer_release(void *buf) {
    try {
        Buffer *b = static_cast<Buffer *>(buf);
        if (b->owned) {
            DeviceScope scope(b->ctx->ordinal);
            cudaError_t e = cudaSuccess;
            switch (b->placement) {
            case TURBO_PLACE_HOST: free(b->ptr); break;
            case TURBO_PLACE_PINNED: e = cudaFreeHost(b->ptr); break;
            default: e = cudaFree(b->ptr);
            }
            if (e != cudaSuccess) {
                (void)cudaGetLastError();
                b->ctx->say(LOG_WARNING, "cuda backend: freeing a %s buffer of %llu bytes: %s: %s",
                            placement_name(b->placement), (unsigned long long)b->bytes, cudaGetErrorName(e),
                            cudaGetErrorString(e));
            }
        }
        delete b;
    } catch (...) {
    }
}

/* The buffer's own address, no copy: a device pointer for DEVICE and
 * SHARED, a host pointer for HOST, PINNED and SHARED. */
int32_t buffer_export(void *buf, uint32_t kind, turbo_native_handle *out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        const Buffer *b = static_cast<const Buffer *>(buf);
        const uint32_t pl = b->placement;
        const bool device = pl == TURBO_PLACE_DEVICE || pl == TURBO_PLACE_SHARED;
        const bool host = pl != TURBO_PLACE_DEVICE;
        if (kind == TURBO_HANDLE_CUDA_PTR && !device)
            return refuse(err, TURBO_E_UNSUPPORTED,
                          "kind: TURBO_HANDLE_CUDA_PTR: a %s buffer is host memory; export it as TURBO_HANDLE_HOST_PTR",
                          placement_name(pl));
        if (kind == TURBO_HANDLE_HOST_PTR && !host)
            return refuse(
                err, TURBO_E_UNSUPPORTED,
                "kind: TURBO_HANDLE_HOST_PTR: a DEVICE buffer has no host address; export it as TURBO_HANDLE_CUDA_PTR");
        if (kind != TURBO_HANDLE_CUDA_PTR && kind != TURBO_HANDLE_HOST_PTR)
            return refuse(err, TURBO_E_UNSUPPORTED,
                          "kind: %s: the cuda backend exports TURBO_HANDLE_CUDA_PTR and TURBO_HANDLE_HOST_PTR",
                          handle_name(kind));
        out->kind = kind;
        out->handle = (uint64_t)(uintptr_t)b->ptr;
        out->aux = kind == TURBO_HANDLE_CUDA_PTR ? (uint64_t)b->ctx->ordinal : 0;
        out->offset = 0;
        return TURBO_OK;
    });
}

/* A copy to the caller's host memory on the context's stream, behind
 * whatever was queued there before it, waited for before it returns. */
int32_t buffer_read(void *buf, void *dst, uint64_t bytes, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        const Buffer *b = static_cast<const Buffer *>(buf);
        if (bytes > b->bytes)
            return refuse(err, TURBO_E_INVALID_ARGUMENT, "%llu bytes from a buffer of %llu", (unsigned long long)bytes,
                          (unsigned long long)b->bytes);
        ON_DEVICE(b->ctx->ordinal);
        std::lock_guard<std::mutex> g(b->ctx->lock);
        const cudaError_t e = cudaMemcpyAsync(dst, b->ptr, bytes, cudaMemcpyDeviceToHost, b->ctx->stream);
        const cudaError_t done = cudaStreamSynchronize(b->ctx->stream);
        TRY_CUDA(e, "copying to the host");
        TRY_CUDA(done, "copying to the host");
        return TURBO_OK;
    });
}

// ---- Models --------------------------------------------------------------------
//
// Loading copies each tensor, in the dtype it is stored in, into one
// device allocation: that copy is the load, and the core's host bytes are
// not read again; lay_out says where each tensor goes in it. A model
// stored in F16 or BF16 gets a second allocation,
// its weights widened to F32 on the device, made by the first session
// that needs them and shared by every later one, as turbo.h's precision
// rules say; it goes with the model. Every session needs them: an F16
// session computes in F32 all but its GEMMs. A model stored in F32 or
// BF16 gets another for its first F16 session, its GEMM weights rounded
// to F16, shared the same way; an F16 model's GEMMs read its own weights.

/* TURBO_BERT_* in turbo_backend.h. */
enum : int {
    WORD = TURBO_BERT_WORD_EMBEDDINGS,
    POSITION = TURBO_BERT_POSITION_EMBEDDINGS,
    TOKEN_TYPE = TURBO_BERT_TOKEN_TYPE_EMBEDDINGS,
    EMB_LN_W = TURBO_BERT_EMBEDDINGS_LN_WEIGHT,
    EMB_LN_B = TURBO_BERT_EMBEDDINGS_LN_BIAS,
};

struct Model {
    Context *ctx = nullptr;
    turbo_backend_model desc{};
    /* Every tensor, as stored, and where each starts in it. */
    void *stored = nullptr;
    std::vector<size_t> offsets;
    std::vector<uint64_t> counts;
    /* Every tensor in F32: into stored for an F32 model, else into
     * widened once a session made it. */
    std::mutex widen_lock;
    void *widened = nullptr;
    std::vector<const float *> f32;
    /* The GEMM weights in F16, by tensor index, NULL for the other
     * tensors: into stored for an F16 model, else into narrowed once an
     * F16 session made it; empty when a weight is past F16's range. */
    std::mutex narrow_lock;
    bool narrow_tried = false;
    void *narrowed = nullptr;
    std::vector<const uint16_t *> f16;
};

void release_model(Model *m) {
    DeviceScope scope(m->ctx->ordinal);
    if (m->stored) cudaFree(m->stored);
    if (m->widened) cudaFree(m->widened);
    if (m->narrowed) cudaFree(m->narrowed);
    delete m;
}

const char *dtype_name(uint32_t d) { return d == TURBO_DTYPE_F32 ? "F32" : d == TURBO_DTYPE_F16 ? "F16" : "BF16"; }

/* Whether tensor i is a linear layer's weight, which a GEMM reads. */
bool gemm_weight(size_t i) {
    if (i < TURBO_BERT_EMBEDDING_TENSORS) return false;
    switch ((i - TURBO_BERT_EMBEDDING_TENSORS) % TURBO_BERT_LAYER_TENSORS) {
    case TURBO_BERT_Q_WEIGHT:
    case TURBO_BERT_K_WEIGHT:
    case TURBO_BERT_V_WEIGHT:
    case TURBO_BERT_ATTN_OUT_WEIGHT:
    case TURBO_BERT_FFN_IN_WEIGHT:
    case TURBO_BERT_FFN_OUT_WEIGHT: return true;
    default: return false;
    }
}

bool every_tensor(size_t) { return true; }

/* Where each tensor keep() takes starts in one allocation of elem-byte
 * values, into at (by tensor index; 0 for the others), and the
 * allocation's size. Tensors go in index order, each on DEVICE_ALIGN,
 * except that each layer's Q, K and V weights sit back to back, then
 * their biases, so one GEMM of [3 * hidden, hidden] makes the three
 * projections and their biases are one [3 * hidden] vector. Every copy
 * of the weights is laid out so. */
size_t lay_out(const std::vector<uint64_t> &counts, size_t elem, bool (*keep)(size_t), std::vector<size_t> &at) {
    static const int order[TURBO_BERT_LAYER_TENSORS] = {
        TURBO_BERT_Q_WEIGHT,        TURBO_BERT_K_WEIGHT,       TURBO_BERT_V_WEIGHT,       TURBO_BERT_Q_BIAS,
        TURBO_BERT_K_BIAS,          TURBO_BERT_V_BIAS,         TURBO_BERT_ATTN_OUT_WEIGHT, TURBO_BERT_ATTN_OUT_BIAS,
        TURBO_BERT_ATTN_LN_WEIGHT,  TURBO_BERT_ATTN_LN_BIAS,   TURBO_BERT_FFN_IN_WEIGHT,  TURBO_BERT_FFN_IN_BIAS,
        TURBO_BERT_FFN_OUT_WEIGHT,  TURBO_BERT_FFN_OUT_BIAS,   TURBO_BERT_FFN_LN_WEIGHT,  TURBO_BERT_FFN_LN_BIAS,
    };
    at.assign(counts.size(), 0);
    size_t total = 0;
    auto place = [&](size_t i, bool next_follows) {
        if (!keep(i)) return;
        at[i] = total;
        total += counts[i] * elem;
        if (!next_follows) total = round_up(total, DEVICE_ALIGN);
    };
    for (size_t i = 0; i < counts.size() && i < TURBO_BERT_EMBEDDING_TENSORS; i++) place(i, false);
    for (size_t l0 = TURBO_BERT_EMBEDDING_TENSORS; l0 < counts.size(); l0 += TURBO_BERT_LAYER_TENSORS)
        for (int k = 0; k < TURBO_BERT_LAYER_TENSORS; k++) {
            const int r = order[k];
            place(l0 + r, r == TURBO_BERT_Q_WEIGHT || r == TURBO_BERT_K_WEIGHT || r == TURBO_BERT_Q_BIAS ||
                              r == TURBO_BERT_K_BIAS);
        }
    return total;
}

int32_t model_load(void *ctx, const turbo_backend_model *desc, void **out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Context *c = static_cast<Context *>(ctx);
        if (desc->family != TURBO_FAMILY_BERT)
            return refuse(err, TURBO_E_UNSUPPORTED, "family %u: the cuda backend holds BERT encoders", desc->family);
        if (desc->heads == 0 || desc->hidden % desc->heads != 0)
            return refuse(err, TURBO_E_UNSUPPORTED, "hidden %u is not a multiple of heads %u", desc->hidden,
                          desc->heads);
        if (desc->hidden / desc->heads > (uint32_t)ATTENTION_MAX_HEAD_DIM)
            return refuse(err, TURBO_E_UNSUPPORTED, "heads %u wide: the cuda backend's attention takes up to %d",
                          desc->hidden / desc->heads, ATTENTION_MAX_HEAD_DIM);
        const size_t elem = desc->dtype == TURBO_DTYPE_F32 ? 4 : 2;
        Model *m = make<Model>();
        m->ctx = c;
        m->desc = *desc;
        m->desc.tensors = nullptr;
        for (uint32_t i = 0; i < desc->tensor_count; i++) m->counts.push_back(desc->tensors[i].bytes / elem);
        const size_t total = lay_out(m->counts, elem, every_tensor, m->offsets);
        const int32_t rc = [&]() -> int32_t {
            ON_DEVICE(c->ordinal);
            TRY_CUDA(device_malloc(&m->stored, total), "device memory for the weights");
            std::lock_guard<std::mutex> g(c->lock);
            char *base = static_cast<char *>(m->stored);
            for (uint32_t i = 0; i < desc->tensor_count; i++) {
                const turbo_backend_tensor &t = desc->tensors[i];
                TRY_CUDA(cudaMemcpyAsync(base + m->offsets[i], t.data, t.bytes, cudaMemcpyHostToDevice, c->stream),
                         t.name);
            }
            TRY_CUDA(cudaStreamSynchronize(c->stream), "copying the weights to the device");
            if (desc->dtype == TURBO_DTYPE_F32)
                for (uint32_t i = 0; i < desc->tensor_count; i++)
                    m->f32.push_back(reinterpret_cast<const float *>(base + m->offsets[i]));
            return TURBO_OK;
        }();
        if (rc != TURBO_OK) {
            release_model(m);
            return rc;
        }
        c->say(LOG_DEBUG, "cuda device %d: a BERT of %u layers, %zu bytes of %s weights", c->ordinal, desc->layers,
               total, dtype_name(desc->dtype));
        *out = m;
        return TURBO_OK;
    });
}

void model_release(void *model) {
    try {
        release_model(static_cast<Model *>(model));
    } catch (...) {
    }
}

/* The model's weights in F32, made on first need. */
int32_t f32_weights(Model *m, turbo_error *err) {
    std::lock_guard<std::mutex> g(m->widen_lock);
    if (!m->f32.empty()) return TURBO_OK;
    Context *c = m->ctx;
    std::vector<size_t> at;
    const size_t total = lay_out(m->counts, 4, every_tensor, at);
    ON_DEVICE(c->ordinal);
    void *wide = nullptr;
    TRY_CUDA(device_malloc(&wide, total), "device memory for the weights in F32");
    const int32_t rc = [&]() -> int32_t {
        std::lock_guard<std::mutex> cg(c->lock);
        const char *src = static_cast<const char *>(m->stored);
        char *dst = static_cast<char *>(wide);
        for (size_t i = 0; i < m->counts.size(); i++) {
            const uint16_t *s = reinterpret_cast<const uint16_t *>(src + m->offsets[i]);
            float *d = reinterpret_cast<float *>(dst + at[i]);
            if (m->desc.dtype == TURBO_DTYPE_F16)
                TRY_CUDA(widen_f16(c->stream, s, m->counts[i], d), "widening F16 weights");
            else
                TRY_CUDA(widen_bf16(c->stream, s, m->counts[i], d), "widening BF16 weights");
        }
        TRY_CUDA(cudaStreamSynchronize(c->stream), "widening the weights");
        return TURBO_OK;
    }();
    if (rc != TURBO_OK) {
        cudaFree(wide);
        return rc;
    }
    m->widened = wide;
    for (size_t i = 0; i < m->counts.size(); i++)
        m->f32.push_back(reinterpret_cast<const float *>(static_cast<char *>(wide) + at[i]));
    return TURBO_OK;
}

/* The model's GEMM weights in F16, made on first need from its F32
 * weights, which f32_weights made before, laid out as lay_out says. A
 * weight past F16's range leaves f16 empty, and F16 sessions of this
 * model compute in F32. */
int32_t f16_weights(Model *m, turbo_error *err) {
    std::lock_guard<std::mutex> g(m->narrow_lock);
    if (m->narrow_tried) return TURBO_OK;
    Context *c = m->ctx;
    const size_t n = m->counts.size();
    if (m->desc.dtype == TURBO_DTYPE_F16) {
        const char *base = static_cast<const char *>(m->stored);
        m->f16.assign(n, nullptr);
        for (size_t i = 0; i < n; i++)
            if (gemm_weight(i)) m->f16[i] = reinterpret_cast<const uint16_t *>(base + m->offsets[i]);
        m->narrow_tried = true;
        return TURBO_OK;
    }
    std::vector<size_t> at;
    size_t total = lay_out(m->counts, 2, gemm_weight, at);
    const size_t flag_at = total;
    total += DEVICE_ALIGN;
    ON_DEVICE(c->ordinal);
    void *narrow = nullptr;
    TRY_CUDA(device_malloc(&narrow, total), "device memory for the weights in F16");
    char *dst = static_cast<char *>(narrow);
    int32_t overflow = 0;
    const int32_t rc = [&]() -> int32_t {
        std::lock_guard<std::mutex> cg(c->lock);
        int32_t *flag = reinterpret_cast<int32_t *>(dst + flag_at);
        TRY_CUDA(cudaMemsetAsync(flag, 0, sizeof *flag, c->stream), "narrowing the weights to F16");
        for (size_t i = 0; i < n; i++)
            if (gemm_weight(i))
                TRY_CUDA(narrow_f16(c->stream, m->f32[i], m->counts[i], reinterpret_cast<uint16_t *>(dst + at[i]),
                                    flag),
                         "narrowing the weights to F16");
        TRY_CUDA(cudaMemcpyAsync(&overflow, flag, sizeof overflow, cudaMemcpyDeviceToHost, c->stream),
                 "narrowing the weights to F16");
        TRY_CUDA(cudaStreamSynchronize(c->stream), "narrowing the weights to F16");
        return TURBO_OK;
    }();
    if (rc != TURBO_OK) {
        cudaFree(narrow);
        return rc;
    }
    m->narrow_tried = true;
    if (overflow) {
        cudaFree(narrow);
        c->say(LOG_WARNING,
               "cuda device %d: a GEMM weight of this %s model is past F16's range; FASTEST computes it in F32",
               c->ordinal, dtype_name(m->desc.dtype));
        return TURBO_OK;
    }
    m->narrowed = narrow;
    m->f16.assign(n, nullptr);
    for (size_t i = 0; i < n; i++)
        if (gemm_weight(i)) m->f16[i] = reinterpret_cast<const uint16_t *>(dst + at[i]);
    return TURBO_OK;
}

// ---- Sessions ------------------------------------------------------------------
//
// A session is device scratch for its largest batch, page-locked staging
// for the rows, and the buffer its vectors are written to, all allocated
// here; a run allocates nothing.
//
// The rows are computed packed, as on the CPU: each row's positions up to
// its last live token, one row after another, so the padding past that
// token is never computed, by a GEMM, a LayerNorm, GELU or attention. No
// output depends on it: attention skips masked keys, and no pooling reads
// past the last live token. The run finds the packing from the mask on
// the device (pack_rows), into the session's start and len arrays. The
// GEMMs' token count, and the largest row a launch covers, are the host's
// to know: embed_write counts them from the mask it is handed, which is
// host memory (the core has read and checked every entry of it, and rows
// in device memory are refused below), so nothing is copied back for
// them and a run waits on nothing before its last launch.
//
// An F16 session (FASTEST) keeps the hidden states, the residuals, the
// LayerNorms, softmax and pooling in F32, as an F32 session does; the
// kernels that feed a GEMM also write their output in F16, and the GEMMs
// read that and the F16 weights, accumulate in F32 and write F32.
//
// embed_write sends the rows to the device: from the caller's memory when
// it is page-locked (a PINNED buffer's, say), waiting for the copy, since
// that memory is the caller's again when the call returns; else through
// the session's staging, without waiting: the run is queued behind the
// copy, and the next write waits for it before it writes the staging
// again. Only the run's end waits on the stream. run then leaves the
// vectors on the device, in the session's DEVICE buffer:
// turbo_result_buffer hands out that memory, and turbo_result_read copies
// it back through buffer_read.
// So UPLOAD is a device stage, DOWNLOAD does not run, h2d_bytes is the
// rows sent, and d2h_bytes is 0 until a read.

struct Session {
    Model *model = nullptr;
    Context *ctx = nullptr;
    uint32_t max_batch = 0, max_seq = 0;
    /* F16 GEMMs, else F32 throughout. */
    bool half = false;
    /* What attention_max_shared gave, for the widest launch that fits. */
    size_t shared_most = 0;
    void *scratch = nullptr;
    int32_t *ids = nullptr, *mask = nullptr, *types = nullptr;
    /* [max_batch] each: every row's first packed token and its length. */
    int32_t *start = nullptr, *len = nullptr;
    /* [tokens, 3 * hidden] for one GEMM of the three projections, else q,
     * k and v are its thirds, [tokens, hidden] each. */
    float *qkv = nullptr;
    float *x = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *att = nullptr, *tmp = nullptr, *ffn = nullptr;
    /* An F16 session's GEMM inputs: the hidden states, the attention
     * context and GELU's output. */
    uint16_t *x16 = nullptr, *att16 = nullptr, *ffn16 = nullptr;
    /* [max_batch, hidden] F32 on the device, handed out as the output. */
    Buffer output;
    /* [3, max_batch * max_seq] int32, page-locked, and the event that
     * says the last write's copies from it are done. */
    int32_t *staging = nullptr;
    cudaEvent_t sent = nullptr;
    /* What the last write left. */
    bool written = false;
    bool has_types = false;
    uint32_t batch = 0, seq = 0, pooling = 0, normalize = 0, output_dim = 0;
    /* The packed tokens, and the longest row through its last live token. */
    uint32_t tokens = 0, max_len = 0;
    uint64_t h2d = 0;
};

void release_session(Session *s) {
    DeviceScope scope(s->ctx->ordinal);
    if (s->sent) {
        cudaEventSynchronize(s->sent);
        cudaEventDestroy(s->sent);
    }
    if (s->scratch) cudaFree(s->scratch);
    if (s->staging) cudaFreeHost(s->staging);
    delete s;
}

int32_t session_create(void *model, uint32_t task, uint32_t max_batch, uint32_t max_seq, uint32_t precision,
                       uint32_t *compute_dtype, void **out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Model *m = static_cast<Model *>(model);
        Context *c = m->ctx;
        const turbo_backend_model &d = m->desc;
        if (task != TURBO_TASK_EMBED)
            return refuse(err, TURBO_E_UNSUPPORTED_TASK, "task %u: the cuda backend runs embed", task);
        if (precision == TURBO_PRECISION_MODEL && d.dtype != TURBO_DTYPE_F32)
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 3,
                                "precision: MODEL computes in the weights' %s, and the cuda backend computes MODEL "
                                "in F32 only; EXACT computes this model in F32, FASTEST in F16",
                                dtype_name(d.dtype));
        if (max_seq > d.max_positions)
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 2, "max_seq %u is over the model's %u positions",
                                max_seq, d.max_positions);
        const size_t shared = attention_shared_bytes((int)max_seq, (int)d.hidden, (int)d.heads);
        size_t most = 0;
        {
            ON_DEVICE(c->ordinal);
            TRY_CUDA(attention_max_shared(&most), "the shared memory attention may have");
        }
        if (shared > most)
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 2,
                                "max_seq %u: attention needs %zu bytes of shared memory per block, and device %d "
                                "gives it %zu",
                                max_seq, shared, c->ordinal, most);
        if (max_batch > 65535)
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 1,
                                "max_batch %u: the cuda backend runs at most 65535 rows", max_batch);
        const size_t tokens = (size_t)max_batch * max_seq;
        // The widest GEMM output: the feed-forward input's, or the three
        // projections' side by side.
        if (tokens > (size_t)INT32_MAX / (d.intermediate > 3 * d.hidden ? d.intermediate : 3 * d.hidden))
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 1, "%u rows of %u tokens is more than one GEMM takes",
                                max_batch, max_seq);
        TRY(f32_weights(m, err));
        bool half = false;
        if (precision == TURBO_PRECISION_FASTEST) {
            TRY(f16_weights(m, err));
            half = !m->f16.empty();
        }

        ON_DEVICE(c->ordinal);
        // Attention may take the device's most, past the default 48 KiB,
        // for every session: the setting is the kernel's, not the session's.
        TRY_CUDA(attention_allow_shared(), "cudaFuncSetAttribute");
        Session *s = make<Session>();
        s->model = m;
        s->ctx = c;
        s->max_batch = max_batch;
        s->max_seq = max_seq;
        s->half = half;
        s->shared_most = most;
        const size_t ints = round_up(tokens * 4, DEVICE_ALIGN);
        const size_t rows = round_up((size_t)max_batch * 4, DEVICE_ALIGN);
        const size_t wide = round_up(tokens * d.hidden * 4, DEVICE_ALIGN);
        const size_t ffn = round_up(tokens * d.intermediate * 4, DEVICE_ALIGN);
        const size_t output = round_up((size_t)max_batch * d.hidden * 4, DEVICE_ALIGN);
        const size_t wide16 = half ? round_up(tokens * d.hidden * 2, DEVICE_ALIGN) : 0;
        const size_t ffn16 = half ? round_up(tokens * d.intermediate * 2, DEVICE_ALIGN) : 0;
        const size_t total = 3 * ints + 2 * rows + 6 * wide + ffn + output + 2 * wide16 + ffn16;
        const int32_t rc = [&]() -> int32_t {
            TRY_CUDA(device_malloc(&s->scratch, total), "device memory for the session");
            TRY_CUDA(pinned_malloc(reinterpret_cast<void **>(&s->staging), 3 * tokens * 4),
                     "page-locked memory for the session's rows");
            TRY_CUDA(cudaEventCreateWithFlags(&s->sent, cudaEventDisableTiming), "cudaEventCreateWithFlags");
            char *p = static_cast<char *>(s->scratch);
            auto take = [&](size_t n) {
                char *at = p;
                p += n;
                return at;
            };
            s->ids = reinterpret_cast<int32_t *>(take(ints));
            s->mask = reinterpret_cast<int32_t *>(take(ints));
            s->types = reinterpret_cast<int32_t *>(take(ints));
            s->start = reinterpret_cast<int32_t *>(take(rows));
            s->len = reinterpret_cast<int32_t *>(take(rows));
            s->x = reinterpret_cast<float *>(take(wide));
            // q, k and v back to back: thirds of one [tokens, 3 * hidden].
            s->qkv = reinterpret_cast<float *>(take(3 * wide));
            s->q = s->qkv;
            s->k = reinterpret_cast<float *>(reinterpret_cast<char *>(s->qkv) + wide);
            s->v = reinterpret_cast<float *>(reinterpret_cast<char *>(s->qkv) + 2 * wide);
            s->att = reinterpret_cast<float *>(take(wide));
            s->tmp = reinterpret_cast<float *>(take(wide));
            s->ffn = reinterpret_cast<float *>(take(ffn));
            if (half) {
                s->x16 = reinterpret_cast<uint16_t *>(take(wide16));
                s->att16 = reinterpret_cast<uint16_t *>(take(wide16));
                s->ffn16 = reinterpret_cast<uint16_t *>(take(ffn16));
            }
            s->output.ctx = c;
            s->output.ptr = take(output);
            s->output.placement = TURBO_PLACE_DEVICE;
            s->output.bytes = (uint64_t)max_batch * d.hidden * 4;
            s->output.owned = false;
            return TURBO_OK;
        }();
        if (rc != TURBO_OK) {
            release_session(s);
            return rc;
        }
        c->say(LOG_DEBUG, "cuda device %d: an embed session of %u rows of %u tokens, computing in %s", c->ordinal,
               max_batch, max_seq, half ? "F16 with F32 accumulation" : "F32");
        *compute_dtype = half ? TURBO_DTYPE_F16 : TURBO_DTYPE_F32;
        *out = s;
        return TURBO_OK;
    });
}

void session_release(void *session) {
    try {
        release_session(static_cast<Session *>(session));
    } catch (...) {
    }
}

/* One [batch, seq] array of the rows to dst on the device, row_stride
 * elements apart in src, adding the bytes sent to *sent, and setting
 * *direct when the copy reads the caller's memory. Device memory is
 * refused: the core has read and checked the rows on the host, so they
 * are host memory, page-locked, managed or pageable. */
int32_t upload(Session &s, const int32_t *src, uint32_t batch, uint32_t seq, uint32_t stride, int32_t *staging,
               int32_t *dst, uint64_t *sent, bool *direct, turbo_error *err) {
    const size_t row = (size_t)seq * 4;
    const cudaPointerAttributes a = attributes(src);
    if (a.type == cudaMemoryTypeDevice)
        return refuse(err, TURBO_E_INVALID_ARGUMENT, "rows are device memory; the core reads rows on the host");
    if (a.type == cudaMemoryTypeHost || a.type == cudaMemoryTypeManaged) {
        TRY_CUDA(
            cudaMemcpy2DAsync(dst, row, src, (size_t)stride * 4, row, batch, cudaMemcpyHostToDevice, s.ctx->stream),
            "sending the rows");
        *direct = true;
    } else {
        for (uint32_t r = 0; r < batch; r++) memcpy(staging + (size_t)r * seq, src + (size_t)r * stride, row);
        TRY_CUDA(cudaMemcpyAsync(dst, staging, row * batch, cudaMemcpyHostToDevice, s.ctx->stream), "sending the rows");
    }
    *sent += row * batch;
    return TURBO_OK;
}

int32_t embed_write(void *session, const turbo_backend_embed_rows *r, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Session &s = *static_cast<Session *>(session);
        s.written = false;
        const size_t tokens = (size_t)s.max_batch * s.max_seq;
        uint64_t sent = 0;
        {
            std::lock_guard<std::mutex> g(s.ctx->lock);
            ON_DEVICE(s.ctx->ordinal);
            // The staging is written again here: the last write's copies
            // from it are done first (at once, after a run, which waited
            // for them).
            TRY_CUDA(cudaEventSynchronize(s.sent), "sending the rows");
            bool direct = false;
            const int32_t rc = [&]() -> int32_t {
                TRY(upload(s, r->ids, r->batch, r->seq, r->row_stride, s.staging, s.ids, &sent, &direct, err));
                TRY(upload(s, r->mask, r->batch, r->seq, r->row_stride, s.staging + tokens, s.mask, &sent, &direct,
                           err));
                if (r->types)
                    TRY(upload(s, r->types, r->batch, r->seq, r->row_stride, s.staging + 2 * tokens, s.types, &sent,
                               &direct, err));
                return TURBO_OK;
            }();
            if (rc != TURBO_OK) {
                // Nothing queued may read memory the next write reuses.
                (void)cudaStreamSynchronize(s.ctx->stream);
                return rc;
            }
            TRY_CUDA(cudaEventRecord(s.sent, s.ctx->stream), "sending the rows");
            // Rows copied from the caller's page-locked or managed memory
            // are read by the device until the copy ends, and the caller's
            // arrays are valid for this call only: wait for them. Rows
            // copied through the staging are the session's, and the run
            // that reads them is queued behind the copy.
            if (direct) TRY_CUDA(cudaEventSynchronize(s.sent), "sending the rows");
        }
        // The packed size, from the mask upload() found on the host: each
        // row through its last live token, as pack_rows finds it on the
        // device from the same entries.
        uint32_t packed = 0, longest = 0;
        for (uint32_t b = 0; b < r->batch; b++) {
            const int32_t *m = r->mask + (size_t)b * r->row_stride;
            uint32_t n = r->seq;
            while (n > 0 && m[n - 1] == 0) n--;
            packed += n;
            longest = n > longest ? n : longest;
        }
        s.has_types = r->types != nullptr;
        s.batch = r->batch;
        s.seq = r->seq;
        s.pooling = r->pooling;
        s.normalize = r->normalize;
        s.output_dim = r->output_dim;
        s.tokens = packed;
        s.max_len = longest;
        s.h2d = sent;
        s.written = true;
        return TURBO_OK;
    });
}

/* y[t, o] = sum_i x[t, i] w[o, i] for t under tokens, with w [n_out, n_in]
 * row-major: in cuBLAS's column-major terms, y^T = w x^T. x and w are F32
 * in an F32 session, F16 in an F16 one; y is F32 in both. */
cublasStatus_t linear(const Session &s, const void *x, int tokens, int n_in, const void *w, int n_out, float *y) {
    const float one = 1.0f, zero = 0.0f;
    cublasHandle_t h = s.ctx->blas;
    if (!s.half)
        return cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n_out, tokens, n_in, &one, static_cast<const float *>(w), n_in,
                           static_cast<const float *>(x), n_in, &zero, y, n_out);
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n_out, tokens, n_in, &one, w, CUDA_R_16F, n_in, x, CUDA_R_16F,
                        n_in, &zero, y, CUDA_R_32F, n_out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

/* The encoder over the written rows, queued on the context's stream. */
int32_t encode(Session &s, turbo_error *err) {
    const Model &mo = *s.model;
    const turbo_backend_model &d = mo.desc;
    const std::vector<const float *> &w = mo.f32;
    const int batch = (int)s.batch, seq = (int)s.seq, tokens = (int)s.tokens, max_len = (int)s.max_len;
    const int h = (int)d.hidden, inter = (int)d.intermediate;
    const float eps = (float)d.layer_norm_eps;
    const bool half = s.half;
    cudaStream_t st = s.ctx->stream;
    auto at = [](uint32_t l, int r) { return (size_t)TURBO_BERT_EMBEDDING_TENSORS + l * TURBO_BERT_LAYER_TENSORS + r; };
    auto layer = [&](uint32_t l, int r) { return w[at(l, r)]; };
    // A GEMM's weight in the session's dtype.
    auto weight = [&](uint32_t l, int r) -> const void * {
        return half ? static_cast<const void *>(mo.f16[at(l, r)]) : static_cast<const void *>(layer(l, r));
    };
    // What the GEMMs read: the F16 copy in an F16 session.
    const void *xin = half ? static_cast<const void *>(s.x16) : s.x;
    const void *attin = half ? static_cast<const void *>(s.att16) : s.att;
    const void *ffnin = half ? static_cast<const void *>(s.ffn16) : s.ffn;

    TRY_CUDA(pack_rows(st, s.mask, batch, seq, s.start, s.len), "packing the rows");
    TRY_CUDA(embed_layer_norm(st, s.ids, s.has_types ? s.types : nullptr, w[WORD], w[POSITION], w[TOKEN_TYPE],
                              w[EMB_LN_W], w[EMB_LN_B], eps, s.start, s.len, batch, max_len, seq, h, s.x, s.x16),
             "the embedding lookup");
    for (uint32_t l = 0; l < d.layers; l++) {
        const float *q = s.q, *k = s.k, *v = s.v;
        int ld = h;
        const char *wq = static_cast<const char *>(weight(l, TURBO_BERT_Q_WEIGHT));
        const size_t wbytes = (size_t)h * h * (half ? 2 : 4);
        if (weight(l, TURBO_BERT_K_WEIGHT) == wq + wbytes && weight(l, TURBO_BERT_V_WEIGHT) == wq + 2 * wbytes) {
            // Q, K and V's weights back to back, as lay_out puts them: one
            // GEMM of 3 * hidden outputs, each token's q, k and v side by
            // side.
            TRY_CUBLAS(linear(s, xin, tokens, h, wq, 3 * h, s.qkv), "the query, key and value projections");
            q = s.qkv;
            k = s.qkv + h;
            v = s.qkv + 2 * h;
            ld = 3 * h;
        } else {
            TRY_CUBLAS(linear(s, xin, tokens, h, weight(l, TURBO_BERT_Q_WEIGHT), h, s.q), "the query projection");
            TRY_CUBLAS(linear(s, xin, tokens, h, weight(l, TURBO_BERT_K_WEIGHT), h, s.k), "the key projection");
            TRY_CUBLAS(linear(s, xin, tokens, h, weight(l, TURBO_BERT_V_WEIGHT), h, s.v), "the value projection");
        }
        TRY_CUDA(attention(st, q, k, v, ld, layer(l, TURBO_BERT_Q_BIAS), layer(l, TURBO_BERT_K_BIAS),
                           layer(l, TURBO_BERT_V_BIAS), s.mask, s.start, s.len, batch, max_len, seq, h, (int)d.heads,
                           s.shared_most, s.att, s.att16),
                 "attention");
        TRY_CUBLAS(linear(s, attin, tokens, h, weight(l, TURBO_BERT_ATTN_OUT_WEIGHT), h, s.tmp),
                   "the attention output projection");
        TRY_CUDA(add_layer_norm(st, s.x, s.tmp, layer(l, TURBO_BERT_ATTN_OUT_BIAS), layer(l, TURBO_BERT_ATTN_LN_WEIGHT),
                                layer(l, TURBO_BERT_ATTN_LN_BIAS), eps, tokens, h, s.x16),
                 "the attention LayerNorm");
        TRY_CUBLAS(linear(s, xin, tokens, h, weight(l, TURBO_BERT_FFN_IN_WEIGHT), inter, s.ffn),
                   "the feed-forward input");
        TRY_CUDA(bias_gelu(st, s.ffn, layer(l, TURBO_BERT_FFN_IN_BIAS), tokens, inter, s.ffn16), "GELU");
        TRY_CUBLAS(linear(s, ffnin, tokens, inter, weight(l, TURBO_BERT_FFN_OUT_WEIGHT), h, s.tmp),
                   "the feed-forward output");
        TRY_CUDA(add_layer_norm(st, s.x, s.tmp, layer(l, TURBO_BERT_FFN_OUT_BIAS), layer(l, TURBO_BERT_FFN_LN_WEIGHT),
                                layer(l, TURBO_BERT_FFN_LN_BIAS), eps, tokens, h, s.x16),
                 "the feed-forward LayerNorm");
    }
    TRY_CUDA(pool(st, s.x, s.mask, s.start, s.len, batch, seq, h, (int)s.output_dim, s.pooling,
                  s.normalize == TURBO_NORMALIZE_L2 ? 1 : 0, static_cast<float *>(s.output.ptr)),
             "pooling");
    return TURBO_OK;
}

int32_t session_run(void *session, turbo_backend_run *out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Session &s = *static_cast<Session *>(session);
        if (!s.written)
            return refuse(err, TURBO_E_INVALID_STATE, "the cuda session has no rows written since its last run");
        s.written = false;
        const uint64_t host0 = host_here, device0 = device_here;
        {
            std::lock_guard<std::mutex> g(s.ctx->lock);
            ON_DEVICE(s.ctx->ordinal);
            const int32_t rc = encode(s, err);
            // The stream is left idle whether or not the run finished.
            const cudaError_t e = cudaStreamSynchronize(s.ctx->stream);
            if (rc != TURBO_OK) {
                (void)cudaGetLastError();
                return rc;
            }
            TRY_CUDA(e, "the run");
        }
        out->placement = TURBO_PLACE_DEVICE;
        out->output = &s.output;
        out->host = nullptr;
        out->h2d_bytes = s.h2d;
        out->d2h_bytes = 0;
        out->host_allocs = host_here - host0;
        out->device_allocs = device_here - device0;
        uint32_t *st = out->stage;
        st[TURBO_EMBED_STAGE_UPLOAD] = TURBO_STAGE_DEVICE;
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

extern const turbo_backend turbo_cuda_backend;

const turbo_backend turbo_cuda_backend = {
    sizeof(turbo_backend),
    0,
    "cuda",
    LINKED_RUNTIME.s,
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
void turbo_cuda_allocations(uint64_t *host, uint64_t *device) {
    *host = host_total.load(std::memory_order_relaxed);
    *device = device_total.load(std::memory_order_relaxed);
}

/* The label device_info would give a device of this name. */
void turbo_cuda_arch_label(const char *name, char *out, size_t len) { arch_label(name, out, len); }

/* The F32 copy of an F16 or BF16 model's weights, once a session made it;
 * NULL before, and for an F32 model, which computes from its own. */
const void *turbo_cuda_widened(void *model) {
    try {
        Model *m = static_cast<Model *>(model);
        std::lock_guard<std::mutex> g(m->widen_lock);
        return m->widened;
    } catch (...) {
        return nullptr;
    }
}

/* The F16 copy of an F32 or BF16 model's GEMM weights, once an F16
 * session made it; NULL before, and for an F16 model, whose GEMMs read
 * its own. */
const void *turbo_cuda_narrowed(void *model) {
    try {
        Model *m = static_cast<Model *>(model);
        std::lock_guard<std::mutex> g(m->narrow_lock);
        return m->narrowed;
    } catch (...) {
        return nullptr;
    }
}

} // extern "C"
