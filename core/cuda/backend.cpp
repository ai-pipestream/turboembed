/* SPDX-License-Identifier: Apache-2.0
 *
 * The CUDA backend: NVIDIA GPUs through the CUDA runtime, behind
 * include/turbo/turbo_backend.h. It lists the devices the driver reports,
 * keeps a stream and a cuBLAS handle per context, holds models in device
 * memory, and runs embed sessions with the BERT encoder in kernels.cu,
 * whose GEMMs are its own: F32 FMAs for an F32 session, F16 inputs with
 * F32 accumulation on the tensor cores for an F16 one. A session runs as
 * one CUDA graph, captured when it is made. cuBLAS computes a GEMM only
 * when TURBO_CUDA_CUBLAS names it, for measuring one against the other,
 * and checks the backend's own GEMMs in the tests.
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

/* Embed at every precision: MODEL and EXACT in F32 (F32 FMAs; with
 * TURBO_CUDA_TF32=1, MODEL's GEMMs as TF32 on the tensor cores from
 * sm_80, F32 accumulation), FASTEST in F16 (F16 GEMM inputs, F32 accumulation, and
 * everything else in F32). As on the
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
// A context is a stream and a cuBLAS handle on one device. The handle is
// for the GEMMs TURBO_CUDA_CUBLAS hands to cuBLAS; it computes on the
// stream with TF32 off (CUBLAS_DEFAULT_MATH: TF32 would round an F32
// GEMM's inputs to 10 bits of mantissa), so an F32 GEMM is F32; an F16
// session's GEMMs take F16 inputs and accumulate in F32
// (CUBLAS_COMPUTE_32F). The handle has a fixed workspace of its own, so
// a GEMM never allocates.
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
        if (desc->hidden > (uint32_t)MAX_HIDDEN || desc->hidden % 8 != 0 || desc->intermediate % 8 != 0)
            return refuse(err, TURBO_E_UNSUPPORTED,
                          "hidden %u and intermediate %u: the cuda backend takes multiples of 8, hidden up to %d",
                          desc->hidden, desc->intermediate, MAX_HIDDEN);
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
// for the rows, the buffer its vectors are written to and the CUDA graph
// of its run, all made here; a run allocates nothing.
//
// The rows are computed packed, as on the CPU: each row's positions up to
// its last live token, one row after another, so the padding past that
// token is never computed, by a GEMM, a LayerNorm, GELU or attention. No
// output depends on it: attention skips masked keys, and no pooling reads
// past the last live token. The run finds the packing from the mask on
// the device (pack_rows): the packed token count, each row's start and
// length, and the order attention takes the rows in, longest first. Every
// later kernel reads the count there and is launched for the session's
// max_batch x max_seq tokens, its blocks looping over the work there is,
// so the run's launches never change: session_create captures them once
// into a graph, and a run is one graph launch, after the fetch and pack
// kernels' arguments (the rows' shape and the embed options) are set in
// it when they differ from the last run's. The rows go to the device as
// they are, one [k][batch][seq] array, and the kernels read their shape
// from the packing, so no launch depends on the rows' width either.
//
// An F16 session (FASTEST) keeps the hidden states, the residuals, the
// LayerNorms, softmax and pooling in F32, as an F32 session does; its
// GEMMs read F16 (the LayerNorms write the hidden states in F16 too) and
// accumulate in F32; the QKV and feed-forward input GEMMs and attention
// write F16, which only a GEMM or attention reads.
//
// TURBO_CUDA_TILE, read when a session is made, names the GEMMs' tile
// for all four: 64x64, 128x64, 128x128 or 128x128-16x8 (the FMA kernel's
// 128x128 over 128 threads of 16 x 8 outputs; 128x128 elsewhere), and
// on the tensor cores 128x128-4w (four warps of 64 x 64), 256x128 (eight
// such warps) or 8w (the eight-warp mix FASTEST took before: 128x128 for
// QKV and the first feed-forward GEMM, 128x64 for the other two), for
// measuring one against another and against the default (128x64 for
// the FMA kernel and TF32; F16 on the tensor cores 256x128 for the
// first feed-forward GEMM and 128x128-4w for the others).
//
// TURBO_CUDA_ATTENTION=split, read when a session is made, gives an F32
// session (and an F16 one without the tensor cores' attention) the FMA
// attention that splits each query's keys among four warps, in place of
// the default that computes Q K^T and P V as register tiles.
//
// TURBO_CUDA_LAYER_NORM=fused, read when a session is made, has the
// attention output and second feed-forward GEMMs' epilogue add the bias
// and residual and run the LayerNorm, in place of the default that
// computes the product alone and then the bias, residual and LayerNorm in
// a kernel of their own (the same bits either way; on an RTX 4080 the
// separate kernel is faster).
// TURBO_CUDA_POOL=columns pools with a thread per column, in place of
// the default's groups of tokens summed apart.
//
// TURBO_CUDA_CUBLAS, read when a session is made, hands the GEMMs it names
// to cuBLAS: a comma-separated list of qkv, out, ffn1 and ffn2, or all.
// cuBLAS's product then goes through a kernel of the same epilogue, and
// the session runs without a graph, its launches sized from the packed
// token count embed_write finds on the host.
//
// embed_write sends the rows to the device: from the caller's memory when
// every array is page-locked (a PINNED buffer's, say) or managed and its
// rows are back to back, one contiguous copy each, waiting for them,
// since that memory is the caller's again when the call returns; else
// into the session's page-locked staging, laid out as on the device,
// which the run's first kernel reads over the bus, so nothing is queued
// ahead of the graph. Only the run's end waits on the stream. run then
// leaves the vectors on the device, in the session's DEVICE buffer:
// turbo_result_buffer hands out that memory, and turbo_result_read copies
// it back through buffer_read.
// So UPLOAD is a device stage, DOWNLOAD does not run, h2d_bytes is the
// rows sent, and d2h_bytes is 0 until a read.

/* The GEMMs TURBO_CUDA_CUBLAS may name. */
enum : unsigned { CUBLAS_QKV = 1, CUBLAS_OUT = 2, CUBLAS_FFN1 = 4, CUBLAS_FFN2 = 8 };

/* What the tests set in place of TURBO_CUDA_CUBLAS; -1 for the variable. */
std::atomic<int> cublas_override{-1};

/* What the tests set in place of TURBO_CUDA_TILE; -1 for the variable. */
std::atomic<int> tile_override{-1};

Tile tile_named() {
    const int o = tile_override.load(std::memory_order_relaxed);
    if (o >= 0) return (Tile)o;
    const char *v = getenv("TURBO_CUDA_TILE");
    if (!v) return TILE_DEFAULT;
    if (!strcasecmp(v, "64x64")) return TILE_64x64;
    if (!strcasecmp(v, "128x64")) return TILE_128x64;
    if (!strcasecmp(v, "128x128")) return TILE_128x128;
    if (!strcasecmp(v, "128x128-16x8")) return TILE_128x128_16x8;
    if (!strcasecmp(v, "128x128-4w")) return TILE_128x128_4W;
    if (!strcasecmp(v, "256x128")) return TILE_256x128;
    if (!strcasecmp(v, "8w")) return TILE_EIGHT_WARPS;
    return TILE_DEFAULT;
}

/* What the tests set in place of TURBO_CUDA_ATTENTION: 1 for the
 * key-split kernel, 0 for the default; -1 for the variable. */
std::atomic<int> split_attention_override{-1};

bool split_attention_named() {
    const int o = split_attention_override.load(std::memory_order_relaxed);
    if (o >= 0) return o != 0;
    const char *v = getenv("TURBO_CUDA_ATTENTION");
    return v && !strcasecmp(v, "split");
}

/* What the tests set in place of TURBO_CUDA_TF32: 1 for TF32 at MODEL,
 * 0 for the default; -1 for the variable. */
std::atomic<int> tf32_override{-1};

/* TURBO_CUDA_TF32=1 puts MODEL's F32 GEMMs on the tensor cores as TF32;
 * unset, they are EXACT's FMA kernels. */
bool tf32_named() {
    const int o = tf32_override.load(std::memory_order_relaxed);
    if (o >= 0) return o != 0;
    const char *v = getenv("TURBO_CUDA_TF32");
    return v && strcmp(v, "1") == 0;
}

/* What the tests set in place of TURBO_CUDA_LAYER_NORM: 1 for the
 * separate kernel (the default), 0 for the fused epilogue; -1 for the
 * variable. */
std::atomic<int> separate_ln_override{-1};

bool separate_ln_named() {
    const int o = separate_ln_override.load(std::memory_order_relaxed);
    if (o >= 0) return o != 0;
    const char *v = getenv("TURBO_CUDA_LAYER_NORM");
    return !(v && !strcasecmp(v, "fused"));
}

/* What the tests set in place of TURBO_CUDA_POOL: 1 for the kernel of a
 * thread per column, 0 for the default; -1 for the variable. */
std::atomic<int> column_pool_override{-1};

bool column_pool_named() {
    const int o = column_pool_override.load(std::memory_order_relaxed);
    if (o >= 0) return o != 0;
    const char *v = getenv("TURBO_CUDA_POOL");
    return v && !strcasecmp(v, "columns");
}

/* TURBO_CUDA_SK_STEPS: the GEMMs' fewest k steps per block, 0 (the
 * kernels' own) when unset or not a count from 1 to 64. */
int sk_steps_named() {
    const char *v = getenv("TURBO_CUDA_SK_STEPS");
    if (!v) return 0;
    char *end = nullptr;
    const long n = strtol(v, &end, 10);
    return end != v && *end == 0 && n >= 1 && n <= 64 ? (int)n : 0;
}

unsigned cublas_gemms() {
    const int o = cublas_override.load(std::memory_order_relaxed);
    if (o >= 0) return (unsigned)o;
    const char *v = getenv("TURBO_CUDA_CUBLAS");
    if (!v) return 0;
    unsigned mask = 0;
    const char *p = v;
    while (*p) {
        const char *w = p;
        while (*p && *p != ',') p++;
        const size_t n = (size_t)(p - w);
        if (word_is(w, n, "all")) mask |= CUBLAS_QKV | CUBLAS_OUT | CUBLAS_FFN1 | CUBLAS_FFN2;
        if (word_is(w, n, "qkv")) mask |= CUBLAS_QKV;
        if (word_is(w, n, "out")) mask |= CUBLAS_OUT;
        if (word_is(w, n, "ffn1")) mask |= CUBLAS_FFN1;
        if (word_is(w, n, "ffn2")) mask |= CUBLAS_FFN2;
        if (*p) p++;
    }
    return mask;
}

struct Session {
    Model *model = nullptr;
    Context *ctx = nullptr;
    uint32_t max_batch = 0, max_seq = 0;
    /* F16 GEMMs and attention, else F32 throughout. */
    bool half = false;
    Shape shape;
    Plan plan;
    /* The GEMMs cuBLAS computes; with any, the run is not a graph. */
    unsigned cublas = 0;
    void *scratch = nullptr;
    /* The written rows, [k][batch][seq]: ids, mask and types. */
    int32_t *rows = nullptr;
    Packing pk{};
    /* The hidden states, F32, and in F16 for an F16 session's GEMMs. */
    float *x = nullptr;
    uint16_t *x16 = nullptr;
    /* The QKV GEMM's head-major output, attention's context and GELU's
     * output: F16 in an F16 session, else F32. */
    void *qkv = nullptr, *att = nullptr, *ffn = nullptr;
    /* The attention output and feed-forward output GEMMs' products,
     * [tokens, hidden], which add_layer_norm adds. */
    float *part = nullptr;
    /* The GEMMs' stream-K workspace and flags. */
    float *ws = nullptr;
    int *flags = nullptr;
    /* A cuBLAS product before its epilogue, when cuBLAS computes one. */
    float *raw = nullptr;
    /* [max_batch, hidden] F32 on the device, handed out as the output. */
    Buffer output;
    /* [3, max_batch * max_seq] int32, page-locked, which the run's fetch
     * kernel reads, and the event that says a write's copies from the
     * caller's memory are done. */
    int32_t *staging = nullptr;
    cudaEvent_t sent = nullptr;
    /* The GEMMs' fault word, after the staging, as the host and the
     * device address it: set when a stream-K wait gave up. */
    int *fault = nullptr, *fault_dev = nullptr;
    /* The run as a graph, its fetch and pack kernels' nodes and arguments,
     * and the arguments last set in it. */
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t pack_node = nullptr, fetch_node = nullptr;
    PackArgs pack{};
    FetchArgs fetch{};
    void *pack_argv[1] = {nullptr}, *fetch_argv[1] = {nullptr};
    RunArgs in_graph{};
    int32_t fetch_in_graph = 0;
    bool graph_set = false;
    /* What the last write left. */
    bool written = false;
    RunArgs run{};
    /* The packed tokens, counted on the host for cuBLAS's GEMMs. */
    uint32_t tokens = 0;
    uint64_t h2d = 0;
};

void release_session(Session *s) {
    DeviceScope scope(s->ctx->ordinal);
    if (s->sent) {
        cudaEventSynchronize(s->sent);
        cudaEventDestroy(s->sent);
    }
    if (s->exec) cudaGraphExecDestroy(s->exec);
    if (s->graph) cudaGraphDestroy(s->graph);
    if (s->scratch) cudaFree(s->scratch);
    if (s->staging) cudaFreeHost(s->staging);
    delete s;
}

/* y[t, o] = sum_i x[t, i] w[o, i] for t under tokens, with w [n_out, n_in]
 * row-major: in cuBLAS's column-major terms, y^T = w x^T. x and w are F32
 * or F16; y is F32 in both. */
cublasStatus_t linear(cublasHandle_t h, bool half, const void *x, int tokens, int n_in, const void *w, int n_out,
                      float *y) {
    const float one = 1.0f, zero = 0.0f;
    if (!half)
        return cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n_out, tokens, n_in, &one, static_cast<const float *>(w), n_in,
                           static_cast<const float *>(x), n_in, &zero, y, n_out);
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n_out, tokens, n_in, &one, w, CUDA_R_16F, n_in, x, CUDA_R_16F,
                        n_in, &zero, y, CUDA_R_32F, n_out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

/* The encoder, queued on the context's stream: captured into the graph,
 * or run as it is queued when cuBLAS computes a GEMM. */
int32_t encode(Session &s, turbo_error *err) {
    const Model &mo = *s.model;
    const turbo_backend_model &d = mo.desc;
    const std::vector<const float *> &w = mo.f32;
    const Plan &plan = s.plan;
    const Shape &sh = s.shape;
    const int h = (int)d.hidden, inter = (int)d.intermediate, heads = (int)d.heads, hd = h / heads;
    const int tcap = sh.tcap, tokens = (int)s.tokens;
    const float eps = (float)d.layer_norm_eps;
    const bool half = s.half, tc = sh.tensor_cores;
    cudaStream_t st = s.ctx->stream;
    cublasHandle_t blas = s.ctx->blas;
    auto at = [](uint32_t l, int r) { return (size_t)TURBO_BERT_EMBEDDING_TENSORS + l * TURBO_BERT_LAYER_TENSORS + r; };
    auto layer = [&](uint32_t l, int r) { return w[at(l, r)]; };
    // A GEMM's weight in the session's dtype.
    auto weight = [&](uint32_t l, int r) -> const void * {
        return half ? static_cast<const void *>(mo.f16[at(l, r)]) : static_cast<const void *>(layer(l, r));
    };
    const void *xin = half ? static_cast<const void *>(s.x16) : s.x;
    const Info *info = s.pk.info;

    TRY_CUDA(fetch_rows(st, s.fetch, plan), "fetching the rows");
    TRY_CUDA(pack_rows(st, s.pack, plan), "packing the rows");
    TRY_CUDA(embed_layer_norm(st, s.rows, w[WORD], w[POSITION], w[TOKEN_TYPE], w[EMB_LN_W], w[EMB_LN_B], eps, s.pk,
                              h, s.x, s.x16, plan),
             "the embedding lookup");
    GemmArgs g{};
    g.info = info;
    g.heads = heads;
    g.head_dim = hd;
    g.hidden = h;
    g.tcap = tcap;
    g.ws = s.ws;
    g.flags = s.flags;
    g.fault = s.fault_dev;
    g.min_steps = s.shape.sk_steps;
    g.rows_done = s.flags + plan.sk_flags;
    g.eps = eps;
    g.x16 = s.x16;
    AttnArgs aa{};
    aa.qkv = s.qkv;
    aa.ctx = s.att;
    aa.p = s.pk;
    aa.heads = heads;
    aa.head_dim = hd;
    aa.hidden = h;
    aa.tcap = tcap;
    aa.chunk = plan.attn_chunk;
    aa.queries = plan.attn_queries;
    const Tile tile = sh.tile;
    aa.scale = 1.0f / sqrtf((float)hd);
    for (uint32_t l = 0; l < d.layers; l++) {
        // Q, K and V's weights back to back, as lay_out puts them: one GEMM
        // of 3 * hidden outputs, their biases added and each head's
        // written apart.
        g.a = xin;
        g.w = weight(l, TURBO_BERT_Q_WEIGHT);
        g.bias = layer(l, TURBO_BERT_Q_BIAS);
        g.out = s.qkv;
        g.n = 3 * h;
        g.k = h;
        if (s.cublas & CUBLAS_QKV) {
            TRY_CUBLAS(linear(blas, half, xin, tokens, h, g.w, 3 * h, s.raw), "the query, key and value projections");
            TRY_CUDA(qkv_epilogue(st, s.raw, g, half, plan), "the query, key and value biases");
        } else {
            TRY_CUDA(gemm(st, EPI_QKV, half, tc, tile, g, plan.qkv_grid), "the query, key and value projections");
        }
        TRY_CUDA(attention(st, aa, sh, plan), "attention");

        g.a = s.att;
        g.w = weight(l, TURBO_BERT_ATTN_OUT_WEIGHT);
        g.n = h;
        if (plan.fused_ln && !(s.cublas & CUBLAS_OUT)) {
            // The product, its bias, the residual and the LayerNorm in one.
            g.bias = layer(l, TURBO_BERT_ATTN_OUT_BIAS);
            g.out = s.x;
            g.ln_w = layer(l, TURBO_BERT_ATTN_LN_WEIGHT);
            g.ln_b = layer(l, TURBO_BERT_ATTN_LN_BIAS);
            TRY_CUDA(gemm(st, EPI_ADD_LN, half, tc, tile, g, plan.out_grid),
                     "the attention output projection and LayerNorm");
        } else {
            g.bias = nullptr;
            g.out = s.part;
            if (s.cublas & CUBLAS_OUT) {
                TRY_CUBLAS(linear(blas, half, s.att, tokens, h, g.w, h, s.part), "the attention output projection");
            } else {
                TRY_CUDA(gemm(st, EPI_PLAIN, half, tc, tile, g, plan.out_grid), "the attention output projection");
            }
            TRY_CUDA(add_layer_norm(st, s.x, s.part, layer(l, TURBO_BERT_ATTN_OUT_BIAS),
                                    layer(l, TURBO_BERT_ATTN_LN_WEIGHT), layer(l, TURBO_BERT_ATTN_LN_BIAS), eps, info,
                                    h, s.x16, plan),
                     "the attention LayerNorm");
        }

        g.a = xin;
        g.w = weight(l, TURBO_BERT_FFN_IN_WEIGHT);
        g.bias = layer(l, TURBO_BERT_FFN_IN_BIAS);
        g.out = s.ffn;
        g.n = inter;
        if (s.cublas & CUBLAS_FFN1) {
            TRY_CUBLAS(linear(blas, half, xin, tokens, h, g.w, inter, s.raw), "the feed-forward input");
            TRY_CUDA(gelu_epilogue(st, s.raw, g, half, plan), "GELU");
        } else {
            TRY_CUDA(gemm(st, EPI_GELU, half, tc, tile, g, plan.ffn1_grid), "the feed-forward input");
        }

        g.a = s.ffn;
        g.w = weight(l, TURBO_BERT_FFN_OUT_WEIGHT);
        g.n = h;
        g.k = inter;
        if (plan.fused_ln && !(s.cublas & CUBLAS_FFN2)) {
            g.bias = layer(l, TURBO_BERT_FFN_OUT_BIAS);
            g.out = s.x;
            g.ln_w = layer(l, TURBO_BERT_FFN_LN_WEIGHT);
            g.ln_b = layer(l, TURBO_BERT_FFN_LN_BIAS);
            TRY_CUDA(gemm(st, EPI_ADD_LN, half, tc, tile, g, plan.ffn2_grid), "the feed-forward output and LayerNorm");
        } else {
            g.bias = nullptr;
            g.out = s.part;
            if (s.cublas & CUBLAS_FFN2) {
                TRY_CUBLAS(linear(blas, half, s.ffn, tokens, inter, g.w, h, s.part), "the feed-forward output");
            } else {
                TRY_CUDA(gemm(st, EPI_PLAIN, half, tc, tile, g, plan.ffn2_grid), "the feed-forward output");
            }
            TRY_CUDA(add_layer_norm(st, s.x, s.part, layer(l, TURBO_BERT_FFN_OUT_BIAS),
                                    layer(l, TURBO_BERT_FFN_LN_WEIGHT), layer(l, TURBO_BERT_FFN_LN_BIAS), eps, info,
                                    h, s.x16, plan),
                     "the feed-forward LayerNorm");
        }
    }
    TRY_CUDA(pool(st, s.x, s.rows, s.pk, h, static_cast<float *>(s.output.ptr), plan), "pooling");
    return TURBO_OK;
}

/* The run captured into the session's graph, instantiated and uploaded,
 * and its pack kernel's node found. */
int32_t capture(Session &s, turbo_error *err) {
    cudaStream_t st = s.ctx->stream;
    TRY_CUDA(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capturing the run");
    const int32_t rc = encode(s, err);
    cudaGraph_t graph = nullptr;
    const cudaError_t e = cudaStreamEndCapture(st, &graph);
    if (rc != TURBO_OK) {
        if (graph) cudaGraphDestroy(graph);
        (void)cudaGetLastError();
        return rc;
    }
    TRY_CUDA(e, "capturing the run");
    s.graph = graph;
    size_t n = 0;
    TRY_CUDA(cudaGraphGetNodes(graph, nullptr, &n), "the run's graph");
    std::vector<cudaGraphNode_t> nodes(n);
    TRY_CUDA(cudaGraphGetNodes(graph, nodes.data(), &n), "the run's graph");
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type;
        TRY_CUDA(cudaGraphNodeGetType(node, &type), "the run's graph");
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams kp;
        TRY_CUDA(cudaGraphKernelNodeGetParams(node, &kp), "the run's graph");
        if (kp.func == pack_rows_function()) s.pack_node = node;
        if (kp.func == fetch_rows_function()) s.fetch_node = node;
    }
    if (!s.pack_node || !s.fetch_node)
        return refuse(err, TURBO_E_INTERNAL, "the run's graph has no packing or fetch kernel");
    TRY_CUDA(cudaGraphInstantiateWithFlags(&s.exec, graph, 0), "cudaGraphInstantiate");
    TRY_CUDA(cudaGraphUpload(s.exec, st), "cudaGraphUpload");
    TRY_CUDA(cudaStreamSynchronize(st), "cudaGraphUpload");
    return TURBO_OK;
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
        // The QKV GEMM reads Q, K and V's weights and biases as one: every
        // copy of the weights is laid out so (lay_out).
        for (uint32_t l = 0; l < d.layers; l++) {
            const size_t q = TURBO_BERT_EMBEDDING_TENSORS + (size_t)l * TURBO_BERT_LAYER_TENSORS;
            const size_t hh = (size_t)d.hidden * d.hidden;
            const bool bias = m->f32[q + TURBO_BERT_K_BIAS] == m->f32[q + TURBO_BERT_Q_BIAS] + d.hidden &&
                              m->f32[q + TURBO_BERT_V_BIAS] == m->f32[q + TURBO_BERT_Q_BIAS] + 2 * d.hidden;
            const bool wide = m->f32[q + TURBO_BERT_K_WEIGHT] == m->f32[q + TURBO_BERT_Q_WEIGHT] + hh &&
                              m->f32[q + TURBO_BERT_V_WEIGHT] == m->f32[q + TURBO_BERT_Q_WEIGHT] + 2 * hh;
            const bool narrow = !half || (m->f16[q + TURBO_BERT_K_WEIGHT] == m->f16[q + TURBO_BERT_Q_WEIGHT] + hh &&
                                          m->f16[q + TURBO_BERT_V_WEIGHT] == m->f16[q + TURBO_BERT_Q_WEIGHT] + 2 * hh);
            if (!bias || !wide || !narrow)
                return refuse(err, TURBO_E_INTERNAL, "layer %u's Q, K and V are not laid out back to back", l);
        }

        ON_DEVICE(c->ordinal);
        Shape sh;
        sh.batch_cap = (int)max_batch;
        sh.seq_cap = (int)max_seq;
        sh.tcap = (int)tokens;
        sh.hidden = (int)d.hidden;
        sh.heads = (int)d.heads;
        sh.inter = (int)d.intermediate;
        sh.half = half;
        int major = 0, sms = 0, optin = 0;
        TRY_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, c->ordinal), "the device's sm");
        TRY_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, c->ordinal), "the device's SMs");
        TRY_CUDA(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, c->ordinal),
                 "the device's shared memory");
        // The tensor cores: F16 at FASTEST; TF32 for F32 products when
        // TURBO_CUDA_TF32=1 asks, but at EXACT, F32 FMAs throughout.
        sh.tensor_cores = major >= 8 && (half || (precision != TURBO_PRECISION_EXACT && tf32_named()));
        sh.sms = sms;
        sh.smem_optin = (size_t)optin;
        sh.tile = tile_named();
        sh.split_attention = split_attention_named();
        sh.sk_steps = sk_steps_named();
        sh.fused_ln = !separate_ln_named();
        sh.column_pool = column_pool_named();
        Plan plan;
        TRY_CUDA(make_plan(sh, &plan), "planning the session's launches");
        if (plan.gemm_crowded)
            c->say(LOG_WARNING,
                   "cuda device %d: a GEMM's kernel fits fewer blocks to an SM than it was built for, so it runs "
                   "slower than it should",
                   c->ordinal);

        Session *s = make<Session>();
        s->model = m;
        s->ctx = c;
        s->max_batch = max_batch;
        s->max_seq = max_seq;
        s->half = half;
        s->shape = sh;
        s->plan = plan;
        s->cublas = cublas_gemms();
        const size_t act = half ? 2 : 4;
        const size_t h = d.hidden, inter = d.intermediate;
        const size_t ints = round_up(tokens * 4, DEVICE_ALIGN);
        const size_t rows = round_up(((size_t)max_batch + 1) * 4, DEVICE_ALIGN);
        const size_t info = round_up(sizeof(Info), DEVICE_ALIGN);
        const size_t x = round_up(tokens * h * 4, DEVICE_ALIGN);
        const size_t x16 = half ? round_up(tokens * h * 2, DEVICE_ALIGN) : 0;
        const size_t qkv = round_up(3 * tokens * h * act, DEVICE_ALIGN);
        const size_t att = round_up(tokens * h * act, DEVICE_ALIGN);
        const size_t ffn = round_up(tokens * inter * act, DEVICE_ALIGN);
        const size_t part = round_up(tokens * h * 4, DEVICE_ALIGN);
        const size_t ws = round_up(plan.sk_floats * 4, DEVICE_ALIGN);
        const size_t flags = round_up((size_t)(plan.sk_flags + plan.ln_counts) * 4, DEVICE_ALIGN);
        const size_t widest = 3 * h > inter ? 3 * h : inter;
        const size_t raw = s->cublas & (CUBLAS_QKV | CUBLAS_FFN1) ? round_up(tokens * widest * 4, DEVICE_ALIGN) : 0;
        const size_t output = round_up((size_t)max_batch * h * 4, DEVICE_ALIGN);
        const size_t total = 5 * ints + 5 * rows + info + x + x16 + qkv + att + ffn + part + ws + flags + raw + output;
        const int32_t rc = [&]() -> int32_t {
            TRY_CUDA(device_malloc(&s->scratch, total), "device memory for the session");
            TRY_CUDA(pinned_malloc(reinterpret_cast<void **>(&s->staging), 3 * tokens * 4 + 16),
                     "page-locked memory for the session's rows");
            TRY_CUDA(cudaEventCreateWithFlags(&s->sent, cudaEventDisableTiming), "cudaEventCreateWithFlags");
            char *p = static_cast<char *>(s->scratch);
            auto take = [&](size_t n) {
                char *here = p;
                p += n;
                return here;
            };
            s->rows = reinterpret_cast<int32_t *>(take(3 * ints));
            s->pk.tok_row = reinterpret_cast<int32_t *>(take(ints));
            s->pk.key_bias = reinterpret_cast<float *>(take(ints));
            s->pk.start = reinterpret_cast<int32_t *>(take(rows));
            s->pk.len = reinterpret_cast<int32_t *>(take(rows));
            s->pk.holes = reinterpret_cast<int32_t *>(take(rows));
            s->pk.order = reinterpret_cast<int32_t *>(take(rows));
            s->pk.item_start = reinterpret_cast<int32_t *>(take(rows));
            s->pk.info = reinterpret_cast<Info *>(take(info));
            s->x = reinterpret_cast<float *>(take(x));
            if (half) s->x16 = reinterpret_cast<uint16_t *>(take(x16));
            s->qkv = take(qkv);
            s->att = take(att);
            s->ffn = take(ffn);
            s->part = reinterpret_cast<float *>(take(part));
            s->ws = reinterpret_cast<float *>(take(ws));
            s->flags = reinterpret_cast<int *>(take(flags));
            if (raw) s->raw = reinterpret_cast<float *>(take(raw));
            s->output.ctx = c;
            s->output.ptr = take(output);
            s->output.placement = TURBO_PLACE_DEVICE;
            s->output.bytes = (uint64_t)max_batch * d.hidden * 4;
            s->output.owned = false;
            s->pack.rows = s->rows;
            s->pack.heads = (int32_t)d.heads;
            s->pack.queries = plan.attn_queries;
            s->pack.p = s->pk;
            void *staged = nullptr;
            TRY_CUDA(cudaHostGetDevicePointer(&staged, s->staging, 0), "the staging's device address");
            s->fetch.src = static_cast<const int32_t *>(staged);
            s->fault = s->staging + 3 * tokens;
            s->fault_dev = static_cast<int32_t *>(staged) + 3 * tokens;
            *s->fault = 0;
            s->fetch.dst = s->rows;
            s->fetch.n = 0;
            std::lock_guard<std::mutex> g(c->lock);
            // Rows past a run's tokens are never read into an output; zeroed
            // once, they hold finite values whatever reads them.
            TRY_CUDA(cudaMemsetAsync(s->scratch, 0, total, c->stream), "clearing the session's memory");
            TRY_CUDA(cudaStreamSynchronize(c->stream), "clearing the session's memory");
            if (!s->cublas) TRY(capture(*s, err));
            return TURBO_OK;
        }();
        if (rc != TURBO_OK) {
            release_session(s);
            return rc;
        }
        c->say(LOG_DEBUG,
               "cuda device %d: an embed session of %u rows of %u tokens, computing in %s%s; attention %zu bytes of "
               "shared memory for %d keys at a time",
               c->ordinal, max_batch, max_seq, half ? "F16 with F32 accumulation" : "F32",
               s->cublas ? ", some GEMMs on cuBLAS, without a graph" : ", as one graph", plan.attn_smem,
               plan.attn_chunk);
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

/* The rows to the device, into s.rows as [k][batch][seq]. When every
 * array is page-locked or managed and its rows are back to back, each is
 * one contiguous copy from the caller's memory, with *direct set, and the
 * run's fetch kernel does nothing. Otherwise the arrays are laid out in
 * the page-locked staging as on the device and the run's fetch kernel
 * reads them from there, so no copy is queued ahead of the run. Device
 * memory is refused: the core has read and checked the rows on the host,
 * so they are host memory, page-locked, managed or pageable. *sent is the
 * bytes sent. */
int32_t upload(Session &s, const turbo_backend_embed_rows *r, uint64_t *sent, bool *direct, turbo_error *err) {
    const int32_t *src[3] = {r->ids, r->mask, r->types};
    const int k = r->types ? 3 : 2;
    const size_t row = (size_t)r->seq * 4, plane = (size_t)r->batch * r->seq;
    bool pinned = r->row_stride == r->seq;
    for (int i = 0; i < k; i++) {
        const cudaPointerAttributes a = attributes(src[i]);
        if (a.type == cudaMemoryTypeDevice)
            return refuse(err, TURBO_E_INVALID_ARGUMENT, "rows are device memory; the core reads rows on the host");
        if (a.type != cudaMemoryTypeHost && a.type != cudaMemoryTypeManaged) pinned = false;
    }
    *sent = (uint64_t)k * row * r->batch;
    if (pinned) {
        for (int i = 0; i < k; i++)
            TRY_CUDA(cudaMemcpyAsync(s.rows + i * plane, src[i], plane * 4, cudaMemcpyHostToDevice, s.ctx->stream),
                     "sending the rows");
        *direct = true;
        s.fetch.n = 0;
        return TURBO_OK;
    }
    for (int i = 0; i < k; i++) {
        int32_t *at = s.staging + i * plane;
        if (r->row_stride == r->seq)
            memcpy(at, src[i], row * r->batch);
        else
            for (uint32_t b = 0; b < r->batch; b++)
                memcpy(at + (size_t)b * r->seq, src[i] + (size_t)b * r->row_stride, row);
    }
    s.fetch.n = (int32_t)(k * plane);
    return TURBO_OK;
}

int32_t embed_write(void *session, const turbo_backend_embed_rows *r, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Session &s = *static_cast<Session *>(session);
        s.written = false;
        // The packed size, which only cuBLAS's GEMMs take from the host:
        // each row through its last live token, as pack_rows finds it on
        // the device from the same entries, the host memory the core read.
        uint32_t packed = 0;
        if (s.cublas)
            for (uint32_t b = 0; b < r->batch; b++) {
                const int32_t *m = r->mask + (size_t)b * r->row_stride;
                uint32_t n = r->seq;
                while (n > 0 && m[n - 1] == 0) n--;
                packed += n;
            }
        uint64_t sent = 0;
        {
            std::lock_guard<std::mutex> g(s.ctx->lock);
            ON_DEVICE(s.ctx->ordinal);
            // Nothing reads the staging or the device's rows now: a run
            // waits for its end, and a write for its own copies.
            bool direct = false;
            const int32_t rc = upload(s, r, &sent, &direct, err);
            if (rc != TURBO_OK) {
                // Nothing queued may read memory the next write reuses.
                (void)cudaStreamSynchronize(s.ctx->stream);
                return rc;
            }
            // Rows copied from the caller's page-locked or managed memory
            // are read by the device until the copy ends, and the caller's
            // arrays are valid for this call only: wait for them. Rows in
            // the staging are the session's, and the run reads them.
            if (direct) {
                TRY_CUDA(cudaEventRecord(s.sent, s.ctx->stream), "sending the rows");
                TRY_CUDA(cudaEventSynchronize(s.sent), "sending the rows");
            }
        }
        s.run.batch = (int32_t)r->batch;
        s.run.seq = (int32_t)r->seq;
        s.run.pooling = (int32_t)r->pooling;
        s.run.l2 = r->normalize == TURBO_NORMALIZE_L2 ? 1 : 0;
        s.run.output_dim = (int32_t)r->output_dim;
        s.run.has_types = r->types ? 1 : 0;
        s.tokens = packed;
        s.h2d = sent;
        s.written = true;
        return TURBO_OK;
    });
}

/* The written rows through the encoder: the graph, with this run's rows'
 * shape and options set in its fetch and pack kernels when they differ
 * from the last run's, or the launches one by one when cuBLAS computes a
 * GEMM. */
int32_t launch(Session &s, turbo_error *err) {
    s.pack.run = s.run;
    if (s.cublas) return encode(s, err);
    if (!s.graph_set || memcmp(&s.in_graph, &s.run, sizeof s.run) != 0 || s.fetch_in_graph != s.fetch.n) {
        // Marked unset first: a failure leaves the graph's arguments unknown.
        s.graph_set = false;
        cudaKernelNodeParams kp;
        pack_rows_node(&s.pack, s.pack_argv, s.plan, &kp);
        TRY_CUDA(cudaGraphExecKernelNodeSetParams(s.exec, s.pack_node, &kp), "setting the run's rows");
        fetch_rows_node(&s.fetch, s.fetch_argv, s.plan, &kp);
        TRY_CUDA(cudaGraphExecKernelNodeSetParams(s.exec, s.fetch_node, &kp), "setting the run's rows");
        s.in_graph = s.run;
        s.fetch_in_graph = s.fetch.n;
        s.graph_set = true;
    }
    TRY_CUDA(cudaGraphLaunch(s.exec, s.ctx->stream), "cudaGraphLaunch");
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
            const int32_t rc = launch(s, err);
            // The stream is left idle whether or not the run finished.
            const cudaError_t e = cudaStreamSynchronize(s.ctx->stream);
            if (rc != TURBO_OK) {
                (void)cudaGetLastError();
                return rc;
            }
            TRY_CUDA(e, "the run");
            if (*reinterpret_cast<volatile int *>(s.fault)) {
                // The flags a wait gave up on may be left raised: cleared,
                // so the next run starts as the first did.
                *reinterpret_cast<volatile int *>(s.fault) = 0;
                TRY_CUDA(cudaMemsetAsync(s.flags, 0, (size_t)(s.plan.sk_flags + s.plan.ln_counts) * 4, s.ctx->stream),
                         "the run");
                TRY_CUDA(cudaStreamSynchronize(s.ctx->stream), "the run");
                return refuse(err, TURBO_E_RUNTIME,
                              "a GEMM's block waited too long for another's partial product; the run's vectors "
                              "are not valid");
            }
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
        st[TURBO_EMBED_STAGE_NORMALIZE] = s.run.l2 ? TURBO_STAGE_FUSED : TURBO_STAGE_UNUSED;
        st[TURBO_EMBED_STAGE_DOWNLOAD] = TURBO_STAGE_UNUSED;
        return TURBO_OK;
    });
}

// ---- The GEMMs against cuBLAS ------------------------------------------------------
//
// One GEMM of the backend's own on random operands, and cuBLAS's product
// of the same operands with the epilogue applied on the host, for the
// tests: the largest difference and the largest reference value.

/* Values in [-1, 1) from a fixed sequence. */
struct Lcg {
    uint64_t x = 0x9e3779b97f4a7c15ull;
    float next() {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        return (float)((x >> 40) & 0xffffff) / 8388608.0f - 1.0f;
    }
};

int32_t gemm_check(uint32_t ordinal, int32_t m, int32_t n, int32_t k, int32_t epilogue, int32_t half,
                   int32_t tensor_cores, int32_t tile_in, int32_t blocks, int32_t heads, double *max_diff,
                   double *max_ref) {
    turbo_error e0{};
    turbo_error *err = &e0;
    ON_DEVICE((int)ordinal);
    const Epilogue epi = (Epilogue)epilogue;
    const bool h16 = half != 0, tc = tensor_cores != 0;
    const Tile tile = (Tile)tile_in;
    const int hidden = epi == EPI_QKV ? n / 3 : n;
    int sms = 0, grid = 0;
    size_t ws_floats = 0;
    TRY_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, (int)ordinal), "the device's SMs");
    TRY_CUDA(gemm_prepare(epi, h16, tc, tile), "gemm_prepare");
    TRY_CUDA(gemm_grid(epi, h16, tc, tile, sms, &grid, &ws_floats), "gemm_grid");
    // Fewer blocks than the device holds, when asked: other shares of the
    // same work. Never more, which could wait on a block not yet running.
    if (blocks > 0 && blocks < grid) grid = blocks;
    GemmArgs g{};
    g.n = n;
    g.k = k;
    g.heads = heads;
    g.hidden = hidden;
    g.head_dim = hidden / heads;
    g.tcap = m;
    const size_t in = h16 ? 2 : 4, outb = epi == EPI_PLAIN ? 4 : in;
    const size_t out_n = (size_t)m * n;
    std::vector<float> a((size_t)m * k), w((size_t)n * k), bias(n);
    Lcg r;
    for (float &v : a) v = r.next();
    for (float &v : w) v = r.next();
    for (float &v : bias) v = r.next();
    std::vector<uint16_t> a16, w16;
    if (h16) {
        for (float v : a) a16.push_back(__half_as_ushort(__float2half_rn(v)));
        for (float v : w) w16.push_back(__half_as_ushort(__float2half_rn(v)));
    }
    char *dev = nullptr;
    const size_t sa = round_up(a.size() * in, DEVICE_ALIGN), sw = round_up(w.size() * in, DEVICE_ALIGN);
    const size_t sb = round_up((size_t)n * 4, DEVICE_ALIGN), so = round_up(out_n * outb, DEVICE_ALIGN);
    const size_t sr = round_up((size_t)m * n * 4, DEVICE_ALIGN), si = round_up(sizeof(Info), DEVICE_ALIGN);
    const size_t sws = round_up(ws_floats * 4, DEVICE_ALIGN), sf = round_up((size_t)grid * 4 + 4, DEVICE_ALIGN);
    const size_t total = sa + sw + sb + so + sr + si + sws + sf;
    TRY_CUDA(cudaMalloc(reinterpret_cast<void **>(&dev), total), "cudaMalloc");
    cudaStream_t st = nullptr;
    cublasHandle_t blas = nullptr;
    const int32_t rc = [&]() -> int32_t {
        TRY_CUDA(cudaStreamCreate(&st), "cudaStreamCreate");
        TRY_CUBLAS(cublasCreate(&blas), "cublasCreate");
        TRY_CUBLAS(cublasSetStream(blas, st), "cublasSetStream");
        TRY_CUBLAS(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
        char *pa = dev, *pw = pa + sa, *pb = pw + sw, *po = pb + sb, *pr = po + so, *pi = pr + sr;
        char *pws = pi + si, *pf = pws + sws;
        TRY_CUDA(cudaMemset(dev, 0, total), "cudaMemset");
        TRY_CUDA(cudaMemcpy(pa, h16 ? (const void *)a16.data() : (const void *)a.data(), a.size() * in,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy");
        TRY_CUDA(cudaMemcpy(pw, h16 ? (const void *)w16.data() : (const void *)w.data(), w.size() * in,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy");
        TRY_CUDA(cudaMemcpy(pb, bias.data(), (size_t)n * 4, cudaMemcpyHostToDevice), "cudaMemcpy");
        Info info{};
        info.tokens = m;
        TRY_CUDA(cudaMemcpy(pi, &info, sizeof info, cudaMemcpyHostToDevice), "cudaMemcpy");
        g.a = pa;
        g.w = pw;
        g.bias = reinterpret_cast<const float *>(pb);
        g.out = po;
        g.info = reinterpret_cast<const Info *>(pi);
        g.ws = reinterpret_cast<float *>(pws);
        g.flags = reinterpret_cast<int *>(pf);
        g.fault = g.flags + grid;
        std::vector<char> got(out_n * outb), again(out_n * outb);
        // Twice: the second launch must find the flags the first cleared
        // and repeat its bits.
        TRY_CUDA(gemm(st, epi, h16, tc, tile, g, grid), "the GEMM");
        TRY_CUDA(cudaMemcpyAsync(again.data(), po, again.size(), cudaMemcpyDeviceToHost, st), "cudaMemcpy");
        TRY_CUDA(cudaMemsetAsync(po, 0, again.size(), st), "cudaMemset");
        TRY_CUDA(gemm(st, epi, h16, tc, tile, g, grid), "the GEMM");
        TRY_CUBLAS(linear(blas, h16, pa, m, k, pw, n, reinterpret_cast<float *>(pr)), "cuBLAS");
        TRY_CUDA(cudaStreamSynchronize(st), "the GEMMs");
        std::vector<float> ref((size_t)m * n);
        TRY_CUDA(cudaMemcpy(ref.data(), pr, ref.size() * 4, cudaMemcpyDeviceToHost), "cudaMemcpy");
        TRY_CUDA(cudaMemcpy(got.data(), po, got.size(), cudaMemcpyDeviceToHost), "cudaMemcpy");
        int fault = 0;
        TRY_CUDA(cudaMemcpy(&fault, g.fault, 4, cudaMemcpyDeviceToHost), "cudaMemcpy");
        if (fault) return refuse(err, TURBO_E_INTERNAL, "a block of the GEMM gave up waiting");
        if (got != again)
            return refuse(err, TURBO_E_INTERNAL, "the GEMM gave other bits when launched again");
        auto value = [&](size_t i) -> float {
            if (outb == 4) return reinterpret_cast<const float *>(got.data())[i];
            return __half2float(__ushort_as_half(reinterpret_cast<const uint16_t *>(got.data())[i]));
        };
        double worst = 0, largest = 0;
        for (int t = 0; t < m; t++)
            for (int c = 0; c < n; c++) {
                float want = ref[(size_t)t * n + c], have = 0;
                if (epi == EPI_PLAIN) {
                    have = value((size_t)t * n + c);
                } else if (epi == EPI_GELU) {
                    const float v = want + bias[c];
                    want = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
                    have = value((size_t)t * n + c);
                } else {
                    want += bias[c];
                    const int which = c / hidden, hc = c % hidden, head = hc / g.head_dim, dd = hc % g.head_dim;
                    have = value((((size_t)(which * heads + head)) * m + t) * g.head_dim + dd);
                }
                if (outb == 2) want = __half2float(__float2half_rn(want));
                const double diff = fabs((double)have - (double)want);
                worst = diff > worst ? diff : worst;
                largest = fabs(want) > largest ? fabs(want) : largest;
            }
        *max_diff = worst;
        *max_ref = largest;
        return TURBO_OK;
    }();
    if (blas) cublasDestroy(blas);
    if (st) cudaStreamDestroy(st);
    cudaFree(dev);
    return rc;
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
    TURBO_FORMAT_BIT(TURBO_FORMAT_SAFETENSORS),
    0,
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

/* One of the backend's GEMMs against cuBLAS on random operands, on
 * device ordinal: epilogue is an Epilogue, half F16 operands, tensor_cores
 * whether the GEMM takes mma.sync (F16, or TF32 for F32 operands; else
 * FMAs), tile a Tile, blocks
 * the launch's blocks (0 for what the device holds at once, and never
 * more), heads the QKV epilogue's heads, n being 3 * hidden. The GEMM
 * runs twice, and must give the same bits both times. The largest
 * absolute difference from cuBLAS's product with the epilogue done on the
 * host (rounded to F16 where the GEMM's output is), and the largest
 * reference value. */
int32_t turbo_cuda_gemm_check(uint32_t ordinal, int32_t m, int32_t n, int32_t k, int32_t epilogue, int32_t half,
                              int32_t tensor_cores, int32_t tile, int32_t blocks, int32_t heads, double *max_diff,
                              double *max_ref) {
    try {
        return gemm_check(ordinal, m, n, k, epilogue, half, tensor_cores, tile, blocks, heads, max_diff, max_ref);
    } catch (...) {
        return TURBO_E_INTERNAL;
    }
}

/* The GEMMs sessions made from now on hand to cuBLAS, as TURBO_CUDA_CUBLAS
 * would name them (1 QKV, 2 attention output, 4 feed-forward input, 8
 * feed-forward output); -1 to read the variable again. */
void turbo_cuda_use_cublas(int32_t gemms) { cublas_override.store(gemms, std::memory_order_relaxed); }

/* The GEMMs' tile in sessions made from now on, as TURBO_CUDA_TILE would
 * name it (a Tile); -1 to read the variable again. */
void turbo_cuda_use_tile(int32_t tile) { tile_override.store(tile, std::memory_order_relaxed); }

/* The FMA attention of sessions made from now on: 1 the key-split kernel
 * (TURBO_CUDA_ATTENTION=split), 0 the default, -1 to read the variable
 * again. */
void turbo_cuda_use_split_attention(int32_t split) {
    split_attention_override.store(split, std::memory_order_relaxed);
}

/* The LayerNorms of sessions made from now on: 1 a kernel of their own
 * after the GEMM (the default), 0 the GEMM's epilogue
 * (TURBO_CUDA_LAYER_NORM=fused), -1 to read the variable again. */
void turbo_cuda_use_separate_layer_norm(int32_t separate) {
    separate_ln_override.store(separate, std::memory_order_relaxed);
}

/* The F32 GEMMs of MODEL sessions made from now on: 1 TF32 on the tensor
 * cores (TURBO_CUDA_TF32=1), 0 the FMA kernels (the default), -1 to read
 * the variable again. */
void turbo_cuda_use_tf32(int32_t tf32) { tf32_override.store(tf32, std::memory_order_relaxed); }

/* The pooling of sessions made from now on: 1 the kernel of a thread per
 * column (TURBO_CUDA_POOL=columns), 0 the default, -1 to read the
 * variable again. */
void turbo_cuda_use_column_pool(int32_t columns) { column_pool_override.store(columns, std::memory_order_relaxed); }

} // extern "C"
