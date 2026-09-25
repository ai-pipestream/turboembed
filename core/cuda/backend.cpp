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

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <strings.h>
#include <utility>
#include <vector>

#include "autotune.h"
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
        // Room for a session's choices line and what the tuner says of it.
        char m[2048];
        va_list ap;
        va_start(ap, fmt);
        const int n = vsnprintf(m, sizeof m, fmt, ap);
        va_end(ap);
        const size_t len = n < 0 ? 0 : (size_t)n < sizeof m ? (size_t)n : sizeof m - 1;
        log(log_user_data, level, turbo_text{m, len});
    }
};

constexpr uint32_t LOG_WARNING = 1;
constexpr uint32_t LOG_INFO = 2;
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
// TURBO_CUDA_GELU=poly, read when a session is made, gives FASTEST's GELU
// erf from a fit (gelu_f16 in kernels.cu) in place of the default's erff,
// which an F32 output's always is; TURBO_CUDA_GELU=erf names the default.
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

// ---- Choices from the environment ------------------------------------------------
//
// Each switch below is read when a session is made. The tests set an
// override in place of each variable (turbo_cuda_use_*), -1 to read the
// variable again. A switch that is set forces its knob in every token
// bin; one that is not leaves the knob to the defaults.

std::atomic<int> cublas_override{-1};
std::atomic<int> tile_override{-1};
std::atomic<int> split_attention_override{-1};
std::atomic<int> wide_attention_override{-1};
std::atomic<int> exact_attention_override{-1};
std::atomic<int> fa32_attention_override{-1};
std::atomic<int> tf32_override{-1};
std::atomic<int> f16_accumulate_override{-1};
std::atomic<int> separate_ln_override{-1};
std::atomic<int> column_pool_override{-1};
std::atomic<int> gelu_erf_override{-1};
/* In place of TURBO_CUDA_SK_STEPS: 0 the kernels' own, 1 to 64 steps,
 * SK_OVERRIDE_TILES whole tiles; -1 for the variable. */
std::atomic<int> sk_override{-1};
constexpr int SK_OVERRIDE_TILES = -2;

int overridden(const std::atomic<int> &o) { return o.load(std::memory_order_relaxed); }

/* TURBO_CUDA_TILE by name; TILE_DEFAULT, not forced, when unset or not a
 * name it takes. The F16 accumulators' eight-warp tiles are
 * TURBO_CUDA_F16_ACCUMULATE's; a whole-k tile it names (an f16k one) sets
 * that experiment as the switch does. */
bool tile_named(Tile *out) {
    const int o = overridden(tile_override);
    if (o >= 0) {
        *out = (Tile)o;
        return true;
    }
    const char *v = getenv("TURBO_CUDA_TILE");
    if (!v) return false;
    for (const TileName &n : TILE_NAMES)
        if (n.switch_names && !strcasecmp(v, n.name)) {
            *out = n.tile;
            return true;
        }
    return false;
}

/* TURBO_CUDA_ATTENTION: split gives an F32 session (and an F16 one without
 * the tensor cores' attention) the FMA attention that splits each query's
 * keys among four warps; 64 gives FASTEST's attention on the tensor cores
 * 64 queries to a block in place of 128. */
bool split_attention_named(bool *forced) {
    const int o = overridden(split_attention_override);
    if (o >= 0) {
        *forced = true;
        return o != 0;
    }
    const char *v = getenv("TURBO_CUDA_ATTENTION");
    *forced = *forced || v;
    return v && !strcasecmp(v, "split");
}

/* On an RTX 4080 SUPER at 32 x 256 full rows the kernel of 128 takes 324
 * us a pass against 398 for 64's. */
bool wide_attention_named(bool *forced) {
    const int o = overridden(wide_attention_override);
    if (o >= 0) {
        *forced = true;
        return o != 0;
    }
    const char *v = getenv("TURBO_CUDA_ATTENTION");
    *forced = *forced || v;
    return !(v && !strcmp(v, "64"));
}

/* TURBO_CUDA_ATTENTION=exact: the attention of 128 queries with its
 * earlier softmax, exp2f on scores scaled before the largest is taken,
 * in place of ex2.approx with the scale in the exponent's multiply-add.
 * It gives the earlier bits, for comparing against the default. */
bool exact_attention_named(bool *forced) {
    const int o = overridden(exact_attention_override);
    if (o >= 0) {
        *forced = true;
        return o != 0;
    }
    const char *v = getenv("TURBO_CUDA_ATTENTION");
    *forced = *forced || v;
    return v && !strcasecmp(v, "exact");
}

/* TURBO_CUDA_ATTENTION=fa32: at heads of 32, the attention of 128
 * queries with 32 queries to each of four warps, for measuring against
 * the default; the same bits. Heads of 64 keep the default. */
bool fa32_attention_named(bool *forced) {
    const int o = overridden(fa32_attention_override);
    if (o >= 0) {
        *forced = true;
        return o != 0;
    }
    const char *v = getenv("TURBO_CUDA_ATTENTION");
    *forced = *forced || v;
    return v && !strcasecmp(v, "fa32");
}

/* TURBO_CUDA_TF32=1 puts MODEL's F32 GEMMs on the tensor cores as TF32;
 * unset, they are EXACT's FMA kernels. */
bool tf32_named() {
    const int o = overridden(tf32_override);
    if (o >= 0) return o != 0;
    const char *v = getenv("TURBO_CUDA_TF32");
    return v && strcmp(v, "1") == 0;
}

/* TURBO_CUDA_F16_ACCUMULATE=1 gives FASTEST's GEMMs on the tensor cores
 * F16 accumulators over each 64 terms of k, added into F32 ones
 * (TILE_EIGHT_WARPS_F16_ACCUMULATE, or TILE_SWIZZLED_8W_F16_ACCUMULATE
 * when TURBO_CUDA_TILE=sw8w, whatever other tile it names); unset, F32
 * accumulators throughout. */
bool f16_accumulate_named() {
    const int o = overridden(f16_accumulate_override);
    if (o >= 0) return o != 0;
    const char *v = getenv("TURBO_CUDA_F16_ACCUMULATE");
    return v && strcmp(v, "1") == 0;
}

/* TURBO_CUDA_LAYER_NORM=fused: the attention output and second
 * feed-forward GEMMs' epilogue adds the bias and residual and runs the
 * LayerNorm, in place of the default's kernel of their own after the
 * product (the same bits either way). */
bool separate_ln_named(bool *forced) {
    const int o = overridden(separate_ln_override);
    if (o >= 0) {
        *forced = true;
        return o != 0;
    }
    const char *v = getenv("TURBO_CUDA_LAYER_NORM");
    *forced = v != nullptr;
    return !(v && !strcasecmp(v, "fused"));
}

/* TURBO_CUDA_POOL=columns pools with a thread per column, in place of the
 * default's groups of tokens summed apart. */
bool column_pool_named(bool *forced) {
    const int o = overridden(column_pool_override);
    if (o >= 0) {
        *forced = true;
        return o != 0;
    }
    const char *v = getenv("TURBO_CUDA_POOL");
    *forced = v != nullptr;
    return v && !strcasecmp(v, "columns");
}

/* TURBO_CUDA_GELU=poly gives an F16 output's GELU the fit; unset or erf,
 * the default's erff. */
bool gelu_erf_named() {
    const int o = overridden(gelu_erf_override);
    if (o >= 0) return o != 0;
    const char *v = getenv("TURBO_CUDA_GELU");
    return !(v && !strcasecmp(v, "poly"));
}

/* TURBO_CUDA_SK_STEPS: the GEMMs' fewest k steps per block, a count from
 * 1 to 64, or tiles for whole tiles to a block; not forced when unset or
 * neither. */
bool sk_named(GemmChoice *c) {
    const int o = overridden(sk_override);
    if (o == SK_OVERRIDE_TILES || (o < 0 && getenv("TURBO_CUDA_SK_STEPS") &&
                                   !strcasecmp(getenv("TURBO_CUDA_SK_STEPS"), "tiles"))) {
        c->sk = SK_TILES;
        c->sk_steps = 0;
        return true;
    }
    if (o >= 0) {
        c->sk = SK_STREAM;
        c->sk_steps = o;
        return true;
    }
    const char *v = getenv("TURBO_CUDA_SK_STEPS");
    if (!v) return false;
    char *end = nullptr;
    const long n = strtol(v, &end, 10);
    if (end == v || *end != 0 || n < 1 || n > 64) return false;
    c->sk = SK_STREAM;
    c->sk_steps = (int)n;
    return true;
}

unsigned cublas_gemms() {
    const int o = overridden(cublas_override);
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

/* What the tests set in place of TURBO_CUDA_CHOICES, when set. */
std::mutex choices_lock;
bool choices_set = false;
std::string choices_override;

/* TURBO_CUDA_CHOICES, or the tests' string in its place: the items it
 * names forced over the switches, each where it names it. */
bool choices_named(Choices *c, uint32_t *absent, char *why, size_t len) {
    *absent = 0;
    std::string v;
    {
        std::lock_guard<std::mutex> g(choices_lock);
        if (choices_set) {
            v = choices_override;
        } else {
            const char *e = getenv("TURBO_CUDA_CHOICES");
            if (!e) return true;
            v = e;
        }
    }
    return parse_choices(v.c_str(), c, absent, why, len);
}

/* The numeric classes each precision allows when the core hands none (a
 * session made through session_create): the core's own table. */
uint32_t default_numerics(uint32_t precision) {
    return precision == TURBO_PRECISION_FASTEST ? TURBO_NUMERIC_F16_F32ACC : TURBO_NUMERIC_F32_FMA;
}

const char *precision_name(uint32_t p) {
    return p == TURBO_PRECISION_FASTEST ? "FASTEST" : p == TURBO_PRECISION_EXACT ? "EXACT" : "MODEL";
}

/* Whether a session of the shape takes the tensor cores' attention. */
bool mma_attention(const Shape &base) {
    const int d = base.hidden / base.heads;
    return base.half && base.tensor_cores && (d == 32 || d == 64);
}

/* The backend's own choices for a session of the shape, in every bin: the
 * default tile and stream-K for each GEMM, the tensor cores' attention of
 * 128 queries where it applies and else the FMA kernel of register tiles,
 * the LayerNorm as a kernel of its own, and pooling by groups. */
Choices defaults(const Shape &base) {
    Choices c;
    c.bins = bins_for(base.tcap);
    for (BinChoices &b : c.bin) {
        b = BinChoices{};
        b.attention = mma_attention(base) ? ATT_MMA_128 : ATT_FMA_TILED;
    }
    return c;
}

/* The environment's and the tests' switches over *c, each forcing its knob
 * in every bin. *tf32 and *f16_accumulate say whether the switch of that
 * name is in effect for the session: TF32 needs F32 operands on a device
 * with tensor cores at a precision other than EXACT, F16 accumulators F16
 * operands on one. In a tuned session the two widen the kernels the tuner
 * may time instead of forcing their own, unless a tile is forced too. */
void forced_from_environment(const Shape &base, uint32_t precision, bool tuned, Choices *c, bool *tf32,
                             bool *f16_accumulate) {
    Tile tile = TILE_DEFAULT;
    const bool tile_forced = tile_named(&tile);
    *f16_accumulate = base.half && base.tensor_cores && (f16_accumulate_named() || f16_accumulates(tile));
    const bool acc_forced = *f16_accumulate && !f16_accumulates(tile) && (!tuned || tile_forced);
    if (acc_forced) tile = tile == TILE_SWIZZLED_8W ? TILE_SWIZZLED_8W_F16_ACCUMULATE : TILE_EIGHT_WARPS_F16_ACCUMULATE;
    *tf32 = !base.half && base.tensor_cores && precision != TURBO_PRECISION_EXACT && tf32_named();
    GemmChoice sk;
    const bool sk_forced = sk_named(&sk);
    bool attn_forced = false;
    const bool split = split_attention_named(&attn_forced), wide = wide_attention_named(&attn_forced);
    const bool exact = exact_attention_named(&attn_forced), fa32 = fa32_attention_named(&attn_forced);
    bool ln_forced = false, pool_forced = false;
    const bool separate = separate_ln_named(&ln_forced);
    const bool columns = column_pool_named(&pool_forced);
    for (int b = 0; b < BIN_COUNT; b++) {
        BinChoices &bc = c->bin[b];
        for (int g = 0; g < GEMM_COUNT; g++) {
            GemmChoice &gc = bc.gemm[g];
            if (tile_forced || acc_forced) {
                gc.tile = tile;
                bc.gemm_forced[g] |= KNOB_TILE;
            }
            if (sk_forced) {
                gc.sk = sk.sk;
                gc.sk_steps = sk.sk_steps;
                bc.gemm_forced[g] |= KNOB_SK;
            }
            if (*tf32 && (!tuned || tile_forced)) {
                gc.tf32 = true;
                bc.gemm_forced[g] |= KNOB_TF32;
            }
        }
        if (attn_forced) {
            if (mma_attention(base))
                bc.attention = !wide ? ATT_MMA_64 : fa32 ? ATT_MMA_128_FA32 : exact ? ATT_MMA_128_EXACT : ATT_MMA_128;
            else
                bc.attention = split ? ATT_FMA_SPLIT : ATT_FMA_TILED;
            bc.forced |= KNOB_ATTN;
        }
        if (ln_forced) {
            bc.ln = separate ? LN_SEPARATE : LN_FUSED;
            bc.forced |= KNOB_LN;
        }
    }
    if (pool_forced) {
        c->pool = columns ? POOL_COLUMNS : POOL_GROUPS;
        c->pool_forced = KNOB_POOL;
    }
}

/* Whole rows take the LayerNorm in the epilogue of the attention output
 * and second feed-forward GEMMs whose tile is TILE_SWIZZLED_ROWS or
 * TILE_F16_WHOLE_K_ROWS, with F16 operands on the tensor cores; hidden
 * states wider than their tile take the eight-warp shapes instead, or
 * TILE_F16_WHOLE_K_3. */
void whole_rows(const Shape &base, Choices *c) {
    if (!base.half || !base.tensor_cores) return;
    for (BinChoices &bc : c->bin)
        for (Gemm g : {GEMM_OUT, GEMM_FFN2}) {
            const Tile t = bc.gemm[g].tile;
            if (t != TILE_SWIZZLED_ROWS && t != TILE_F16_WHOLE_K_ROWS) continue;
            if (base.hidden <= ROW_LN_WIDTH) {
                bc.ln = LN_FUSED;
                if (bc.gemm_forced[g] & KNOB_TILE) bc.forced |= KNOB_LN;
            } else {
                bc.gemm[g].tile = t == TILE_SWIZZLED_ROWS ? TILE_SWIZZLED_8W : TILE_F16_WHOLE_K_3;
            }
        }
}

/* The shape one bin's graph is captured from. */
Shape shape_for(const Shape &base, const Choices &c, int bin) {
    Shape sh = base;
    const BinChoices &bc = c.bin[bin];
    for (int g = 0; g < GEMM_COUNT; g++) sh.gemm[g] = bc.gemm[g];
    sh.split_attention = bc.attention == ATT_FMA_SPLIT;
    sh.wide_attention =
        bc.attention == ATT_MMA_128 || bc.attention == ATT_MMA_128_EXACT || bc.attention == ATT_MMA_128_FA32;
    sh.exact_exp2 = bc.attention == ATT_MMA_128_EXACT;
    sh.fa32 = bc.attention == ATT_MMA_128_FA32;
    sh.fused_ln = bc.ln == LN_FUSED;
    sh.column_pool = c.pool == POOL_COLUMNS;
    return sh;
}

/* Whether two bins run the same kernels, and so can share a graph. */
bool same_kernels(const BinChoices &a, const BinChoices &b) {
    for (int g = 0; g < GEMM_COUNT; g++) {
        const GemmChoice &x = a.gemm[g], &y = b.gemm[g];
        if (x.tile != y.tile || x.sk != y.sk || x.sk_steps != y.sk_steps || x.tf32 != y.tf32) return false;
    }
    return a.attention == b.attention && a.ln == b.ln;
}

// ---- Sessions, continued -------------------------------------------------------------

/* One captured run: its graph, its fetch and pack kernels' nodes, and the
 * arguments last set in it. */
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t pack_node = nullptr, fetch_node = nullptr;
    int bin = 0; /* the first bin it serves, whose shape and plan it was captured from */
    RunArgs in_graph{};
    int32_t fetch_in_graph = 0;
    bool set = false;
};

struct Session {
    Model *model = nullptr;
    Context *ctx = nullptr;
    uint32_t max_batch = 0, max_seq = 0;
    /* F16 GEMMs and attention, else F32 throughout. */
    bool half = false;
    /* The kernels, and by bin the shape each bin's graph is captured from
     * and its launches' plan. */
    Choices choices;
    Shape shape[BIN_COUNT];
    Plan plan[BIN_COUNT];
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
    /* The GEMMs' stream-K workspace and flags, sized for every bin's plan,
     * then ADD_LN's counters of finished tiles. */
    float *ws = nullptr;
    int *flags = nullptr, *rows_done = nullptr;
    int flag_ints = 0;
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
    /* The run as graphs, one per distinct choice of kernels, and each
     * bin's. */
    Graph graphs[BIN_COUNT];
    int graph_count = 0;
    int graph_of[BIN_COUNT] = {0, 0, 0, 0, 0};
    PackArgs pack{};
    FetchArgs fetch{};
    void *pack_argv[1] = {nullptr}, *fetch_argv[1] = {nullptr};
    /* What the last write left. */
    bool written = false;
    RunArgs run{};
    /* The packed tokens the last write holds, counted on the host where
     * cuBLAS or a second graph needs them, else 0. */
    uint32_t tokens = 0;
    uint64_t h2d = 0;
};

void release_session(Session *s) {
    DeviceScope scope(s->ctx->ordinal);
    if (s->sent) {
        cudaEventSynchronize(s->sent);
        cudaEventDestroy(s->sent);
    }
    for (Graph &g : s->graphs) {
        if (g.exec) cudaGraphExecDestroy(g.exec);
        if (g.graph) cudaGraphDestroy(g.graph);
    }
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

/* GemmArgs.min_steps for a GEMM's choice. */
int min_steps(const GemmChoice &c) { return c.sk == SK_TILES ? SK_WHOLE_TILES : c.sk_steps; }

/* The encoder with one bin's kernels, queued on the context's stream:
 * captured into the bin's graph, or run as it is queued when cuBLAS
 * computes a GEMM. */
int32_t encode(Session &s, int bin, turbo_error *err) {
    const Model &mo = *s.model;
    const turbo_backend_model &d = mo.desc;
    const std::vector<const float *> &w = mo.f32;
    const Plan &plan = s.plan[bin];
    const Shape &sh = s.shape[bin];
    const int h = (int)d.hidden, inter = (int)d.intermediate, heads = (int)d.heads, hd = h / heads;
    const int tcap = sh.tcap, tokens = (int)s.tokens;
    const float eps = (float)d.layer_norm_eps;
    const bool half = s.half;
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
    // One GEMM of the layer with its own kernel.
    auto gemm_as = [&](Gemm which, Epilogue e, GemmArgs &g) {
        g.min_steps = min_steps(sh.gemm[which]);
        return gemm(st, e, half, gemm_mma(sh, which), sh.gemm[which].tile, g, plan.gemm_grid[which]);
    };

    s.pack.queries = plan.attn_queries;
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
    g.rows_done = s.rows_done;
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
            TRY_CUDA(gemm_as(GEMM_QKV, EPI_QKV, g), "the query, key and value projections");
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
            TRY_CUDA(gemm_as(GEMM_OUT, EPI_ADD_LN, g), "the attention output projection and LayerNorm");
        } else {
            g.bias = nullptr;
            g.out = s.part;
            if (s.cublas & CUBLAS_OUT) {
                TRY_CUBLAS(linear(blas, half, s.att, tokens, h, g.w, h, s.part), "the attention output projection");
            } else {
                TRY_CUDA(gemm_as(GEMM_OUT, EPI_PLAIN, g), "the attention output projection");
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
            TRY_CUDA(gelu_epilogue(st, s.raw, g, half, sh.gelu_erf, plan), "GELU");
        } else {
            TRY_CUDA(gemm_as(GEMM_FFN1, gemm_epilogue(GEMM_FFN1, plan.fused_ln, sh.gelu_erf), g),
                     "the feed-forward input");
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
            TRY_CUDA(gemm_as(GEMM_FFN2, EPI_ADD_LN, g), "the feed-forward output and LayerNorm");
        } else {
            g.bias = nullptr;
            g.out = s.part;
            if (s.cublas & CUBLAS_FFN2) {
                TRY_CUBLAS(linear(blas, half, s.ffn, tokens, inter, g.w, h, s.part), "the feed-forward output");
            } else {
                TRY_CUDA(gemm_as(GEMM_FFN2, EPI_PLAIN, g), "the feed-forward output");
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

/* One bin's run captured into a graph, instantiated and uploaded, and its
 * fetch and pack kernels' nodes found. */
int32_t capture(Session &s, int bin, Graph *out, turbo_error *err) {
    cudaStream_t st = s.ctx->stream;
    out->bin = bin;
    TRY_CUDA(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capturing the run");
    const int32_t rc = encode(s, bin, err);
    cudaGraph_t graph = nullptr;
    const cudaError_t e = cudaStreamEndCapture(st, &graph);
    if (rc != TURBO_OK) {
        if (graph) cudaGraphDestroy(graph);
        (void)cudaGetLastError();
        return rc;
    }
    TRY_CUDA(e, "capturing the run");
    out->graph = graph;
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
        if (kp.func == pack_rows_function()) out->pack_node = node;
        if (kp.func == fetch_rows_function()) out->fetch_node = node;
    }
    if (!out->pack_node || !out->fetch_node)
        return refuse(err, TURBO_E_INTERNAL, "the run's graph has no packing or fetch kernel");
    TRY_CUDA(cudaGraphInstantiateWithFlags(&out->exec, graph, 0), "cudaGraphInstantiate");
    TRY_CUDA(cudaGraphUpload(out->exec, st), "cudaGraphUpload");
    TRY_CUDA(cudaStreamSynchronize(st), "cudaGraphUpload");
    return TURBO_OK;
}

/* A graph per distinct choice of kernels, and each bin pointed at its
 * own. */
int32_t capture_bins(Session &s, turbo_error *err) {
    for (int b = 0; b < s.choices.bins; b++) {
        int same = 0;
        while (same < s.graph_count && !same_kernels(s.choices.bin[s.graphs[same].bin], s.choices.bin[b])) same++;
        if (same == s.graph_count) {
            TRY(capture(s, b, &s.graphs[same], err));
            s.graph_count++;
        }
        s.graph_of[b] = same;
    }
    return TURBO_OK;
}

// ---- The tuner ---------------------------------------------------------------------
//
// A tuned session times its GEMMs' kernel variants on its own buffers when
// it is made, after its memory is allocated and cleared and before its
// graphs are captured, under the context's lock on the context's stream,
// and takes per bin and GEMM the fastest whose numeric class the session
// allows (docs/autotune.md).

/* The variant name the tests make fail to launch, in place of a device
 * that cannot take it; empty for none. */
std::mutex fail_lock;
std::string fail_name;

bool made_to_fail(const GemmChoice &g) {
    std::lock_guard<std::mutex> l(fail_lock);
    return !fail_name.empty() && fail_name == variant_name(g);
}

/* Whether a GEMM's variant can launch here: its kernel's shared memory
 * set, a grid of at least one block, its workspace in *ws floats. */
cudaError_t launchable(const Shape &base, Epilogue ep, const GemmChoice &g, int *grid, size_t *ws) {
    if (made_to_fail(g)) return cudaErrorInvalidValue;
    const bool mma = base.tensor_cores && (base.half || g.tf32);
    cudaError_t e = gemm_prepare(ep, base.half, mma, g.tile);
    if (e == cudaSuccess) e = gemm_grid(ep, base.half, mma, g.tile, base.sms, grid, ws);
    if (e == cudaSuccess && *grid <= 0) e = cudaErrorInvalidConfiguration;
    if (e != cudaSuccess) (void)cudaGetLastError();
    return e;
}

/* The packed tokens a bin is tuned at: its upper edge, or the session's
 * tokens when fewer; past 16384, at most 32768. */
int tuning_tokens(int bin, int tcap) {
    const int edge = bin == BIN_COUNT - 1 ? 32768 : BIN_EDGE[bin];
    return edge < tcap ? edge : tcap;
}

/* The bins in the order they are tuned, those nearest a mixed batch's
 * size first, so a budget spent early skips the least used. */
constexpr int TUNE_ORDER[BIN_COUNT] = {2, 1, 3, 0, 4};

/* A GEMM's variant the tuner may time: its choice and its launch. */
struct Candidate {
    GemmChoice c;
    int grid = 0;
};

/* What the tuner does for one session: the candidates of each bin's
 * GEMMs, the incumbent first, and what it logs and reports. */
struct Tuning {
    std::vector<Candidate> cand[BIN_COUNT][GEMM_COUNT];
    std::vector<std::string> unlaunchable; /* logged once each */
};

/* Every variant a GEMM of each bin may be timed as, with the workspace
 * and flags the largest of them needs added to *sk_floats and *sk_flags:
 * the incumbent (the bin's choice, which its plan launches), then the
 * candidates whose class runs allows, each once by the kernel it runs.
 * A GEMM whose tile is forced is not timed. */
void find_candidates(Context &c, const Shape &base, const Choices &ch, const Plan *plan, uint32_t runs, Tuning *t,
                     size_t *sk_floats, int *sk_flags) {
    Variant vs[32];
    const int nv = gemm_variants(base, vs, 32);
    for (int b = 0; b < ch.bins; b++)
        for (int g = 0; g < GEMM_COUNT; g++) {
            const BinChoices &bc = ch.bin[b];
            std::vector<Candidate> &list = t->cand[b][g];
            if (bc.gemm_forced[g] & KNOB_TILE) continue;
            const Epilogue ep = gemm_epilogue((Gemm)g, plan[b].fused_ln, base.gelu_erf);
            list.push_back(Candidate{bc.gemm[g], plan[b].gemm_grid[g]});
            for (int i = 0; i < nv; i++) {
                if (!vs[i].candidate) continue;
                if ((bc.gemm_forced[g] & KNOB_TF32) && vs[i].tf32 != bc.gemm[g].tf32) continue;
                GemmChoice gc = bc.gemm[g];
                gc.tile = vs[i].tile;
                gc.tf32 = vs[i].tf32;
                gc = canonical_gemm((Gemm)g, base, gc);
                if (!(gemm_numeric(base, gc) & runs)) continue;
                bool seen = false;
                for (const Candidate &k : list) seen = seen || (k.c.tile == gc.tile && k.c.tf32 == gc.tf32);
                if (seen) continue;
                int grid = 0;
                size_t ws = 0;
                const cudaError_t e = launchable(base, ep, gc, &grid, &ws);
                if (e != cudaSuccess) {
                    const std::string name = variant_name(gc);
                    bool said = false;
                    for (const std::string &u : t->unlaunchable) said = said || u == name;
                    if (!said) {
                        t->unlaunchable.push_back(name);
                        c.say(LOG_INFO, "cuda device %d: the GEMM kernel %s cannot launch here (%s), so it is not timed",
                              c.ordinal, name.c_str(), cudaGetErrorName(e));
                    }
                    continue;
                }
                *sk_floats = ws > *sk_floats ? ws : *sk_floats;
                *sk_flags = grid > *sk_flags ? grid : *sk_flags;
                list.push_back(Candidate{gc, grid});
            }
        }
}

/* The tuner's rows for m packed tokens into the staging, as embed_write
 * leaves a write: ids 1 and types 0, the row lengths of a mixed batch
 * (fractions of max_seq in a fixed order, cut to what is left), then rows
 * topped up to max_seq, until exactly m tokens are live. */
void stage_rows(Session &s, int m) {
    static constexpr int EIGHTHS[8] = {8, 3, 5, 1, 6, 2, 7, 4};
    const int seq = (int)s.max_seq, cap = (int)s.max_batch;
    std::vector<int> len((size_t)cap, 0);
    int left = m, rows = 0;
    for (int r = 0; r < cap && left > 0; r++) {
        int n = seq * EIGHTHS[r % 8] / 8;
        n = n < 1 ? 1 : n > left ? left : n;
        len[(size_t)r] = n;
        left -= n;
        rows = r + 1;
    }
    for (int r = 0; r < cap && left > 0; r++) {
        const int more = seq - len[(size_t)r] < left ? seq - len[(size_t)r] : left;
        len[(size_t)r] += more;
        left -= more;
        rows = r + 1 > rows ? r + 1 : rows;
    }
    const size_t plane = (size_t)rows * seq;
    int32_t *ids = s.staging, *mask = s.staging + plane;
    for (int r = 0; r < rows; r++)
        for (int i = 0; i < seq; i++) {
            const bool live = i < len[(size_t)r];
            ids[(size_t)r * seq + i] = live ? 1 : 0;
            mask[(size_t)r * seq + i] = live ? 1 : 0;
        }
    s.fetch.n = (int32_t)(2 * plane);
    s.run = RunArgs{rows, seq, TURBO_POOLING_MEAN, 0, (int32_t)s.model->desc.hidden, 0};
    s.pack.run = s.run;
    s.tokens = (uint32_t)m;
}

/* The first layer's GEMM of the bin as encode launches it, with the
 * epilogue ep. */
GemmArgs layer_gemm(const Session &s, int bin, Gemm which, Epilogue ep) {
    const Model &mo = *s.model;
    const turbo_backend_model &d = mo.desc;
    auto at = [](int r) { return (size_t)TURBO_BERT_EMBEDDING_TENSORS + r; };
    auto weight = [&](int r) -> const void * {
        return s.half ? static_cast<const void *>(mo.f16[at(r)]) : static_cast<const void *>(mo.f32[at(r)]);
    };
    auto layer = [&](int r) { return mo.f32[at(r)]; };
    const int h = (int)d.hidden;
    GemmArgs g{};
    g.info = s.pk.info;
    g.heads = (int)d.heads;
    g.head_dim = h / (int)d.heads;
    g.hidden = h;
    g.tcap = s.shape[bin].tcap;
    g.ws = s.ws;
    g.flags = s.flags;
    g.fault = s.fault_dev;
    g.rows_done = s.rows_done;
    g.eps = (float)d.layer_norm_eps;
    g.x16 = s.x16;
    const void *xin = s.half ? static_cast<const void *>(s.x16) : s.x;
    const bool ln = ep == EPI_ADD_LN;
    switch (which) {
    case GEMM_QKV:
        g.a = xin;
        g.w = weight(TURBO_BERT_Q_WEIGHT);
        g.bias = layer(TURBO_BERT_Q_BIAS);
        g.out = s.qkv;
        g.n = 3 * h;
        g.k = h;
        break;
    case GEMM_OUT:
        g.a = s.att;
        g.w = weight(TURBO_BERT_ATTN_OUT_WEIGHT);
        g.bias = ln ? layer(TURBO_BERT_ATTN_OUT_BIAS) : nullptr;
        g.out = ln ? static_cast<void *>(s.x) : s.part;
        g.ln_w = layer(TURBO_BERT_ATTN_LN_WEIGHT);
        g.ln_b = layer(TURBO_BERT_ATTN_LN_BIAS);
        g.n = h;
        g.k = h;
        break;
    case GEMM_FFN1:
        g.a = xin;
        g.w = weight(TURBO_BERT_FFN_IN_WEIGHT);
        g.bias = layer(TURBO_BERT_FFN_IN_BIAS);
        g.out = s.ffn;
        g.n = (int)d.intermediate;
        g.k = h;
        break;
    default:
        g.a = s.ffn;
        g.w = weight(TURBO_BERT_FFN_OUT_WEIGHT);
        g.bias = ln ? layer(TURBO_BERT_FFN_OUT_BIAS) : nullptr;
        g.out = ln ? static_cast<void *>(s.x) : s.part;
        g.ln_w = layer(TURBO_BERT_FFN_LN_WEIGHT);
        g.ln_b = layer(TURBO_BERT_FFN_LN_BIAS);
        g.n = h;
        g.k = (int)d.intermediate;
        break;
    }
    return g;
}

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

/* A kernel's times in milliseconds: the least of them ranks it, the
 * median shows a lucky least; spread is the first five's (median - min) /
 * min. */
struct Timed {
    float min = 0, median = 0, spread = 0;
    int n = 0;
};

/* How far apart the least and the most of n times are, over the least. */
float spread_of(const float *ms, int n) {
    float lo = ms[0], hi = ms[0];
    for (int i = 1; i < n; i++) {
        lo = ms[i] < lo ? ms[i] : lo;
        hi = ms[i] > hi ? ms[i] : hi;
    }
    return lo > 0 ? (hi - lo) / lo : 0;
}

/* How far the median of ms[0..n) is above the least, over the least: one
 * slow launch in five moves it less than the whole range would. */
float lift_of(const float *ms, int n) {
    float v[16];
    for (int i = 0; i < n; i++) v[i] = ms[i];
    std::sort(v, v + n);
    return v[0] > 0 ? (v[n / 2] - v[0]) / v[0] : 0;
}

constexpr int TIMED_FIRST = 5, TIMED_MOST = 15;
constexpr float TIMED_NOISY = 0.10f;

/* launch timed on the stream with event pairs: once untimed, then five
 * times, and five more while the times are more than 10% apart, up to 15
 * while the deadline allows. */
template <typename F>
cudaError_t time_kernel(cudaStream_t st, cudaEvent_t *ev, F launch, Clock::time_point deadline, Timed *out) {
    cudaError_t e = launch();
    if (e != cudaSuccess) return e;
    float ms[TIMED_MOST];
    int n = 0;
    do {
        for (int i = 0; i < TIMED_FIRST; i++) {
            if ((e = cudaEventRecord(ev[2 * i], st)) != cudaSuccess) return e;
            if ((e = launch()) != cudaSuccess) return e;
            if ((e = cudaEventRecord(ev[2 * i + 1], st)) != cudaSuccess) return e;
        }
        if ((e = cudaStreamSynchronize(st)) != cudaSuccess) return e;
        for (int i = 0; i < TIMED_FIRST; i++)
            if ((e = cudaEventElapsedTime(&ms[n++], ev[2 * i], ev[2 * i + 1])) != cudaSuccess) return e;
        if (n == TIMED_FIRST) out->spread = lift_of(ms, n);
    } while (n < TIMED_MOST && spread_of(ms, n) > TIMED_NOISY && Clock::now() < deadline);
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && ms[j] < ms[j - 1]; j--) std::swap(ms[j], ms[j - 1]);
    out->min = ms[0];
    out->median = ms[n / 2];
    out->n = n;
    return cudaSuccess;
}

/* A candidate replaces the incumbent only when this much faster, so a
 * measurement again on the same device keeps the choice. */
constexpr float MARGIN = 0.95f;
/* The median of the incumbent's first five times further above their
 * least than this, in each of BUSY_ROUNDS timings, says the device is
 * shared or throttling. */
constexpr float BUSY = 0.25f;
/* How many times the first incumbent is timed before its times apart
 * say the device is busy. */
constexpr int BUSY_ROUNDS = 3;

/* What the tuner found: whether it measured, or stopped for a busy
 * device, and what it says in the log and the record. */
struct Tuned {
    bool measured = false, busy = false;
    int skipped = 0;
    std::string timings, summary;
};

/* The tuner over s.choices, from the incumbents: each bin in TUNE_ORDER
 * gets the tuner's rows for its tokens, each of its GEMMs times its
 * incumbent then its candidates, and the fastest by 5% is the bin's
 * choice. The first bin runs the whole encoder first, untimed as a
 * choice, which loads the modules, brings the clocks up and leaves
 * activations for the GEMMs to read. */
int32_t tune(Session &s, const Tuning &t, uint32_t budget_ms, Tuned *out, turbo_error *err) {
    Context &c = *s.ctx;
    cudaStream_t st = c.stream;
    const Clock::time_point start = Clock::now();
    const Clock::time_point deadline = start + std::chrono::milliseconds(budget_ms);
    cudaEvent_t ev[2 * TIMED_FIRST] = {};
    struct Events {
        cudaEvent_t *ev;
        ~Events() {
            for (int i = 0; i < 2 * TIMED_FIRST; i++)
                if (ev[i]) cudaEventDestroy(ev[i]);
        }
    } events{ev};
    for (cudaEvent_t &e : ev) TRY_CUDA(cudaEventCreate(&e), "the tuner's events");
    const Choices before = s.choices;
    bool first = true, busy_checked = false;
    char line[160];
    for (int b : TUNE_ORDER) {
        if (b >= s.choices.bins) continue;
        const Plan &plan = s.plan[b];
        const int m = tuning_tokens(b, s.shape[b].tcap);
        stage_rows(s, m);
        if (first) {
            TRY(encode(s, b, err));
        } else {
            s.pack.queries = plan.attn_queries;
            TRY_CUDA(fetch_rows(st, s.fetch, plan), "the tuner's rows");
            TRY_CUDA(pack_rows(st, s.pack, plan), "the tuner's rows");
        }
        TRY_CUDA(cudaStreamSynchronize(st), "the tuner's rows");
        double was = 0, now = 0;
        for (int g = 0; g < GEMM_COUNT; g++) {
            const std::vector<Candidate> &list = t.cand[b][g];
            if (list.empty()) continue;
            const Epilogue ep = gemm_epilogue((Gemm)g, plan.fused_ln, s.shape[b].gelu_erf);
            GemmArgs ga = layer_gemm(s, b, (Gemm)g, ep);
            int best = -1;
            float incumbent = 0, fastest = 0;
            for (size_t i = 0; i < list.size(); i++) {
                const Candidate &k = list[i];
                const std::string name = variant_name(k.c);
                if (Clock::now() >= deadline) {
                    // Past the budget: what is left is not timed, and
                    // without its incumbent's time nothing replaces it.
                    for (size_t j = i; j < list.size(); j++) {
                        out->skipped++;
                        c.say(LOG_DEBUG, "cuda device %d: %s/%s/%s not timed: the budget of %u ms is spent",
                              c.ordinal, BIN_NAME[b], GEMM_NAMES[g], variant_name(list[j].c).c_str(), budget_ms);
                    }
                    break;
                }
                ga.min_steps = min_steps(k.c);
                const bool mma = s.shape[b].tensor_cores && (s.half || k.c.tf32);
                auto launch = [&]() { return gemm(st, ep, s.half, mma, k.c.tile, ga, k.grid); };
                Timed tm;
                const cudaError_t e = time_kernel(st, ev, launch, deadline, &tm);
                if (e != cudaSuccess) {
                    // A launch that fails leaves the stream as it was; a
                    // fault in a kernel fails the session.
                    (void)cudaGetLastError();
                    TRY_CUDA(cudaStreamSynchronize(st), "the tuner's GEMM");
                    c.say(LOG_INFO, "cuda device %d: %s/%s/%s failed to launch (%s), so it is not chosen", c.ordinal,
                          BIN_NAME[b], GEMM_NAMES[g], name.c_str(), cudaGetErrorName(e));
                    if (i == 0) break;
                    continue;
                }
                // The first incumbent's times far apart may be the clocks
                // still coming up or a moment's contention: timed again,
                // up to BUSY_ROUNDS in all, before the device is called
                // busy. A shared or throttling device stays apart.
                for (int r = 1; i == 0 && !busy_checked && tm.spread > BUSY && r < BUSY_ROUNDS &&
                                Clock::now() < deadline;
                     r++) {
                    c.say(LOG_DEBUG, "cuda device %d: %s/%s/%s's times were %.0f%% apart, so it is timed again",
                          c.ordinal, BIN_NAME[b], GEMM_NAMES[g], name.c_str(), 100.0 * tm.spread);
                    TRY_CUDA(time_kernel(st, ev, launch, deadline, &tm), "the tuner's GEMM");
                }
                if (*reinterpret_cast<volatile int *>(s.fault)) {
                    // A stream-K wait gave up: the time is not the
                    // kernel's, and its flags are cleared for the next.
                    *reinterpret_cast<volatile int *>(s.fault) = 0;
                    TRY_CUDA(cudaMemsetAsync(s.flags, 0, (size_t)s.flag_ints * 4, st), "the tuner's GEMM");
                    TRY_CUDA(cudaStreamSynchronize(st), "the tuner's GEMM");
                    c.say(LOG_INFO, "cuda device %d: %s/%s/%s waited too long for a partial product, so it is not chosen",
                          c.ordinal, BIN_NAME[b], GEMM_NAMES[g], name.c_str());
                    if (i == 0) break;
                    continue;
                }
                c.say(LOG_DEBUG, "cuda device %d: %s/%s/%s at %d tokens: least %.4f ms, median %.4f ms of %d",
                      c.ordinal, BIN_NAME[b], GEMM_NAMES[g], name.c_str(), m, tm.min, tm.median, tm.n);
                snprintf(line, sizeof line, "%s/%s/%s=%.4f\n", BIN_NAME[b], GEMM_NAMES[g], name.c_str(), tm.min);
                out->timings += line;
                out->measured = true;
                if (i == 0) {
                    incumbent = tm.min;
                    if (!busy_checked && tm.spread > BUSY) {
                        out->busy = true;
                        c.say(LOG_WARNING,
                              "cuda device %d: not tuned: the same GEMM's times were %.0f%% apart, so the device is "
                              "shared or throttling; the next session measures again",
                              c.ordinal, 100.0 * tm.spread);
                        s.choices = before;
                        out->measured = false;
                        return TURBO_OK;
                    }
                    busy_checked = true;
                    continue;
                }
                if (best < 0 || tm.min < fastest) {
                    best = (int)i;
                    fastest = tm.min;
                }
            }
            if (incumbent <= 0) continue;
            float chosen = incumbent;
            if (best > 0 && fastest <= MARGIN * incumbent) {
                GemmChoice &gc = s.choices.bin[b].gemm[g];
                gc.tile = list[(size_t)best].c.tile;
                gc.tf32 = list[(size_t)best].c.tf32;
                chosen = fastest;
            }
            was += incumbent;
            now += chosen;
        }
        first = false;
        if (was > 0) {
            snprintf(line, sizeof line, "%s%s: the GEMMs of a layer %.3f ms, the incumbents %.3f ms",
                     out->summary.empty() ? "" : "; ", BIN_NAME[b], now, was);
            out->summary += line;
        }
    }
    return TURBO_OK;
}

int32_t session_create_tuned(void *model, uint32_t task, uint32_t max_batch, uint32_t max_seq, uint32_t precision,
                             turbo_backend_tuning *tuning, uint32_t *compute_dtype, void **out, turbo_error *err) {
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
        Shape base;
        base.batch_cap = (int)max_batch;
        base.seq_cap = (int)max_seq;
        base.tcap = (int)tokens;
        base.hidden = (int)d.hidden;
        base.heads = (int)d.heads;
        base.inter = (int)d.intermediate;
        base.half = half;
        base.gelu_erf = half && gelu_erf_named();
        int major = 0, sms = 0, optin = 0;
        TRY_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, c->ordinal), "the device's sm");
        TRY_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, c->ordinal), "the device's SMs");
        TRY_CUDA(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, c->ordinal),
                 "the device's shared memory");
        // The tensor cores: F16 at FASTEST; TF32 for F32 products when
        // TURBO_CUDA_TF32=1 asks, but at EXACT, F32 FMAs throughout
        // (forced_from_environment).
        base.tensor_cores = major >= 8;
        base.sms = sms;
        base.smem_optin = (size_t)optin;
        const uint32_t mode = tuning ? tuning->mode : (uint32_t)TURBO_AUTOTUNE_OFF;
        const bool tuned = mode == TURBO_AUTOTUNE_ON || mode == TURBO_AUTOTUNE_RETUNE;
        Choices ch = defaults(base);
        bool tf32 = false, f16_accumulate = false;
        forced_from_environment(base, precision, tuned, &ch, &tf32, &f16_accumulate);
        char why[TURBO_ERROR_MESSAGE_LEN];
        uint32_t absent = 0;
        if (!choices_named(&ch, &absent, why, sizeof why)) return refuse(err, TURBO_E_INVALID_ARGUMENT, "%s", why);
        for (int b = 0; b < BIN_COUNT; b++)
            if (absent & (1u << b))
                c->say(LOG_DEBUG,
                       "cuda device %d: TURBO_CUDA_CHOICES names %s, a token bin a session of %u x %u does not have",
                       c->ordinal, BIN_NAME[b], max_batch, max_seq);
        whole_rows(base, &ch);
        canonicalize(base, &ch);
        // The classes the precision allows, as the environment's
        // experiments widen them; the F32 kernels are FASTEST's too, for a
        // model past F16's range.
        const uint32_t allowed = tuning ? tuning->numerics_allowed : default_numerics(precision);
        const uint32_t tier = allowed | (half ? 0u : TURBO_NUMERIC_F32_FMA);
        const uint32_t runs = tier | (tf32 ? TURBO_NUMERIC_TF32 : 0u) |
                              (f16_accumulate ? TURBO_NUMERIC_F16_CHUNKACC : 0u);
        char named[160];
        const char *numeric = nullptr;
        if (outside(base, ch, runs, named, sizeof named, &numeric))
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 3,
                                "precision %s: %s computes in %s, which the precision does not allow",
                                precision_name(precision), named, numeric);
        for (int b = 0; b < ch.bins; b++)
            for (int g = 0; g < GEMM_COUNT; g++)
                if (made_to_fail(ch.bin[b].gemm[g]))
                    return refuse(err, TURBO_E_UNSUPPORTED, "%s:%s=%s: the GEMM kernel cannot launch on this device",
                                  BIN_NAME[b], GEMM_NAMES[g], variant_name(ch.bin[b].gemm[g]).c_str());
        // The choices an earlier session with the same key measured, the
        // incumbents here: taken only by a session that forces nothing and
        // widens nothing, as the core caches only such sessions.
        // cuBLAS's GEMMs run without a graph and have no variants to time,
        // and a cached line would name kernels they do not run: such a
        // session takes neither.
        const unsigned cublas = cublas_gemms();
        if (tuned && cublas)
            c->say(LOG_INFO,
                   "cuda device %d: not tuned: TURBO_CUDA_CUBLAS hands GEMMs to cuBLAS, whose kernels are its own",
                   c->ordinal);
        // The tuner times only GEMM tiles: with every one forced, it has
        // nothing to time, and the session reports FORCED.
        const bool untimed = tuned && !cublas && tiles_forced(ch);
        if (untimed) c->say(LOG_INFO, "cuda device %d: not tuned: every GEMM tile is forced", c->ordinal);
        bool cached = false;
        const bool plain = !forced_knobs(ch) && !tf32 && !f16_accumulate && !cublas;
        if (tuned && tuning->cached && *tuning->cached && plain) {
            Choices cc = ch;
            uint32_t missing = 0;
            char no[TURBO_ERROR_MESSAGE_LEN] = "a kernel of it cannot launch here";
            bool ok = parse_choices(tuning->cached, &cc, &missing, no, sizeof no);
            if (ok) {
                for (BinChoices &bc : cc.bin) {
                    bc.forced = 0;
                    for (uint32_t &f : bc.gemm_forced) f = 0;
                }
                cc.pool_forced = 0;
                whole_rows(base, &cc);
                canonicalize(base, &cc);
                ok = !outside(base, cc, tier, named, sizeof named, &numeric);
                if (!ok) snprintf(no, sizeof no, "%s computes in %s", named, numeric);
                for (int b = 0; ok && b < cc.bins; b++)
                    for (int g = 0; g < GEMM_COUNT; g++) ok = ok && !made_to_fail(cc.bin[b].gemm[g]);
            }
            if (ok) {
                ch = cc;
                cached = true;
                c->say(LOG_DEBUG, "cuda device %d: kernels from the cache: %s", c->ordinal, tuning->cached);
            } else {
                c->say(LOG_DEBUG, "cuda device %d: the cached kernels are not taken (%s): %s", c->ordinal, no,
                       tuning->cached);
            }
        }
        // ON with a cached choice takes it unmeasured; RETUNE measures
        // against it.
        const bool measure = tuned && !cublas && !untimed && !(cached && mode == TURBO_AUTOTUNE_ON);

        Session *s = make<Session>();
        s->model = m;
        s->ctx = c;
        s->max_batch = max_batch;
        s->max_seq = max_seq;
        s->half = half;
        s->choices = ch;
        s->cublas = cublas;
        // Every bin's launches, and the workspace the largest of them needs.
        size_t sk_floats = 0;
        int sk_flags = 0, ln_counts = 0;
        bool crowded = false;
        auto plan_bins = [&]() -> cudaError_t {
            crowded = false;
            for (int b = 0; b < s->choices.bins; b++) {
                s->shape[b] = shape_for(base, s->choices, b);
                int same = 0;
                while (same < b && !same_kernels(s->choices.bin[same], s->choices.bin[b])) same++;
                if (same < b) {
                    s->plan[b] = s->plan[same];
                } else {
                    s->plan[b] = Plan{};
                    const cudaError_t e = make_plan(s->shape[b], &s->plan[b]);
                    if (e != cudaSuccess) return e;
                }
                const Plan &p = s->plan[b];
                sk_floats = p.sk_floats > sk_floats ? p.sk_floats : sk_floats;
                sk_flags = p.sk_flags > sk_flags ? p.sk_flags : sk_flags;
                ln_counts = p.ln_counts > ln_counts ? p.ln_counts : ln_counts;
                crowded = crowded || p.gemm_crowded;
            }
            return cudaSuccess;
        };
        if (const cudaError_t e = plan_bins(); e != cudaSuccess) {
            release_session(s);
            return cuda_failed(err, e, "planning the session's launches");
        }
        // The variants the tuner may time, and the workspace the largest
        // needs: the session's memory holds every one of them.
        Tuning tn;
        if (measure) find_candidates(*c, base, s->choices, s->plan, runs, &tn, &sk_floats, &sk_flags);
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
        const size_t ws = round_up(sk_floats * 4, DEVICE_ALIGN);
        s->flag_ints = sk_flags + ln_counts;
        const size_t flags = round_up((size_t)s->flag_ints * 4, DEVICE_ALIGN);
        const size_t widest = 3 * h > inter ? 3 * h : inter;
        const size_t raw = s->cublas & (CUBLAS_QKV | CUBLAS_FFN1) ? round_up(tokens * widest * 4, DEVICE_ALIGN) : 0;
        const size_t output = round_up((size_t)max_batch * h * 4, DEVICE_ALIGN);
        Tuned found;
        uint32_t tune_ms = 0;
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
            s->rows_done = s->flags + sk_flags;
            if (raw) s->raw = reinterpret_cast<float *>(take(raw));
            s->output.ctx = c;
            s->output.ptr = take(output);
            s->output.placement = TURBO_PLACE_DEVICE;
            s->output.bytes = (uint64_t)max_batch * d.hidden * 4;
            s->output.owned = false;
            s->pack.rows = s->rows;
            s->pack.heads = (int32_t)d.heads;
            s->pack.queries = s->plan[0].attn_queries;
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
            if (measure) {
                const Clock::time_point tune_start = Clock::now();
                TRY(tune(*s, tn, tuning->budget_ms, &found, err));
                // The chosen kernels' launches, in the memory sized for
                // every candidate's, cleared again for the graphs.
                TRY_CUDA(plan_bins(), "planning the session's launches");
                *s->fault = 0;
                TRY_CUDA(cudaMemsetAsync(s->scratch, 0, total, c->stream), "clearing the session's memory");
                TRY_CUDA(cudaStreamSynchronize(c->stream), "clearing the session's memory");
                s->fetch.n = 0;
                s->tokens = 0;
                s->run = RunArgs{};
                tune_ms = (uint32_t)std::ceil(ms_since(tune_start));
            }
            if (!s->cublas) TRY(capture_bins(*s, err));
            return TURBO_OK;
        }();
        if (rc != TURBO_OK) {
            release_session(s);
            return rc;
        }
        // The pooling as the plan takes it: a thread per column for hidden
        // states too wide for the groups.
        s->choices.pool = s->plan[0].column_pool ? POOL_COLUMNS : POOL_GROUPS;
        if (crowded)
            c->say(LOG_WARNING,
                   "cuda device %d: a GEMM's kernel fits fewer blocks to an SM than it was built for, so it runs "
                   "slower than it should",
                   c->ordinal);
        // What the kernels compute in: the precision's classes, widened
        // only by a class a chosen kernel runs in beyond them.
        const uint32_t ran = numerics_of(base, s->choices);
        const uint32_t used = allowed | (ran & ~tier);
        char line[TURBO_CHOICES_LEN];
        format_choices(s->choices, line, sizeof line);
        if (found.measured) {
            c->say(LOG_INFO, "cuda device %d: kernels chosen in %u ms for %d token bins: %s (%s%s%d not timed)",
                   c->ordinal, tune_ms, s->choices.bins, line, found.summary.c_str(),
                   found.summary.empty() ? "" : "; ", found.skipped);
        } else if (cached && !measure) {
            c->say(LOG_DEBUG, "cuda device %d: kernels from the cache for %d token bins: %s", c->ordinal,
                   s->choices.bins, line);
        }
        c->say(LOG_DEBUG,
               "cuda device %d: an embed session of %u rows of %u tokens, its GEMMs computing in %s%s; attention %zu "
               "bytes of shared memory for %d keys at a time",
               c->ordinal, max_batch, max_seq, numerics_named(ran).c_str(),
               s->cublas ? ", some GEMMs on cuBLAS, without a graph" : ", as graphs", s->plan[0].attn_smem,
               s->plan[0].attn_chunk);
        if (tuning) {
            if (found.measured)
                tuning->tuned = TURBO_TUNED_MEASURED;
            else if (cached)
                tuning->tuned = TURBO_TUNED_CACHE;
            else
                tuning->tuned = untimed || all_forced(s->choices) ? TURBO_TUNED_FORCED : TURBO_TUNED_DEFAULT;
            tuning->tune_ms = found.measured ? tune_ms : 0;
            copy_str(tuning->choices, sizeof tuning->choices, line);
            if (tuning->timings && tuning->timings_len) {
                // Whole lines, as many as fit.
                std::string tl = found.measured ? found.timings : std::string();
                if (tl.size() >= tuning->timings_len) {
                    tl.resize(tuning->timings_len - 1);
                    const size_t end = tl.rfind('\n');
                    tl.resize(end == std::string::npos ? 0 : end + 1);
                }
                copy_str(tuning->timings, tuning->timings_len, tl.c_str());
            }
            tuning->numerics_used = used;
        }
        *compute_dtype = half ? TURBO_DTYPE_F16 : TURBO_DTYPE_F32;
        *out = s;
        return TURBO_OK;
    });
}

/* A session with the built-in choices and what the environment forces, as
 * session_create_tuned makes it with tuning OFF. */
int32_t session_create(void *model, uint32_t task, uint32_t max_batch, uint32_t max_seq, uint32_t precision,
                       uint32_t *compute_dtype, void **out, turbo_error *err) {
    return session_create_tuned(model, task, max_batch, max_seq, precision, nullptr, compute_dtype, out, err);
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
        // The packed size, which picks the run's token bin and which
        // cuBLAS's GEMMs take from the host: each row through its last
        // live token, as pack_rows finds it on the device from the same
        // entries, the host memory the core read. A session of one graph
        // needs neither, and leaves it 0.
        uint32_t packed = 0;
        if (s.cublas || s.graph_count > 1)
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

/* The written rows through the encoder with their bin's kernels: the
 * bin's graph, with this run's rows' shape and options set in its fetch
 * and pack kernels when they differ from the last run's in it, or the
 * launches one by one when cuBLAS computes a GEMM. */
int32_t launch(Session &s, turbo_error *err) {
    const int bin = bin_of(s.tokens, s.choices.bins);
    s.pack.run = s.run;
    if (s.cublas) return encode(s, bin, err);
    Graph &g = s.graphs[s.graph_of[bin]];
    const Plan &plan = s.plan[g.bin];
    if (!g.set || memcmp(&g.in_graph, &s.run, sizeof s.run) != 0 || g.fetch_in_graph != s.fetch.n) {
        // Marked unset first: a failure leaves the graph's arguments unknown.
        g.set = false;
        s.pack.queries = plan.attn_queries;
        cudaKernelNodeParams kp;
        pack_rows_node(&s.pack, s.pack_argv, plan, &kp);
        TRY_CUDA(cudaGraphExecKernelNodeSetParams(g.exec, g.pack_node, &kp), "setting the run's rows");
        fetch_rows_node(&s.fetch, s.fetch_argv, plan, &kp);
        TRY_CUDA(cudaGraphExecKernelNodeSetParams(g.exec, g.fetch_node, &kp), "setting the run's rows");
        g.in_graph = s.run;
        g.fetch_in_graph = s.fetch.n;
        g.set = true;
    }
    TRY_CUDA(cudaGraphLaunch(g.exec, s.ctx->stream), "cudaGraphLaunch");
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
                TRY_CUDA(cudaMemsetAsync(s.flags, 0, (size_t)s.flag_ints * 4, s.ctx->stream), "the run");
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
                } else if (epi == EPI_GELU || epi == EPI_GELU_ERF) {
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
    session_create_tuned,
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

/* The choices string of sessions made from now on, in place of
 * TURBO_CUDA_CHOICES; NULL to read the variable again. */
void turbo_cuda_use_choices(const char *choices) {
    std::lock_guard<std::mutex> g(choices_lock);
    choices_set = choices != nullptr;
    choices_override = choices ? choices : "";
}

/* The GEMM kernel variants a session at precision on device ordinal may
 * force, as lines of "<name> <TURBO_NUMERIC_*> <1 when the tuner times
 * it, else 0>", into out (len bytes with the NUL). A FASTEST session is
 * taken to have F16 operands. */
int32_t turbo_cuda_variants(uint32_t ordinal, uint32_t precision, char *out, size_t len) {
    try {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, (int)ordinal) != cudaSuccess) {
            (void)cudaGetLastError();
            return TURBO_E_RUNTIME;
        }
        Shape base;
        base.half = precision == TURBO_PRECISION_FASTEST;
        base.tensor_cores = major >= 8;
        Variant v[32];
        const int n = gemm_variants(base, v, 32);
        std::string s;
        for (int i = 0; i < n; i++) {
            s += v[i].name;
            s += ' ';
            s += std::to_string(v[i].numeric);
            s += v[i].candidate ? " 1\n" : " 0\n";
        }
        copy_str(out, len, s.c_str());
        return TURBO_OK;
    } catch (...) {
        return TURBO_E_INTERNAL;
    }
}

/* The GEMMs' stream-K in sessions made from now on, as
 * TURBO_CUDA_SK_STEPS would name it: 0 the kernels' own, 1 to 64 the
 * fewest k steps a block takes, -2 whole tiles to a block; -1 to read the
 * variable again. */
/* The GEMM variant, by the name a choices string gives its kernel
 * ("sw8w", "128x64/tf32"), that fails to launch in the sessions made next,
 * or NULL for none: the tuner skips it and a session that forces it is
 * refused. For the tests. */
void turbo_cuda_fail_variant(const char *name) {
    std::lock_guard<std::mutex> l(fail_lock);
    fail_name = name ? name : "";
}

void turbo_cuda_use_sk_steps(int32_t steps) { sk_override.store(steps, std::memory_order_relaxed); }

/* The FMA attention of sessions made from now on: 1 the key-split kernel
 * (TURBO_CUDA_ATTENTION=split), 0 the default, -1 to read the variable
 * again. */
void turbo_cuda_use_split_attention(int32_t split) {
    split_attention_override.store(split, std::memory_order_relaxed);
}

/* FASTEST's attention on the tensor cores in sessions made from now on:
 * 1 the kernel of 128 queries to a block, the default, 0 the kernel of
 * 64 (TURBO_CUDA_ATTENTION=64), -1 to read the variable again. */
void turbo_cuda_use_wide_attention(int32_t wide) { wide_attention_override.store(wide, std::memory_order_relaxed); }

/* The softmax of the attention of 128 queries in sessions made from now
 * on: 1 exp2f on scores scaled first (TURBO_CUDA_ATTENTION=exact), 0 the
 * default's ex2.approx, -1 to read the variable again. */
void turbo_cuda_use_exact_attention(int32_t exact) {
    exact_attention_override.store(exact, std::memory_order_relaxed);
}

/* The attention of 128 queries at heads of 32 in sessions made from now
 * on: 1 with 32 queries to each of four warps (TURBO_CUDA_ATTENTION=fa32),
 * 0 the default of 16 to each of eight, -1 to read the variable again. */
void turbo_cuda_use_fa32_attention(int32_t fa32) { fa32_attention_override.store(fa32, std::memory_order_relaxed); }

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

/* The F16 GEMMs of FASTEST sessions made from now on: 1 F16 accumulators
 * over each 64 terms (TURBO_CUDA_F16_ACCUMULATE=1), 0 F32 throughout (the
 * default), -1 to read the variable again. */
void turbo_cuda_use_f16_accumulate(int32_t f16) { f16_accumulate_override.store(f16, std::memory_order_relaxed); }

/* The pooling of sessions made from now on: 1 the kernel of a thread per
 * column (TURBO_CUDA_POOL=columns), 0 the default, -1 to read the
 * variable again. */
void turbo_cuda_use_column_pool(int32_t columns) { column_pool_override.store(columns, std::memory_order_relaxed); }

/* FASTEST's GELU in sessions made from now on: 1 erff, the default
 * (TURBO_CUDA_GELU=erf), 0 the fit (TURBO_CUDA_GELU=poly), -1 to read the
 * variable again. */
void turbo_cuda_use_gelu_erf(int32_t erf) { gelu_erf_override.store(erf, std::memory_order_relaxed); }

} // extern "C"
