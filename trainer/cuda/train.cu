/* SPDX-License-Identifier: Apache-2.0
 *
 * The encoder trainer's CUDA kernels: the ops of trainer/src/encoder on
 * the device, one for one, and the C API (train.h) the Rust side
 * launches them through. Written to run under shim.h as plain C++ as
 * well: no shared memory, no barriers, no atomics; each thread owns its
 * outputs and sums them in a fixed order. The first leg of the CUDA
 * backward (#1519): the GEMM is a register-tiled one over global memory,
 * to be replaced by a tiled one when the path is proven.
 */

#include "train.h"

#include <math.h>

#ifndef TT_LAUNCH
#include <cuda_runtime.h>
#define TT_LAUNCH(kernel, grid, block, ...) kernel<<<(grid), (block)>>>(__VA_ARGS__)
#endif

namespace {

constexpr int BLOCK = 256;
constexpr unsigned MAX_GRID = 4096;

/* Blocks of BLOCK threads for `count` items, in a grid-stride loop. */
unsigned grid_for(size_t count) {
    size_t blocks = (count + BLOCK - 1) / BLOCK;
    return (unsigned)(blocks < MAX_GRID ? (blocks ? blocks : 1) : MAX_GRID);
}

__device__ __forceinline__ size_t stride() { return (size_t)gridDim.x * blockDim.x; }
__device__ __forceinline__ size_t first() { return (size_t)blockIdx.x * blockDim.x + threadIdx.x; }

/* ---- GEMM: a 4 x 4 micro-tile a thread, 16 x 16 threads a block. ---- */

constexpr int MT = 4;
constexpr int GEMM_THREADS = 16;

struct GemmArgs {
    const float *a, *b;
    float *c;
    int m, n, k, ta, tb;
    float alpha, beta;
};

__device__ __forceinline__ float a_at(const GemmArgs &g, int i, int kk) {
    return g.ta ? g.a[(size_t)kk * g.m + i] : g.a[(size_t)i * g.k + kk];
}
__device__ __forceinline__ float b_at(const GemmArgs &g, int kk, int j) {
    return g.tb ? g.b[(size_t)j * g.k + kk] : g.b[(size_t)kk * g.n + j];
}

__global__ void gemm_kernel(GemmArgs g) {
    int row0 = (blockIdx.y * GEMM_THREADS + threadIdx.y) * MT;
    int col0 = (blockIdx.x * GEMM_THREADS + threadIdx.x) * MT;
    if (row0 >= g.m || col0 >= g.n) return;
    float acc[MT][MT];
    for (int i = 0; i < MT; i++)
        for (int j = 0; j < MT; j++) acc[i][j] = 0.0f;
    int rows = g.m - row0 < MT ? g.m - row0 : MT;
    int cols = g.n - col0 < MT ? g.n - col0 : MT;
    for (int kk = 0; kk < g.k; kk++) {
        float av[MT], bv[MT];
        for (int i = 0; i < MT; i++) av[i] = i < rows ? a_at(g, row0 + i, kk) : 0.0f;
        for (int j = 0; j < MT; j++) bv[j] = j < cols ? b_at(g, kk, col0 + j) : 0.0f;
        for (int i = 0; i < MT; i++)
            for (int j = 0; j < MT; j++) acc[i][j] += av[i] * bv[j];
    }
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            float *c = g.c + (size_t)(row0 + i) * g.n + col0 + j;
            *c = g.beta == 0.0f ? g.alpha * acc[i][j] : g.alpha * acc[i][j] + g.beta * *c;
        }
}

/* ---- The word table. ---- */

__global__ void gather_kernel(const float *table, const int32_t *ids, int n, int h, float *out) {
    size_t total = (size_t)n * h;
    for (size_t i = first(); i < total; i += stride()) {
        size_t t = i / h, c = i % h;
        out[i] = table[(size_t)ids[t] * h + c];
    }
}

/* One thread a column, the tokens in order: a repeated id sums in order. */
__global__ void scatter_add_kernel(const float *rows, const int32_t *ids, int n, int h, float *table) {
    for (size_t c = first(); c < (size_t)h; c += stride())
        for (int t = 0; t < n; t++) table[(size_t)ids[t] * h + c] += rows[(size_t)t * h + c];
}

/* ---- LayerNorm, the statistics in double as the CPU sums them. ---- */

__global__ void layernorm_kernel(const float *x, const float *w, int rows, int width, double eps, float *y,
                                 double *mean, double *rstd) {
    for (size_t r = first(); r < (size_t)rows; r += stride()) {
        const float *xr = x + r * width;
        double sum = 0.0;
        for (int i = 0; i < width; i++) sum += (double)xr[i];
        double mu = sum / width;
        double var = 0.0;
        for (int i = 0; i < width; i++) {
            double d = (double)xr[i] - mu;
            var += d * d;
        }
        var /= width;
        double rs = 1.0 / sqrt(var + eps);
        mean[r] = mu;
        rstd[r] = rs;
        for (int i = 0; i < width; i++) y[r * width + i] = (float)(((double)xr[i] - mu) * rs) * w[i];
    }
}

/* dx by row: rstd (g - mean g - xhat mean(g xhat)), g = dy w, in double. */
__global__ void layernorm_dx_kernel(const float *x, const float *w, const double *mean, const double *rstd,
                                    const float *dy, int rows, int width, float *dx) {
    for (size_t r = first(); r < (size_t)rows; r += stride()) {
        const float *xr = x + r * width, *dyr = dy + r * width;
        double mu = mean[r], rs = rstd[r], m1 = 0.0, m2 = 0.0;
        for (int i = 0; i < width; i++) {
            double g = (double)dyr[i] * (double)w[i];
            m1 += g;
            m2 += g * (((double)xr[i] - mu) * rs);
        }
        m1 /= width;
        m2 /= width;
        for (int i = 0; i < width; i++) {
            double g = (double)dyr[i] * (double)w[i], xhat = ((double)xr[i] - mu) * rs;
            dx[r * width + i] += (float)(rs * (g - m1 - xhat * m2));
        }
    }
}

/* dw by column, the rows in order. */
__global__ void layernorm_dw_kernel(const float *x, const double *mean, const double *rstd, const float *dy,
                                    int rows, int width, float *dw) {
    for (size_t i = first(); i < (size_t)width; i += stride()) {
        float acc = dw[i];
        for (int r = 0; r < rows; r++)
            acc += (float)((double)dy[(size_t)r * width + i] * (((double)x[(size_t)r * width + i] - mean[r]) * rstd[r]));
        dw[i] = acc;
    }
}

/* ---- Rotary positions, rotate_half. ---- */

__global__ void rope_kernel(float *x, int rows, int ld, int cols, int pairs, const float *cs, const float *sn,
                            int back) {
    int d = 2 * pairs, heads = cols / d;
    size_t total = (size_t)rows * heads * pairs;
    for (size_t idx = first(); idx < total; idx += stride()) {
        int i = (int)(idx % pairs);
        size_t rest = idx / pairs;
        int hd = (int)(rest % heads);
        int t = (int)(rest / heads);
        float *a = x + (size_t)t * ld + hd * d + i, *b = a + pairs;
        float c = cs[(size_t)t * pairs + i], s = sn[(size_t)t * pairs + i];
        float x1 = *a, x2 = *b;
        if (back) {
            *a = x1 * c + x2 * s;
            *b = x2 * c - x1 * s;
        } else {
            *a = x1 * c - x2 * s;
            *b = x2 * c + x1 * s;
        }
    }
}

/* ---- Attention, one thread a (head, query) or (head, key). ---- */

__device__ __forceinline__ void keys(int i, int n, int window, int *lo, int *hi) {
    if (window < 0) {
        *lo = 0;
        *hi = n;
    } else {
        *lo = i - window < 0 ? 0 : i - window;
        *hi = i + window + 1 < n ? i + window + 1 : n;
    }
}

__device__ __forceinline__ float bf16_round(float v) {
    if (v != v) return v;
    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t lsb = (bits >> 16) & 1u;
    bits = (bits + 0x7FFFu + lsb) & 0xFFFF0000u;
    float out;
    memcpy(&out, &bits, 4);
    return out;
}

__device__ __forceinline__ float dotf(const float *a, const float *b, int d) {
    float s = 0.0f;
    for (int c = 0; c < d; c++) s += a[c] * b[c];
    return s;
}

__global__ void attention_probs_kernel(const float *qkv, int n, int hidden, int heads, int window, int round,
                                       float *p) {
    int d = hidden / heads, ld = 3 * hidden;
    float scale = (float)(1.0 / sqrt((double)d));
    for (size_t idx = first(); idx < (size_t)heads * n; idx += stride()) {
        int h = (int)(idx / n), i = (int)(idx % n);
        const float *q = qkv + (size_t)i * ld + h * d;
        float *row = p + ((size_t)h * n + i) * n;
        int lo, hi;
        keys(i, n, window, &lo, &hi);
        for (int j = 0; j < n; j++) row[j] = 0.0f;
        float mx = -INFINITY;
        for (int j = lo; j < hi; j++) {
            float s = dotf(q, qkv + (size_t)j * ld + hidden + h * d, d) * scale;
            row[j] = s;
            if (s > mx) mx = s;
        }
        float sum = 0.0f;
        for (int j = lo; j < hi; j++) {
            row[j] = expf(row[j] - mx);
            sum += row[j];
        }
        float inv = 1.0f / sum;
        for (int j = lo; j < hi; j++) row[j] = round ? bf16_round(row[j] * inv) : row[j] * inv;
    }
}

__global__ void attention_ctx_kernel(const float *qkv, const float *p, int n, int hidden, int heads, int window,
                                     float *ctx) {
    int d = hidden / heads, ld = 3 * hidden;
    for (size_t idx = first(); idx < (size_t)heads * n; idx += stride()) {
        int h = (int)(idx / n), i = (int)(idx % n);
        const float *row = p + ((size_t)h * n + i) * n;
        float *out = ctx + (size_t)i * hidden + h * d;
        int lo, hi;
        keys(i, n, window, &lo, &hi);
        for (int c = 0; c < d; c++) out[c] = 0.0f;
        for (int j = lo; j < hi; j++) {
            const float *v = qkv + (size_t)j * ld + 2 * hidden + h * d;
            for (int c = 0; c < d; c++) out[c] += row[j] * v[c];
        }
    }
}

__global__ void attention_ds_kernel(const float *qkv, const float *p, const float *dctx, int n, int hidden,
                                    int heads, int window, float *ds) {
    int d = hidden / heads, ld = 3 * hidden;
    float scale = (float)(1.0 / sqrt((double)d));
    for (size_t idx = first(); idx < (size_t)heads * n; idx += stride()) {
        int h = (int)(idx / n), i = (int)(idx % n);
        const float *row = p + ((size_t)h * n + i) * n, *dout = dctx + (size_t)i * hidden + h * d;
        float *drow = ds + ((size_t)h * n + i) * n;
        int lo, hi;
        keys(i, n, window, &lo, &hi);
        for (int j = 0; j < n; j++) drow[j] = 0.0f;
        float s = 0.0f;
        for (int j = lo; j < hi; j++) {
            float dp = dotf(dout, qkv + (size_t)j * ld + 2 * hidden + h * d, d);
            drow[j] = dp;
            s += row[j] * dp;
        }
        for (int j = lo; j < hi; j++) drow[j] = row[j] * (drow[j] - s) * scale;
    }
}

__global__ void attention_dq_kernel(const float *qkv, const float *ds, int n, int hidden, int heads, int window,
                                    float *dqkv) {
    int d = hidden / heads, ld = 3 * hidden;
    for (size_t idx = first(); idx < (size_t)heads * n; idx += stride()) {
        int h = (int)(idx / n), i = (int)(idx % n);
        const float *drow = ds + ((size_t)h * n + i) * n;
        float *dq = dqkv + (size_t)i * ld + h * d;
        int lo, hi;
        keys(i, n, window, &lo, &hi);
        for (int c = 0; c < d; c++) {
            float acc = 0.0f;
            for (int j = lo; j < hi; j++) acc += drow[j] * qkv[(size_t)j * ld + hidden + h * d + c];
            dq[c] += acc;
        }
    }
}

/* By key j: the queries i whose window holds j are those with |i - j| <=
 * window, in order. */
__global__ void attention_dkv_kernel(const float *qkv, const float *p, const float *ds, const float *dctx, int n,
                                     int hidden, int heads, int window, float *dqkv) {
    int d = hidden / heads, ld = 3 * hidden;
    for (size_t idx = first(); idx < (size_t)heads * n; idx += stride()) {
        int h = (int)(idx / n), j = (int)(idx % n);
        float *dk = dqkv + (size_t)j * ld + hidden + h * d, *dv = dk + hidden;
        int lo, hi;
        keys(j, n, window, &lo, &hi);
        for (int c = 0; c < d; c++) {
            float ak = 0.0f, av = 0.0f;
            for (int i = lo; i < hi; i++) {
                size_t at = ((size_t)h * n + i) * n + j;
                ak += ds[at] * qkv[(size_t)i * ld + h * d + c];
                av += p[at] * dctx[(size_t)i * hidden + h * d + c];
            }
            dk[c] += ak;
            dv[c] += av;
        }
    }
}

/* ---- The gated GELU, by the error function in double. ---- */

constexpr double FRAC_1_SQRT_2 = 0.707106781186547524400844362104849039;
constexpr double SQRT_2PI = 2.5066282746310007;

__device__ __forceinline__ float gelu(float x) {
    double v = x;
    return (float)(0.5 * v * (1.0 + erf(v * FRAC_1_SQRT_2)));
}
__device__ __forceinline__ float gelu_grad(float x) {
    double v = x;
    return (float)(0.5 * (1.0 + erf(v * FRAC_1_SQRT_2)) + v * exp(-0.5 * v * v) / SQRT_2PI);
}

__global__ void geglu_kernel(const float *pre, int rows, int inter, float *act) {
    size_t total = (size_t)rows * inter;
    for (size_t idx = first(); idx < total; idx += stride()) {
        size_t r = idx / inter, i = idx % inter;
        const float *p = pre + r * 2 * inter;
        act[idx] = gelu(p[i]) * p[inter + i];
    }
}

__global__ void geglu_back_kernel(const float *pre, const float *dact, int rows, int inter, float *dpre) {
    size_t total = (size_t)rows * inter;
    for (size_t idx = first(); idx < total; idx += stride()) {
        size_t r = idx / inter, i = idx % inter;
        const float *p = pre + r * 2 * inter;
        float *dp = dpre + r * 2 * inter;
        float da = dact[idx];
        dp[i] += da * p[inter + i] * gelu_grad(p[i]);
        dp[inter + i] += da * gelu(p[i]);
    }
}

/* ---- Dropout from the trainer's SplitMix64, the i-th draw directly. ---- */

__device__ __forceinline__ uint64_t splitmix(uint64_t state) {
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

__global__ void dropout_kernel(float *x, size_t count, double p, uint64_t key, float scale) {
    for (size_t i = first(); i < count; i += stride()) {
        uint64_t z = splitmix(key + (i + 1) * 0x9e3779b97f4a7c15ull);
        double unit = (double)(z >> 11) / 9007199254740992.0;
        x[i] = unit < p ? 0.0f : x[i] * scale;
    }
}

/* ---- Elementwise. ---- */

__global__ void axpy_kernel(float *y, const float *x, size_t count, float alpha) {
    for (size_t i = first(); i < count; i += stride()) y[i] += alpha * x[i];
}

__global__ void round_bf16_kernel(float *x, size_t count) {
    for (size_t i = first(); i < count; i += stride()) x[i] = bf16_round(x[i]);
}

int launched() { return (int)cudaGetLastError(); }

} // namespace

extern "C" {

int tt_device_count(int *n) { return (int)cudaGetDeviceCount(n); }
int tt_set_device(int device) { return (int)cudaSetDevice(device); }
const char *tt_error_string(int error) { return cudaGetErrorString((cudaError_t)error); }

int tt_malloc(size_t bytes, void **p) { return (int)cudaMalloc(p, bytes); }
int tt_free(void *p) { return (int)cudaFree(p); }
int tt_upload(void *dst, const void *src, size_t bytes) {
    return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}
int tt_download(void *dst, const void *src, size_t bytes) {
    return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}
int tt_copy(void *dst, const void *src, size_t bytes) {
    return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
}
int tt_zero(void *dst, size_t bytes) { return (int)cudaMemset(dst, 0, bytes); }
int tt_sync(void) { return (int)cudaDeviceSynchronize(); }

int tt_gemm(const float *a, const float *b, float *c, int m, int n, int k, int ta, int tb, float alpha,
            float beta) {
    if (m <= 0 || n <= 0 || k < 0) return (int)cudaErrorInvalidValue;
    GemmArgs g = {a, b, c, m, n, k, ta, tb, alpha, beta};
    dim3 block(GEMM_THREADS, GEMM_THREADS);
    dim3 grid((n + GEMM_THREADS * MT - 1) / (GEMM_THREADS * MT), (m + GEMM_THREADS * MT - 1) / (GEMM_THREADS * MT));
    TT_LAUNCH(gemm_kernel, grid, block, g);
    return launched();
}

int tt_gather(const float *table, const int32_t *ids, int n, int h, float *out) {
    TT_LAUNCH(gather_kernel, grid_for((size_t)n * h), BLOCK, table, ids, n, h, out);
    return launched();
}

int tt_scatter_add(const float *rows, const int32_t *ids, int n, int h, float *table) {
    TT_LAUNCH(scatter_add_kernel, grid_for(h), BLOCK, rows, ids, n, h, table);
    return launched();
}

int tt_layernorm(const float *x, const float *w, int rows, int width, double eps, float *y, double *mean,
                 double *rstd) {
    TT_LAUNCH(layernorm_kernel, grid_for(rows), BLOCK, x, w, rows, width, eps, y, mean, rstd);
    return launched();
}

int tt_layernorm_back(const float *x, const float *w, const double *mean, const double *rstd, const float *dy,
                      int rows, int width, float *dx, float *dw) {
    TT_LAUNCH(layernorm_dx_kernel, grid_for(rows), BLOCK, x, w, mean, rstd, dy, rows, width, dx);
    int e = launched();
    if (e) return e;
    if (dw) TT_LAUNCH(layernorm_dw_kernel, grid_for(width), BLOCK, x, mean, rstd, dy, rows, width, dw);
    return launched();
}

int tt_rope(float *x, int rows, int ld, int cols, int pairs, const float *cs, const float *sn, int back) {
    if (pairs <= 0 || cols % (2 * pairs) != 0) return (int)cudaErrorInvalidValue;
    size_t total = (size_t)rows * (cols / (2 * pairs)) * pairs;
    TT_LAUNCH(rope_kernel, grid_for(total), BLOCK, x, rows, ld, cols, pairs, cs, sn, back);
    return launched();
}

int tt_attention_probs(const float *qkv, int n, int hidden, int heads, int window, int round, float *p) {
    TT_LAUNCH(attention_probs_kernel, grid_for((size_t)heads * n), BLOCK, qkv, n, hidden, heads, window, round, p);
    return launched();
}

int tt_attention_ctx(const float *qkv, const float *p, int n, int hidden, int heads, int window, float *ctx) {
    TT_LAUNCH(attention_ctx_kernel, grid_for((size_t)heads * n), BLOCK, qkv, p, n, hidden, heads, window, ctx);
    return launched();
}

int tt_attention_ds(const float *qkv, const float *p, const float *dctx, int n, int hidden, int heads, int window,
                    float *ds) {
    TT_LAUNCH(attention_ds_kernel, grid_for((size_t)heads * n), BLOCK, qkv, p, dctx, n, hidden, heads, window, ds);
    return launched();
}

int tt_attention_dq(const float *qkv, const float *ds, int n, int hidden, int heads, int window, float *dqkv) {
    TT_LAUNCH(attention_dq_kernel, grid_for((size_t)heads * n), BLOCK, qkv, ds, n, hidden, heads, window, dqkv);
    return launched();
}

int tt_attention_dkv(const float *qkv, const float *p, const float *ds, const float *dctx, int n, int hidden,
                     int heads, int window, float *dqkv) {
    TT_LAUNCH(attention_dkv_kernel, grid_for((size_t)heads * n), BLOCK, qkv, p, ds, dctx, n, hidden, heads, window,
              dqkv);
    return launched();
}

int tt_geglu(const float *pre, int rows, int inter, float *act) {
    TT_LAUNCH(geglu_kernel, grid_for((size_t)rows * inter), BLOCK, pre, rows, inter, act);
    return launched();
}

int tt_geglu_back(const float *pre, const float *dact, int rows, int inter, float *dpre) {
    TT_LAUNCH(geglu_back_kernel, grid_for((size_t)rows * inter), BLOCK, pre, dact, rows, inter, dpre);
    return launched();
}

int tt_dropout(float *x, size_t count, double p, uint64_t key) {
    if (p <= 0.0) return 0;
    if (!(p < 1.0)) return (int)cudaErrorInvalidValue;
    float scale = (float)(1.0 / (1.0 - p));
    TT_LAUNCH(dropout_kernel, grid_for(count), BLOCK, x, count, p, key, scale);
    return launched();
}

int tt_axpy(float *y, const float *x, size_t count, float alpha) {
    TT_LAUNCH(axpy_kernel, grid_for(count), BLOCK, y, x, count, alpha);
    return launched();
}

int tt_round_bf16(float *x, size_t count) {
    TT_LAUNCH(round_bf16_kernel, grid_for(count), BLOCK, x, count);
    return launched();
}

} // extern "C"
