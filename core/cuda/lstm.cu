/* SPDX-License-Identifier: Apache-2.0
 *
 * The batched LSTM of turbo::device::lstm on an NVIDIA GPU (#860), the
 * CPU form's (core/src/cpu/lstm.rs) rows and order: sequence s holds
 * rows start[s]..start[s] + len[s], the longest first, gates (i, f, g, o)
 * each `hidden` wide, a reverse direction reading each sequence from its
 * last row.
 *
 * One product makes every direction's input projections for every row
 * (or a run reads them from a table by row id); then each step is one
 * kernel over the sequences still running, which are a prefix of the
 * batch, and every direction at once: each thread owns one (direction,
 * sequence, unit), makes its four gates from the projection and the
 * recurrent product over the previous step's state, and writes its cell,
 * its state and its output row. The states are double buffered, since
 * every unit of a sequence reads the whole previous state.
 *
 * A non-finite gate or state is recorded as the least row it was found
 * at and refused when the run ends; the CPU form stops at the first one.
 * The host side checks shapes, lengths and inputs before it calls here.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int TILE = 16;

enum Status { OK = 0, NOT_FINITE = 1, NO_DEVICE = 2, NO_MEMORY = 3, FAILED = 4 };

/* The CPU form's sigmoid, either side of 0, so neither branch overflows. */
__device__ float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

/* A weight as F32, from F32 or (FASTEST) F16 storage. */
__device__ inline float ld(const float *p, int64_t i) { return p[i]; }
__device__ inline float ld(const __half *p, int64_t i) { return __half2float(p[i]); }

/* F32 weights rounded to F16, for FASTEST. */
__global__ void narrow(const float *src, __half *dst, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

/* out[m][n] = bias[n] + sum over k of a[m][k] w[n][k]: every row's input
 * projections, w the stacked input weights of every direction, F32 or F16,
 * the sums F32. */
template <typename W>
__global__ void project(const float *a, const W *w, const float *bias, float *out, int64_t m, int n, int k) {
    __shared__ float as[TILE][TILE + 1];
    __shared__ float ws[TILE][TILE + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int64_t row = (int64_t)blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;
    const int wrow = blockIdx.x * TILE + ty;
    float acc = 0.0f;
    for (int k0 = 0; k0 < k; k0 += TILE) {
        as[ty][tx] = row < m && k0 + tx < k ? a[row * k + k0 + tx] : 0.0f;
        ws[ty][tx] = wrow < n && k0 + tx < k ? ld(w, (int64_t)wrow * k + k0 + tx) : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc = fmaf(as[ty][i], ws[tx][i], acc);
        __syncthreads();
    }
    if (row < m && col < n) out[row * n + col] = bias[col] + acc;
}

struct Step {
    const float *proj;    /* [rows or table rows, pw] */
    const int64_t *index; /* row r's table row, or null for row r itself */
    const void *u;        /* [dirs][4h][h], F32 or F16 */
    const int64_t *len, *start;
    const float *state_in; /* [dirs][batch][h] */
    float *state_out, *cell;
    float *out; /* [rows, dirs * h] */
    unsigned long long *bad;
    const int *reverse;
    int64_t batch;
    int active, hidden, pw, width, t;
};

/* One step of every direction over the first `active` sequences. Block
 * (x, y, z): units x * 16.., sequences y * 16.., direction z. U is W: F32,
 * or F16 at FASTEST; the state, the cell and every sum stay F32. */
template <typename W> __global__ void step(Step p) {
    __shared__ float ss[TILE][TILE + 1];
    __shared__ float us[4][TILE][TILE + 1];
    const int tx = threadIdx.x, ty = threadIdx.y, k = blockIdx.z, h = p.hidden;
    const int j = blockIdx.x * TILE + tx;
    const int s = blockIdx.y * TILE + ty;
    const int ju = blockIdx.x * TILE + ty; /* the U row this thread loads */
    const float *state = p.state_in + (int64_t)k * p.batch * h;
    const W *u = static_cast<const W *>(p.u) + (int64_t)k * 4 * h * h;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (p.t > 0) {
        for (int i0 = 0; i0 < h; i0 += TILE) {
            const bool in = i0 + tx < h;
            ss[ty][tx] = s < p.active && in ? state[(int64_t)s * h + i0 + tx] : 0.0f;
            for (int q = 0; q < 4; q++)
                us[q][ty][tx] = ju < h && in ? ld(u, (int64_t)(q * h + ju) * h + i0 + tx) : 0.0f;
            __syncthreads();
            for (int i = 0; i < TILE; i++) {
                const float v = ss[ty][i];
                for (int q = 0; q < 4; q++) acc[q] = fmaf(v, us[q][tx][i], acc[q]);
            }
            __syncthreads();
        }
    }
    if (s >= p.active || j >= h) return;
    const int64_t n = p.len[s];
    const int64_t row = p.start[s] + (p.reverse[k] ? n - 1 - p.t : p.t);
    const int64_t src = p.index ? p.index[row] : row;
    const float *g = p.proj + src * p.pw + (int64_t)k * 4 * h;
    const float gi = g[j] + acc[0], gf = g[h + j] + acc[1], gg = g[2 * h + j] + acc[2], go = g[3 * h + j] + acc[3];
    const int64_t at = ((int64_t)k * p.batch + s) * h + j;
    const float c = sigmoid(gf) * p.cell[at] + sigmoid(gi) * tanhf(gg);
    const float st = sigmoid(go) * tanhf(c);
    const bool finite = isfinite(gi) && isfinite(gf) && isfinite(gg) && isfinite(go) && isfinite(st);
    if (!finite) atomicMin(p.bad, (unsigned long long)row);
    p.cell[at] = c;
    p.state_out[at] = st;
    p.out[row * p.width + (int64_t)k * h + j] = st;
}

int failed(cudaError_t e, const char *what, char *msg, size_t len) {
    (void)cudaGetLastError();
    snprintf(msg, len, "%s: %s", what, cudaGetErrorString(e));
    if (e == cudaErrorMemoryAllocation) return NO_MEMORY;
    if (e == cudaErrorNoDevice || e == cudaErrorInvalidDevice || e == cudaErrorInsufficientDriver) return NO_DEVICE;
    return FAILED;
}

#define TRY(expr, what)                                                                                                \
    do {                                                                                                               \
        const cudaError_t e_ = (expr);                                                                                 \
        if (e_ != cudaSuccess) return failed(e_, what, msg, len);                                                      \
    } while (0)

/* The calling thread's device set to one ordinal while this lives. */
class Scope {
  public:
    explicit Scope(int ordinal) {
        if (cudaGetDevice(&prev_) != cudaSuccess) {
            (void)cudaGetLastError();
            prev_ = -1;
        }
        status_ = prev_ == ordinal ? cudaSuccess : cudaSetDevice(ordinal);
        changed_ = prev_ != ordinal && status_ == cudaSuccess;
    }
    ~Scope() {
        if (changed_ && prev_ >= 0) (void)cudaSetDevice(prev_);
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    cudaError_t status() const { return status_; }

  private:
    int prev_ = -1;
    cudaError_t status_ = cudaSuccess;
    bool changed_ = false;
};

/* A device buffer that grows and never shrinks. */
struct Buffer {
    void *p = nullptr;
    size_t bytes = 0;
    cudaError_t fit(size_t want) {
        if (want <= bytes) return cudaSuccess;
        if (p) cudaFree(p);
        p = nullptr;
        bytes = 0;
        const cudaError_t e = cudaMalloc(&p, want);
        if (e == cudaSuccess) bytes = want;
        return e;
    }
    void release() {
        if (p) cudaFree(p);
        p = nullptr;
        bytes = 0;
    }
    ~Buffer() { release(); }
    template <typename T> T *as() const { return static_cast<T *>(p); }
};

} // namespace

struct turbo_cuda_lstm {
    int ordinal, input, hidden, dirs;
    /* FASTEST: W and U held as F16 (w16, u16) and the F32 copies freed. */
    bool f16;
    Buffer w, u, b, reverse, w16, u16;
};

struct turbo_cuda_lstm_table {
    int ordinal;
    Buffer values;
};

struct turbo_cuda_lstm_work {
    int ordinal = -1;
    cudaStream_t stream = nullptr;
    Buffer x, index, proj, out, len, start, state, cell, bad;
    std::vector<int64_t> starts;
};

extern "C" {

int turbo_cuda_lstm_new(int ordinal, int input, int hidden, int dirs, const float *w, const float *u, const float *b,
                        const int *reverse, int f16, turbo_cuda_lstm **out, char *msg, size_t len) {
    Scope scope(ordinal);
    TRY(scope.status(), "cudaSetDevice");
    auto *l = new turbo_cuda_lstm{ordinal, input, hidden, dirs, f16 != 0, {}, {}, {}, {}, {}, {}};
    const size_t g = 4 * (size_t)hidden * dirs;
    const struct {
        Buffer *dst;
        const void *src;
        size_t bytes;
    } parts[] = {{&l->w, w, g * input * sizeof(float)},
                 {&l->u, u, g * hidden * sizeof(float)},
                 {&l->b, b, g * sizeof(float)},
                 {&l->reverse, reverse, dirs * sizeof(int)}};
    for (const auto &part : parts) {
        cudaError_t e = part.dst->fit(part.bytes);
        if (e == cudaSuccess) e = cudaMemcpy(part.dst->p, part.src, part.bytes, cudaMemcpyHostToDevice);
        /* A copy from pageable memory can return before its DMA lands, and
         * a run's stream does not wait for the default one: wait here. */
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            delete l;
            return failed(e, "copying the LSTM weights to the device", msg, len);
        }
    }
    if (l->f16) {
        const struct {
            Buffer *src, *dst;
            size_t n;
        } halves[] = {{&l->w, &l->w16, g * input}, {&l->u, &l->u16, g * hidden}};
        for (const auto &h : halves) {
            cudaError_t e = h.dst->fit(h.n * sizeof(__half));
            if (e == cudaSuccess) {
                narrow<<<(unsigned)((h.n + 255) / 256), 256>>>(h.src->as<float>(), h.dst->as<__half>(), h.n);
                e = cudaGetLastError();
            }
            if (e == cudaSuccess) e = cudaDeviceSynchronize();
            if (e != cudaSuccess) {
                delete l;
                return failed(e, "narrowing the LSTM weights to F16", msg, len);
            }
            h.src->release();
        }
    }
    *out = l;
    return OK;
}

void turbo_cuda_lstm_free(turbo_cuda_lstm *l) {
    if (!l) return;
    Scope scope(l->ordinal);
    delete l;
}

int turbo_cuda_lstm_table_new(const turbo_cuda_lstm *l, const float *values, size_t count, turbo_cuda_lstm_table **out,
                              char *msg, size_t len) {
    Scope scope(l->ordinal);
    TRY(scope.status(), "cudaSetDevice");
    auto *t = new turbo_cuda_lstm_table{l->ordinal, {}};
    cudaError_t e = t->values.fit(count ? count * sizeof(float) : 1);
    if (e == cudaSuccess && count) e = cudaMemcpy(t->values.p, values, count * sizeof(float), cudaMemcpyHostToDevice);
    /* As for the weights: the copy has landed before a run reads it. */
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        delete t;
        return failed(e, "copying the LSTM table to the device", msg, len);
    }
    *out = t;
    return OK;
}

void turbo_cuda_lstm_table_free(turbo_cuda_lstm_table *t) {
    if (!t) return;
    Scope scope(t->ordinal);
    delete t;
}

void turbo_cuda_lstm_work_free(turbo_cuda_lstm_work *w) {
    if (!w) return;
    Scope scope(w->ordinal);
    if (w->stream) cudaStreamDestroy(w->stream);
    delete w;
}

/* The sequences of `lengths` (batch of them, the longest first, rows in
 * all) into out [rows, dirs * hidden]: from rows x [rows, input], or,
 * where x is null, from table rows by index [rows]. *work is made on
 * first use and kept. NOT_FINITE puts the least bad row in *bad_row. */
int turbo_cuda_lstm_run(const turbo_cuda_lstm *l, turbo_cuda_lstm_work **work, const float *x,
                        const turbo_cuda_lstm_table *table, const int64_t *index, const int64_t *lengths, int64_t batch,
                        int64_t rows, float *out, int64_t *bad_row, char *msg, size_t len) {
    Scope scope(l->ordinal);
    TRY(scope.status(), "cudaSetDevice");
    turbo_cuda_lstm_work *w = *work;
    if (w && w->ordinal != l->ordinal) {
        turbo_cuda_lstm_work_free(w);
        TRY(cudaSetDevice(l->ordinal), "cudaSetDevice");
        w = *work = nullptr;
    }
    if (!w) {
        w = new turbo_cuda_lstm_work();
        w->ordinal = l->ordinal;
        const cudaError_t e = cudaStreamCreateWithFlags(&w->stream, cudaStreamNonBlocking);
        if (e != cudaSuccess) {
            delete w;
            return failed(e, "cudaStreamCreateWithFlags", msg, len);
        }
        *work = w;
    }
    const int h = l->hidden, dirs = l->dirs, pw = 4 * h * dirs, width = h * dirs;
    w->starts.assign(batch, 0);
    for (int64_t s = 1; s < batch; s++) w->starts[s] = w->starts[s - 1] + lengths[s - 1];
    const size_t f = sizeof(float), state = (size_t)dirs * batch * h * f;
    TRY(w->out.fit(rows * width * f), "allocating the LSTM output");
    TRY(w->len.fit(batch * sizeof(int64_t)), "allocating the LSTM lengths");
    TRY(w->start.fit(batch * sizeof(int64_t)), "allocating the LSTM starts");
    TRY(w->state.fit(2 * state), "allocating the LSTM states");
    TRY(w->cell.fit(state), "allocating the LSTM cells");
    TRY(w->bad.fit(sizeof(unsigned long long)), "allocating the LSTM flag");
    cudaStream_t st = w->stream;
    const float *proj;
    if (x) {
        TRY(w->x.fit(rows * l->input * f), "allocating the LSTM input");
        TRY(w->proj.fit(rows * pw * f), "allocating the LSTM projections");
        TRY(cudaMemcpyAsync(w->x.p, x, rows * l->input * f, cudaMemcpyHostToDevice, st), "copying the LSTM input");
        const dim3 grid((pw + TILE - 1) / TILE, (unsigned)((rows + TILE - 1) / TILE));
        if (l->f16)
            project<<<grid, dim3(TILE, TILE), 0, st>>>(w->x.as<float>(), l->w16.as<__half>(), l->b.as<float>(),
                                                       w->proj.as<float>(), rows, pw, l->input);
        else
            project<<<grid, dim3(TILE, TILE), 0, st>>>(w->x.as<float>(), l->w.as<float>(), l->b.as<float>(),
                                                       w->proj.as<float>(), rows, pw, l->input);
        TRY(cudaGetLastError(), "launching the LSTM projection");
        proj = w->proj.as<float>();
    } else {
        TRY(w->index.fit(rows * sizeof(int64_t)), "allocating the LSTM index");
        TRY(cudaMemcpyAsync(w->index.p, index, rows * sizeof(int64_t), cudaMemcpyHostToDevice, st),
            "copying the LSTM index");
        proj = table->values.as<float>();
    }
    TRY(cudaMemcpyAsync(w->len.p, lengths, batch * sizeof(int64_t), cudaMemcpyHostToDevice, st),
        "copying the LSTM lengths");
    TRY(cudaMemcpyAsync(w->start.p, w->starts.data(), batch * sizeof(int64_t), cudaMemcpyHostToDevice, st),
        "copying the LSTM starts");
    TRY(cudaMemsetAsync(w->cell.p, 0, state, st), "clearing the LSTM cells");
    TRY(cudaMemsetAsync(w->bad.p, 0xff, sizeof(unsigned long long), st), "clearing the LSTM flag");
    Step p{};
    p.proj = proj;
    p.index = x ? nullptr : w->index.as<int64_t>();
    p.u = l->f16 ? l->u16.p : l->u.p;
    p.len = w->len.as<int64_t>();
    p.start = w->start.as<int64_t>();
    p.cell = w->cell.as<float>();
    p.out = w->out.as<float>();
    p.bad = w->bad.as<unsigned long long>();
    p.reverse = l->reverse.as<int>();
    p.batch = batch;
    p.hidden = h;
    p.pw = pw;
    p.width = width;
    float *states[2] = {w->state.as<float>(), w->state.as<float>() + (size_t)dirs * batch * h};
    int64_t active = batch;
    for (int64_t t = 0; t < lengths[0]; t++) {
        while (lengths[active - 1] <= t) active--;
        p.active = (int)active;
        p.t = (int)t;
        p.state_in = states[t & 1];
        p.state_out = states[(t + 1) & 1];
        const dim3 grid((h + TILE - 1) / TILE, (unsigned)((active + TILE - 1) / TILE), dirs);
        if (l->f16)
            step<__half><<<grid, dim3(TILE, TILE), 0, st>>>(p);
        else
            step<float><<<grid, dim3(TILE, TILE), 0, st>>>(p);
        TRY(cudaGetLastError(), "launching an LSTM step");
    }
    unsigned long long bad = 0;
    TRY(cudaMemcpyAsync(out, w->out.p, rows * width * f, cudaMemcpyDeviceToHost, st), "copying the LSTM states");
    TRY(cudaMemcpyAsync(&bad, w->bad.p, sizeof bad, cudaMemcpyDeviceToHost, st), "copying the LSTM flag");
    TRY(cudaStreamSynchronize(st), "running the LSTM");
    if (bad != ~0ULL) {
        *bad_row = (int64_t)bad;
        return NOT_FINITE;
    }
    return OK;
}

} /* extern "C" */
