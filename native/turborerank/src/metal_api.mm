// SPDX-License-Identifier: Apache-2.0
//
// Apple Metal MiniLM-L6 BertForSequenceClassification.
// Token workspace: MTLResourceStorageModeShared (caller writes unified
// memory). Kernels bind those MTLBuffers — no std::vector, no extra
// token copy. Weights + activation scratch are reserved at load.
// GEMM / LN / GELU / attention / pooler match the CPU linear_nt graph.

#ifdef TURBORERANK_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_api.hpp"
#include "reranker.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace turborerank {
namespace impl {
namespace {

static const char *kMetalSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct EmbedParams {
    uint seq;
    uint hidden;
    uint word_rows;
    uint pos_rows;
    uint type_rows;
};

struct LnParams {
    uint seq;
    uint hidden;
    float eps;
};

struct LinearParams {
    uint seq;
    uint k;
    uint out;
    uint has_bias;
};

struct ElemParams {
    uint n;
};

struct AttnParams {
    uint seq;
    uint hidden;
    uint heads;
    uint dh;
    float scale;
};

struct PoolParams {
    uint hidden;
};

kernel void embed_kernel(
    device float *x [[buffer(0)]],
    device const int *ids [[buffer(1)]],
    device const int *pos [[buffer(2)]],
    device const int *types [[buffer(3)]],
    device const float *word [[buffer(4)]],
    device const float *pos_w [[buffer(5)]],
    device const float *type_w [[buffer(6)]],
    constant EmbedParams &p [[buffer(7)]],
    uint t [[thread_position_in_grid]]
) {
    if (t >= p.seq) return;
    const uint uid = uint(ids[t]);
    const uint upos = uint(pos[t]);
    const uint utyp = uint(types[t]);
    device float *row = x + ulong(t) * p.hidden;
    if (uid < p.word_rows && upos < p.pos_rows && utyp < p.type_rows) {
        const device float *we = word + ulong(uid) * p.hidden;
        const device float *pe = pos_w + ulong(upos) * p.hidden;
        const device float *te = type_w + ulong(utyp) * p.hidden;
        for (uint h = 0; h < p.hidden; ++h) {
            row[h] = we[h] + pe[h] + te[h];
        }
    } else {
        for (uint h = 0; h < p.hidden; ++h) {
            row[h] = 0.0f;
        }
    }
}

// One thread per token: same reduction order as the CPU kernel.
kernel void layer_norm_kernel(
    device float *x [[buffer(0)]],
    device const float *gamma [[buffer(1)]],
    device const float *beta [[buffer(2)]],
    constant LnParams &p [[buffer(3)]],
    uint t [[thread_position_in_grid]]
) {
    if (t >= p.seq) return;
    device float *row = x + ulong(t) * p.hidden;
    float mean = 0.0f;
    for (uint i = 0; i < p.hidden; ++i) {
        mean += row[i];
    }
    mean /= float(p.hidden);
    float var = 0.0f;
    for (uint i = 0; i < p.hidden; ++i) {
        const float d = row[i] - mean;
        var += d * d;
    }
    var /= float(p.hidden);
    const float inv = rsqrt(var + p.eps);
    for (uint i = 0; i < p.hidden; ++i) {
        row[i] = (row[i] - mean) * inv * gamma[i] + beta[i];
    }
}

// Exact CPU linear_nt: y[s,o] = bias[o] + dot(x[s], W[o]).
kernel void linear_nt_kernel(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant LinearParams &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint o = gid.x;
    const uint s = gid.y;
    if (s >= p.seq || o >= p.out) return;
    const device float *xr = x + ulong(s) * p.k;
    const device float *wr = w + ulong(o) * p.k;
    float acc = p.has_bias != 0 ? bias[o] : 0.0f;
    for (uint t = 0; t < p.k; ++t) {
        acc += xr[t] * wr[t];
    }
    y[ulong(s) * p.out + o] = acc;
}

kernel void gelu_erf_kernel(
    device float *x [[buffer(0)]],
    constant ElemParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.n) return;
    const float v = x[i];
    x[i] = 0.5f * v * (1.0f + erf(v * 0.7071067811865476f));
}

kernel void add_inplace_kernel(
    device float *x [[buffer(0)]],
    device const float *r [[buffer(1)]],
    constant ElemParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.n) return;
    x[i] += r[i];
}

kernel void residual_from_ctx_kernel(
    device float *x [[buffer(0)]],
    device const float *ctx [[buffer(1)]],
    device const float *residual [[buffer(2)]],
    constant ElemParams &p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.n) return;
    x[i] = ctx[i] + residual[i];
}

kernel void copy_f32_kernel(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    constant ElemParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.n) return;
    dst[i] = src[i];
}

kernel void zero_f32_kernel(
    device float *x [[buffer(0)]],
    constant ElemParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.n) return;
    x[i] = 0.0f;
}

kernel void attention_scores_kernel(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device float *attn [[buffer(2)]],
    device const int *mask [[buffer(3)]],
    constant AttnParams &p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint j = gid.x;
    const uint i = gid.y;
    const uint h = gid.z;
    if (h >= p.heads || i >= p.seq || j >= p.seq) return;
    const device float *qi = q + ulong(i) * p.hidden + h * p.dh;
    const device float *kj = k + ulong(j) * p.hidden + h * p.dh;
    float dot = 0.0f;
    for (uint d = 0; d < p.dh; ++d) {
        dot += qi[d] * kj[d];
    }
    float s = dot * p.scale;
    if (mask[j] == 0) {
        s = -10000.0f;
    }
    attn[(ulong(h) * p.seq + i) * p.seq + j] = s;
}

kernel void softmax_rows_kernel(
    device float *attn [[buffer(0)]],
    constant AttnParams &p [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint h = gid.y;
    if (h >= p.heads || i >= p.seq) return;
    device float *row = attn + (ulong(h) * p.seq + i) * p.seq;
    float m = row[0];
    for (uint j = 1; j < p.seq; ++j) {
        if (row[j] > m) m = row[j];
    }
    float sum = 0.0f;
    for (uint j = 0; j < p.seq; ++j) {
        row[j] = exp(row[j] - m);
        sum += row[j];
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint j = 0; j < p.seq; ++j) {
        row[j] *= inv;
    }
}

kernel void attention_ctx_kernel(
    device const float *attn [[buffer(0)]],
    device const float *v [[buffer(1)]],
    device float *ctx [[buffer(2)]],
    constant AttnParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint h = gid.y;
    if (h >= p.heads || i >= p.seq) return;
    const device float *srow = attn + (ulong(h) * p.seq + i) * p.seq;
    device float *out = ctx + ulong(i) * p.hidden + h * p.dh;
    for (uint d = 0; d < p.dh; ++d) {
        float acc = 0.0f;
        for (uint j = 0; j < p.seq; ++j) {
            const device float *vj = v + ulong(j) * p.hidden + h * p.dh;
            acc += srow[j] * vj[d];
        }
        out[d] = acc;
    }
}

kernel void pooler_kernel(
    device const float *cls [[buffer(0)]],
    device const float *pw [[buffer(1)]],
    device const float *pb [[buffer(2)]],
    device float *pooled [[buffer(3)]],
    constant PoolParams &p [[buffer(4)]],
    uint o [[thread_position_in_grid]]
) {
    if (o >= p.hidden) return;
    float acc = pb[o];
    const device float *wr = pw + ulong(o) * p.hidden;
    for (uint h = 0; h < p.hidden; ++h) {
        acc += cls[h] * wr[h];
    }
    pooled[o] = tanh(acc);
}

kernel void classifier_kernel(
    device const float *head_in [[buffer(0)]],
    device const float *cw [[buffer(1)]],
    device const float *cb [[buffer(2)]],
    device float *logit [[buffer(3)]],
    constant PoolParams &p [[buffer(4)]]
) {
    float acc = cb[0];
    for (uint h = 0; h < p.hidden; ++h) {
        acc += head_in[h] * cw[h];
    }
    *logit = acc;
}
)METAL";

struct SharedEntry {
    id<MTLBuffer> buffer;
    size_t bytes;
};

struct SharedRegistry {
    std::mutex mu;
    std::unordered_map<const void *, SharedEntry> map;
};

SharedRegistry &registry() {
    static SharedRegistry r;
    return r;
}

struct MetalCtx {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> embed = nil;
    id<MTLComputePipelineState> ln = nil;
    id<MTLComputePipelineState> linear = nil;
    id<MTLComputePipelineState> gelu = nil;
    id<MTLComputePipelineState> add = nil;
    id<MTLComputePipelineState> residual = nil;
    id<MTLComputePipelineState> copyf = nil;
    id<MTLComputePipelineState> zerof = nil;
    id<MTLComputePipelineState> scores = nil;
    id<MTLComputePipelineState> softmax = nil;
    id<MTLComputePipelineState> ctx = nil;
    id<MTLComputePipelineState> pooler = nil;
    id<MTLComputePipelineState> classifier = nil;
    bool ready = false;
    std::string why;
};

MetalCtx &ctx() {
    static MetalCtx c;
    return c;
}

void init_ctx_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        MetalCtx &c = ctx();
        c.device = MTLCreateSystemDefaultDevice();
        if (c.device == nil) {
            c.why = "TURBORERANK_DEVICE_METAL requested but "
                    "MTLCreateSystemDefaultDevice returned nil. "
                    "Refusing CPU fallback.";
            return;
        }
        if (!c.device.hasUnifiedMemory) {
            c.why = "TURBORERANK_DEVICE_METAL requested but the GPU has no "
                    "unified memory (MTLResourceStorageModeShared token "
                    "workspace requires it). Refusing CPU fallback.";
            return;
        }
        c.queue = [c.device newCommandQueue];
        if (c.queue == nil) {
            c.why = "TURBORERANK_DEVICE_METAL: failed to create MTLCommandQueue. "
                    "Refusing CPU fallback.";
            return;
        }
        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:kMetalSrc];
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.fastMathEnabled = NO;
        c.library = [c.device newLibraryWithSource:src options:opts error:&err];
        if (c.library == nil) {
            c.why = "TURBORERANK_DEVICE_METAL: shader compile failed";
            if (err != nil) {
                c.why += " (";
                c.why += err.localizedDescription.UTF8String;
                c.why += ")";
            }
            c.why += ". Refusing CPU fallback.";
            return;
        }
        auto pipe = [&](const char *name, id<MTLComputePipelineState> *out) -> bool {
            id<MTLFunction> fn =
                [c.library newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (fn == nil) {
                c.why = std::string("TURBORERANK_DEVICE_METAL: missing kernel ") +
                        name + ". Refusing CPU fallback.";
                return false;
            }
            NSError *pe = nil;
            *out = [c.device newComputePipelineStateWithFunction:fn error:&pe];
            if (*out == nil) {
                c.why = std::string("TURBORERANK_DEVICE_METAL: pipeline ") + name +
                        " failed. Refusing CPU fallback.";
                return false;
            }
            return true;
        };
        if (!pipe("embed_kernel", &c.embed) || !pipe("layer_norm_kernel", &c.ln) ||
            !pipe("linear_nt_kernel", &c.linear) ||
            !pipe("gelu_erf_kernel", &c.gelu) ||
            !pipe("add_inplace_kernel", &c.add) ||
            !pipe("residual_from_ctx_kernel", &c.residual) ||
            !pipe("copy_f32_kernel", &c.copyf) ||
            !pipe("zero_f32_kernel", &c.zerof) ||
            !pipe("attention_scores_kernel", &c.scores) ||
            !pipe("softmax_rows_kernel", &c.softmax) ||
            !pipe("attention_ctx_kernel", &c.ctx) ||
            !pipe("pooler_kernel", &c.pooler) ||
            !pipe("classifier_kernel", &c.classifier)) {
            return;
        }
        c.ready = true;
    });
}

id<MTLBuffer> new_shared(id<MTLDevice> dev, size_t bytes, const void *src) {
    if (bytes == 0) {
        bytes = 4;
    }
    const size_t aligned = (bytes + 255u) & ~size_t(255u);
    id<MTLBuffer> buf = [dev newBufferWithLength:aligned
                                         options:MTLResourceStorageModeShared];
    if (buf == nil) {
        return nil;
    }
    std::memset(buf.contents, 0, aligned);
    if (src != nullptr && bytes > 0) {
        std::memcpy(buf.contents, src, bytes);
    }
    return buf;
}

struct MetalHold {
    id<MTLBuffer> word = nil;
    id<MTLBuffer> pos = nil;
    id<MTLBuffer> type = nil;
    id<MTLBuffer> emb_ln_w = nil;
    id<MTLBuffer> emb_ln_b = nil;
    id<MTLBuffer> q_w[12]{};
    id<MTLBuffer> q_b[12]{};
    id<MTLBuffer> k_w[12]{};
    id<MTLBuffer> k_b[12]{};
    id<MTLBuffer> v_w[12]{};
    id<MTLBuffer> v_b[12]{};
    id<MTLBuffer> attn_o_w[12]{};
    id<MTLBuffer> attn_o_b[12]{};
    id<MTLBuffer> attn_ln_w[12]{};
    id<MTLBuffer> attn_ln_b[12]{};
    id<MTLBuffer> ff_i_w[12]{};
    id<MTLBuffer> ff_i_b[12]{};
    id<MTLBuffer> ff_o_w[12]{};
    id<MTLBuffer> ff_o_b[12]{};
    id<MTLBuffer> ff_ln_w[12]{};
    id<MTLBuffer> ff_ln_b[12]{};
    id<MTLBuffer> pool_w = nil;
    id<MTLBuffer> pool_b = nil;
    id<MTLBuffer> cls_w = nil;
    id<MTLBuffer> cls_b = nil;
    id<MTLBuffer> x = nil;
    id<MTLBuffer> residual = nil;
    id<MTLBuffer> q = nil;
    id<MTLBuffer> k = nil;
    id<MTLBuffer> v = nil;
    id<MTLBuffer> attn = nil;
    id<MTLBuffer> ctxb = nil;
    id<MTLBuffer> inter = nil;
    id<MTLBuffer> pooled = nil;
    id<MTLBuffer> logit = nil;
    id<MTLBuffer> dummy_bias = nil;
    uint32_t word_rows = 0;
    uint32_t pos_rows = 0;
    uint32_t type_rows = 0;
    uint32_t n_layers = 0;
    uint32_t cls_cols = 0;
    bool has_pooler = false;
};

id<MTLBuffer> upload_view(id<MTLDevice> dev, const TensorView &tv) {
    const size_t n = static_cast<size_t>(tv.rows) * static_cast<size_t>(tv.cols);
    const size_t bytes = n * sizeof(float);
    return new_shared(dev, bytes, tv.data);
}

struct Encoder {
    id<MTLCommandBuffer> cmd = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    MetalCtx *c = nullptr;

    bool begin(MetalCtx *ctx_in, std::string *err) {
        c = ctx_in;
        cmd = [c->queue commandBuffer];
        if (cmd == nil) {
            if (err) {
                *err = "Metal command buffer alloc failed; refusing CPU fallback";
            }
            return false;
        }
        enc = [cmd computeCommandEncoder];
        if (enc == nil) {
            if (err) {
                *err = "Metal compute encoder failed; refusing CPU fallback";
            }
            return false;
        }
        return true;
    }

    void dispatch1(id<MTLComputePipelineState> pso, NSUInteger n) {
        if (n == 0) {
            return;
        }
        [enc setComputePipelineState:pso];
        const NSUInteger tw = std::min(n, pso.maxTotalThreadsPerThreadgroup);
        MTLSize grid = MTLSizeMake(n, 1, 1);
        MTLSize tg = MTLSizeMake(tw, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    void dispatch2(
        id<MTLComputePipelineState> pso, NSUInteger x, NSUInteger y
    ) {
        [enc setComputePipelineState:pso];
        const NSUInteger tw = std::min(x, pso.maxTotalThreadsPerThreadgroup);
        MTLSize grid = MTLSizeMake(x, y, 1);
        MTLSize tg = MTLSizeMake(tw, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    void dispatch3(
        id<MTLComputePipelineState> pso, NSUInteger x, NSUInteger y, NSUInteger z
    ) {
        [enc setComputePipelineState:pso];
        const NSUInteger tw = std::min(x, pso.maxTotalThreadsPerThreadgroup);
        MTLSize grid = MTLSizeMake(x, y, z);
        MTLSize tg = MTLSizeMake(tw, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    bool finish(std::string *err) {
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error != nil) {
            if (err) {
                *err = std::string("Metal command failed: ") +
                       cmd.error.localizedDescription.UTF8String +
                       "; refusing CPU fallback";
            }
            return false;
        }
        return true;
    }
};

struct BufView {
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
};

BufView lookup_view(const void *ptr) {
    BufView out;
    if (ptr == nullptr) {
        return out;
    }
    SharedRegistry &reg = registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    auto it = reg.map.find(ptr);
    if (it != reg.map.end()) {
        out.buffer = it->second.buffer;
        out.offset = 0;
        return out;
    }
    const char *p = static_cast<const char *>(ptr);
    for (const auto &kv : reg.map) {
        const char *start = static_cast<const char *>(kv.first);
        const size_t n = kv.second.bytes;
        if (p >= start && p < start + static_cast<ptrdiff_t>(n)) {
            out.buffer = kv.second.buffer;
            out.offset = static_cast<NSUInteger>(p - start);
            return out;
        }
    }
    return out;
}

} // namespace

bool metal_device_present(std::string *why) {
    init_ctx_once();
    if (ctx().ready) {
        return true;
    }
    if (why) {
        *why = ctx().why.empty()
                   ? "TURBORERANK_DEVICE_METAL requested but Metal is "
                     "unavailable. Refusing CPU fallback."
                   : ctx().why;
    }
    return false;
}

bool metal_gpu_name(std::string *name) {
    init_ctx_once();
    if (!ctx().ready || ctx().device == nil) {
        if (name) {
            *name = {};
        }
        return false;
    }
    if (name) {
        *name = ctx().device.name.UTF8String;
    }
    return true;
}

void *metal_shared_alloc_bytes(size_t bytes, Status *status) {
    if (bytes == 0) {
        if (status) {
            *status = Status::InvalidArgument;
        }
        return nullptr;
    }
    init_ctx_once();
    if (!ctx().ready) {
        if (status) {
            *status = Status::Unavailable;
        }
        return nullptr;
    }
    id<MTLBuffer> buf = new_shared(ctx().device, bytes, nullptr);
    if (buf == nil || buf.contents == nullptr) {
        if (status) {
            *status = Status::OutOfMemory;
        }
        return nullptr;
    }
    void *ptr = buf.contents;
    if ((reinterpret_cast<uintptr_t>(ptr) % 64u) != 0) {
        if (status) {
            *status = Status::Internal;
        }
        return nullptr;
    }
    {
        SharedRegistry &reg = registry();
        std::lock_guard<std::mutex> lock(reg.mu);
        reg.map[ptr] = SharedEntry{buf, bytes};
    }
    note_alloc();
    if (status) {
        *status = Status::Ok;
    }
    return ptr;
}

void metal_shared_free_bytes(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    SharedRegistry &reg = registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    reg.map.erase(ptr);
}

bool metal_shared_owns(const void *ptr) {
    SharedRegistry &reg = registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    return reg.map.find(ptr) != reg.map.end();
}

bool metal_resources_init(
    MetalResources *r,
    const BertConfig &cfg,
    const BertWeights &w,
    std::string *err
) {
    if (r == nullptr) {
        if (err) {
            *err = "metal_resources_init: null";
        }
        return false;
    }
    metal_resources_free(r);
    init_ctx_once();
    if (!ctx().ready) {
        if (err) {
            *err = ctx().why.empty()
                       ? "Metal MiniLM CE init failed; refusing CPU fallback"
                       : ctx().why;
        }
        return false;
    }
    id<MTLDevice> dev = ctx().device;
    auto *hold = new (std::nothrow) MetalHold();
    if (hold == nullptr) {
        if (err) {
            *err = "metal_resources_init: OOM";
        }
        return false;
    }
    auto fail = [&](const char *msg) -> bool {
        delete hold;
        if (err) {
            *err = std::string(msg) + "; refusing CPU fallback";
        }
        return false;
    };
    auto up = [&](const TensorView &tv, id<MTLBuffer> *dst, const char *name) -> bool {
        *dst = upload_view(dev, tv);
        if (*dst == nil) {
            fail(name);
            return false;
        }
        return true;
    };
    if (!up(w.word, &hold->word, "word") || !up(w.pos, &hold->pos, "pos") ||
        !up(w.type, &hold->type, "type") || !up(w.emb_ln_w, &hold->emb_ln_w, "emb_ln_w") ||
        !up(w.emb_ln_b, &hold->emb_ln_b, "emb_ln_b")) {
        return false;
    }
    hold->word_rows = w.word.rows;
    hold->pos_rows = w.pos.rows;
    hold->type_rows = w.type.rows;
    hold->n_layers = w.n_layers;
    hold->has_pooler = w.has_pooler;
    hold->cls_cols = w.cls_w.cols > 0 ? w.cls_w.cols : cfg.hidden;
    for (uint32_t i = 0; i < w.n_layers && i < 12; ++i) {
        if (!up(w.q_w[i], &hold->q_w[i], "q_w") || !up(w.q_b[i], &hold->q_b[i], "q_b") ||
            !up(w.k_w[i], &hold->k_w[i], "k_w") || !up(w.k_b[i], &hold->k_b[i], "k_b") ||
            !up(w.v_w[i], &hold->v_w[i], "v_w") || !up(w.v_b[i], &hold->v_b[i], "v_b") ||
            !up(w.attn_o_w[i], &hold->attn_o_w[i], "attn_o_w") ||
            !up(w.attn_o_b[i], &hold->attn_o_b[i], "attn_o_b") ||
            !up(w.attn_ln_w[i], &hold->attn_ln_w[i], "attn_ln_w") ||
            !up(w.attn_ln_b[i], &hold->attn_ln_b[i], "attn_ln_b") ||
            !up(w.ff_i_w[i], &hold->ff_i_w[i], "ff_i_w") ||
            !up(w.ff_i_b[i], &hold->ff_i_b[i], "ff_i_b") ||
            !up(w.ff_o_w[i], &hold->ff_o_w[i], "ff_o_w") ||
            !up(w.ff_o_b[i], &hold->ff_o_b[i], "ff_o_b") ||
            !up(w.ff_ln_w[i], &hold->ff_ln_w[i], "ff_ln_w") ||
            !up(w.ff_ln_b[i], &hold->ff_ln_b[i], "ff_ln_b")) {
            return false;
        }
    }
    if (w.has_pooler) {
        if (!up(w.pool_w, &hold->pool_w, "pool_w") ||
            !up(w.pool_b, &hold->pool_b, "pool_b")) {
            return false;
        }
    }
    if (!up(w.cls_w, &hold->cls_w, "cls_w") || !up(w.cls_b, &hold->cls_b, "cls_b")) {
        return false;
    }
    const uint32_t S = cfg.max_position;
    const uint32_t H = cfg.hidden;
    const uint32_t I = cfg.intermediate;
    const uint32_t heads = cfg.heads;
    auto scratch = [&](id<MTLBuffer> *p, size_t n_floats) -> bool {
        *p = new_shared(dev, n_floats * sizeof(float), nullptr);
        return *p != nil;
    };
    if (!scratch(&hold->x, static_cast<size_t>(S) * H) ||
        !scratch(&hold->residual, static_cast<size_t>(S) * H) ||
        !scratch(&hold->q, static_cast<size_t>(S) * H) ||
        !scratch(&hold->k, static_cast<size_t>(S) * H) ||
        !scratch(&hold->v, static_cast<size_t>(S) * H) ||
        !scratch(&hold->attn, static_cast<size_t>(heads) * S * S) ||
        !scratch(&hold->ctxb, static_cast<size_t>(S) * H) ||
        !scratch(&hold->inter, static_cast<size_t>(S) * I) ||
        !scratch(&hold->pooled, H) || !scratch(&hold->logit, 1)) {
        return fail("metal scratch alloc");
    }
    hold->dummy_bias = new_shared(dev, 4, nullptr);
    if (hold->dummy_bias == nil) {
        return fail("metal dummy bias");
    }
    r->hold = hold;
    r->max_seq = S;
    r->hidden = H;
    r->enabled = true;
    return true;
}

void metal_resources_free(MetalResources *r) {
    if (r == nullptr) {
        return;
    }
    if (r->hold != nullptr) {
        delete static_cast<MetalHold *>(r->hold);
        r->hold = nullptr;
    }
    r->enabled = false;
    r->max_seq = 0;
    r->hidden = 0;
}

bool bert_forward_row_metal(
    MetalResources *r,
    const BertConfig &cfg,
    const int32_t *input_ids,
    const int32_t *attention_mask,
    const int32_t *token_type_ids,
    const int32_t *position_ids,
    uint32_t seq,
    float *logit_out,
    std::string *err
) {
    if (r == nullptr || !r->enabled || r->hold == nullptr) {
        if (err) {
            *err = "Metal MiniLM CE is not initialized; refusing CPU fallback";
        }
        return false;
    }
    if (input_ids == nullptr || attention_mask == nullptr || logit_out == nullptr ||
        seq == 0 || seq > cfg.max_position) {
        if (err) {
            *err = "bert_forward_row_metal: bad arguments";
        }
        return false;
    }
    const BufView ids_v = lookup_view(input_ids);
    const BufView mask_v = lookup_view(attention_mask);
    const BufView type_v = lookup_view(token_type_ids);
    const BufView pos_v = lookup_view(position_ids);
    if (ids_v.buffer == nil || mask_v.buffer == nil || type_v.buffer == nil ||
        pos_v.buffer == nil) {
        if (err) {
            *err = "bert_forward_row_metal: token pointers are not "
                   "MTLResourceStorageModeShared (caller must allocate with "
                   "TURBORERANK_DEVICE_METAL). Refusing a CPU/std::vector copy.";
        }
        return false;
    }

    init_ctx_once();
    if (!ctx().ready) {
        if (err) {
            *err = ctx().why;
        }
        return false;
    }
    auto *hold = static_cast<MetalHold *>(r->hold);
    MetalCtx &c = ctx();
    const uint32_t H = cfg.hidden;
    const uint32_t I = cfg.intermediate;
    const uint32_t heads = cfg.heads;
    const uint32_t dh = H / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    const uint32_t hidden_n = seq * H;
    const uint32_t inter_n = seq * I;

    Encoder e;
    if (!e.begin(&c, err)) {
        return false;
    }

    struct EmbedParams {
        uint32_t seq, hidden, word_rows, pos_rows, type_rows;
    } ep{seq, H, hold->word_rows, hold->pos_rows, hold->type_rows};
    [e.enc setComputePipelineState:c.embed];
    [e.enc setBuffer:hold->x offset:0 atIndex:0];
    [e.enc setBuffer:ids_v.buffer offset:ids_v.offset atIndex:1];
    [e.enc setBuffer:pos_v.buffer offset:pos_v.offset atIndex:2];
    [e.enc setBuffer:type_v.buffer offset:type_v.offset atIndex:3];
    [e.enc setBuffer:hold->word offset:0 atIndex:4];
    [e.enc setBuffer:hold->pos offset:0 atIndex:5];
    [e.enc setBuffer:hold->type offset:0 atIndex:6];
    [e.enc setBytes:&ep length:sizeof(ep) atIndex:7];
    e.dispatch1(c.embed, seq);

    struct LnParams {
        uint32_t seq, hidden;
        float eps;
    } lp{seq, H, cfg.ln_eps};
    [e.enc setComputePipelineState:c.ln];
    [e.enc setBuffer:hold->x offset:0 atIndex:0];
    [e.enc setBuffer:hold->emb_ln_w offset:0 atIndex:1];
    [e.enc setBuffer:hold->emb_ln_b offset:0 atIndex:2];
    [e.enc setBytes:&lp length:sizeof(lp) atIndex:3];
    e.dispatch1(c.ln, seq);

    auto copy_f = [&](id<MTLBuffer> dst, id<MTLBuffer> src, uint32_t n) {
        struct ElemParams {
            uint32_t n;
        } p{n};
        [e.enc setComputePipelineState:c.copyf];
        [e.enc setBuffer:dst offset:0 atIndex:0];
        [e.enc setBuffer:src offset:0 atIndex:1];
        [e.enc setBytes:&p length:sizeof(p) atIndex:2];
        e.dispatch1(c.copyf, n);
    };
    auto zero_f = [&](id<MTLBuffer> dst, uint32_t n) {
        struct ElemParams {
            uint32_t n;
        } p{n};
        [e.enc setComputePipelineState:c.zerof];
        [e.enc setBuffer:dst offset:0 atIndex:0];
        [e.enc setBytes:&p length:sizeof(p) atIndex:1];
        e.dispatch1(c.zerof, n);
    };
    auto linear = [&](id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> b,
                      id<MTLBuffer> y, uint32_t k, uint32_t out) {
        struct LinearParams {
            uint32_t seq, k, out, has_bias;
        } p{seq, k, out, b != nil ? 1u : 0u};
        [e.enc setComputePipelineState:c.linear];
        [e.enc setBuffer:x offset:0 atIndex:0];
        [e.enc setBuffer:w offset:0 atIndex:1];
        [e.enc setBuffer:(b != nil ? b : hold->dummy_bias) offset:0 atIndex:2];
        [e.enc setBuffer:y offset:0 atIndex:3];
        [e.enc setBytes:&p length:sizeof(p) atIndex:4];
        e.dispatch2(c.linear, out, seq);
    };

    struct AttnParams {
        uint32_t seq, hidden, heads, dh;
        float scale;
    } ap{seq, H, heads, dh, scale};
    struct ElemParams {
        uint32_t n;
    };

    for (uint32_t layer = 0; layer < hold->n_layers; ++layer) {
        copy_f(hold->residual, hold->x, hidden_n);
        linear(hold->x, hold->q_w[layer], hold->q_b[layer], hold->q, H, H);
        linear(hold->x, hold->k_w[layer], hold->k_b[layer], hold->k, H, H);
        linear(hold->x, hold->v_w[layer], hold->v_b[layer], hold->v, H, H);
        zero_f(hold->ctxb, hidden_n);

        [e.enc setComputePipelineState:c.scores];
        [e.enc setBuffer:hold->q offset:0 atIndex:0];
        [e.enc setBuffer:hold->k offset:0 atIndex:1];
        [e.enc setBuffer:hold->attn offset:0 atIndex:2];
        [e.enc setBuffer:mask_v.buffer offset:mask_v.offset atIndex:3];
        [e.enc setBytes:&ap length:sizeof(ap) atIndex:4];
        e.dispatch3(c.scores, seq, seq, heads);

        [e.enc setComputePipelineState:c.softmax];
        [e.enc setBuffer:hold->attn offset:0 atIndex:0];
        [e.enc setBytes:&ap length:sizeof(ap) atIndex:1];
        e.dispatch2(c.softmax, seq, heads);

        [e.enc setComputePipelineState:c.ctx];
        [e.enc setBuffer:hold->attn offset:0 atIndex:0];
        [e.enc setBuffer:hold->v offset:0 atIndex:1];
        [e.enc setBuffer:hold->ctxb offset:0 atIndex:2];
        [e.enc setBytes:&ap length:sizeof(ap) atIndex:3];
        e.dispatch2(c.ctx, seq, heads);

        linear(hold->ctxb, hold->attn_o_w[layer], hold->attn_o_b[layer], hold->q, H, H);
        copy_f(hold->ctxb, hold->q, hidden_n);

        ElemParams hp{hidden_n};
        [e.enc setComputePipelineState:c.residual];
        [e.enc setBuffer:hold->x offset:0 atIndex:0];
        [e.enc setBuffer:hold->ctxb offset:0 atIndex:1];
        [e.enc setBuffer:hold->residual offset:0 atIndex:2];
        [e.enc setBytes:&hp length:sizeof(hp) atIndex:3];
        e.dispatch1(c.residual, hidden_n);

        [e.enc setComputePipelineState:c.ln];
        [e.enc setBuffer:hold->x offset:0 atIndex:0];
        [e.enc setBuffer:hold->attn_ln_w[layer] offset:0 atIndex:1];
        [e.enc setBuffer:hold->attn_ln_b[layer] offset:0 atIndex:2];
        [e.enc setBytes:&lp length:sizeof(lp) atIndex:3];
        e.dispatch1(c.ln, seq);

        copy_f(hold->residual, hold->x, hidden_n);
        linear(hold->x, hold->ff_i_w[layer], hold->ff_i_b[layer], hold->inter, H, I);
        ElemParams ip{inter_n};
        [e.enc setComputePipelineState:c.gelu];
        [e.enc setBuffer:hold->inter offset:0 atIndex:0];
        [e.enc setBytes:&ip length:sizeof(ip) atIndex:1];
        e.dispatch1(c.gelu, inter_n);
        linear(hold->inter, hold->ff_o_w[layer], hold->ff_o_b[layer], hold->x, I, H);

        [e.enc setComputePipelineState:c.add];
        [e.enc setBuffer:hold->x offset:0 atIndex:0];
        [e.enc setBuffer:hold->residual offset:0 atIndex:1];
        [e.enc setBytes:&hp length:sizeof(hp) atIndex:2];
        e.dispatch1(c.add, hidden_n);

        [e.enc setComputePipelineState:c.ln];
        [e.enc setBuffer:hold->x offset:0 atIndex:0];
        [e.enc setBuffer:hold->ff_ln_w[layer] offset:0 atIndex:1];
        [e.enc setBuffer:hold->ff_ln_b[layer] offset:0 atIndex:2];
        [e.enc setBytes:&lp length:sizeof(lp) atIndex:3];
        e.dispatch1(c.ln, seq);
    }

    id<MTLBuffer> head = hold->x;
    if (hold->has_pooler && hold->pool_w != nil) {
        struct PoolParams {
            uint32_t hidden;
        } pp{H};
        [e.enc setComputePipelineState:c.pooler];
        [e.enc setBuffer:hold->x offset:0 atIndex:0];
        [e.enc setBuffer:hold->pool_w offset:0 atIndex:1];
        [e.enc setBuffer:hold->pool_b offset:0 atIndex:2];
        [e.enc setBuffer:hold->pooled offset:0 atIndex:3];
        [e.enc setBytes:&pp length:sizeof(pp) atIndex:4];
        e.dispatch1(c.pooler, H);
        head = hold->pooled;
    }
    struct PoolParams {
        uint32_t hidden;
    } cp{hold->cls_cols > 0 ? hold->cls_cols : H};
    [e.enc setComputePipelineState:c.classifier];
    [e.enc setBuffer:head offset:0 atIndex:0];
    [e.enc setBuffer:hold->cls_w offset:0 atIndex:1];
    [e.enc setBuffer:hold->cls_b offset:0 atIndex:2];
    [e.enc setBuffer:hold->logit offset:0 atIndex:3];
    [e.enc setBytes:&cp length:sizeof(cp) atIndex:4];
    e.dispatch1(c.classifier, 1);

    if (!e.finish(err)) {
        return false;
    }
    std::memcpy(logit_out, hold->logit.contents, sizeof(float));
    return true;
}

} // namespace impl
} // namespace turborerank

#endif // TURBORERANK_METAL
