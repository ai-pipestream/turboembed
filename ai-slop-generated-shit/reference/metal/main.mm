// SPDX-License-Identifier: Apache-2.0
//
// The direct-native half of the metal provider's matched benchmark
// (PLAN.md section 11). Metal has no vendor runtime to call: the native
// side is the same Metal Shading Language kernels (providers/metal/src/
// kernels.inc, from the proof of concept) driven by a plain Objective-C++
// program with no libturbo, no provider vtable and no session machinery
// in the timed path. It runs the token rows `turbo-bench embed
// --dump-tokens` wrote and writes a receipt of kind `native` that
// `turbo-bench compare` reads.
//
// The forward pass mirrors the provider's encode (embed, LN, per layer
// Q/K/V, attention, output, LN, FFN, LN) and then pools and normalizes with
// the same kernels, so the comparison measures what the provider adds
// around the kernels: the ABI boundary, option validation, buffers and
// result bookkeeping.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "safetensors.hpp"
#include "turbo_provider_common.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;
using turbo_metal::SafeTensors;
using turbo_metal::Tensor;

static const char *kSrc =
#include "kernels.inc"
    ;

namespace {

[[noreturn]] void die(const std::string &m) {
    std::fprintf(stderr, "error: %s\n", m.c_str());
    std::exit(2);
}

struct Options {
    std::string tokens, out, commit;
    unsigned iters = 30, warmup = 5;
};

Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char *n) -> std::string {
            if (i + 1 >= argc) {
                die(std::string(n) + " needs a value");
            }
            return argv[++i];
        };
        if (a == "--tokens") {
            o.tokens = value("--tokens");
        } else if (a == "--out") {
            o.out = value("--out");
        } else if (a == "--commit") {
            o.commit = value("--commit");
        } else if (a == "--iters") {
            o.iters = static_cast<unsigned>(std::stoul(value("--iters")));
        } else if (a == "--warmup") {
            o.warmup = static_cast<unsigned>(std::stoul(value("--warmup")));
        } else {
            die("unknown flag " + a);
        }
    }
    if (o.tokens.empty() || o.commit.empty()) {
        die("--tokens and --commit are required");
    }
    if (o.iters == 0) {
        die("--iters must be at least 1");
    }
    return o;
}

std::string run_cmd(const char *cmd) {
    std::string out;
    FILE *p = popen(cmd, "r");
    if (p == nullptr) {
        die(std::string("cannot run ") + cmd);
    }
    char buf[256];
    while (fgets(buf, sizeof buf, p) != nullptr) {
        out += buf;
    }
    pclose(p);
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) {
        out.pop_back();
    }
    return out;
}

id<MTLDevice> g_device;

id<MTLBuffer> shared(uint64_t bytes, const void *src) {
    const size_t aligned = (static_cast<size_t>(bytes) + 255) & ~static_cast<size_t>(255);
    id<MTLBuffer> b = [g_device newBufferWithLength:aligned options:MTLResourceStorageModeShared];
    if (b == nil) {
        die("Metal buffer of " + std::to_string(bytes) + " bytes failed");
    }
    std::memset(b.contents, 0, aligned);
    if (src != nullptr) {
        std::memcpy(b.contents, src, static_cast<size_t>(bytes));
    }
    return b;
}

id<MTLBuffer> upload(const Tensor &t) { return shared(t.elements() * 4, t.data); }

struct Layer {
    id<MTLBuffer> q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, ln1_w, ln1_b, ff_i_w, ff_i_b, ff_o_w, ff_o_b, ln2_w, ln2_b;
};

struct Weights {
    uint32_t hidden = 0, inter = 0, layers = 0, heads = 0, word_rows = 0, pos_rows = 0, type_rows = 0;
    float eps = 1e-12f;
    id<MTLBuffer> word, pos, type, emb_ln_w, emb_ln_b, dummy;
    std::vector<Layer> layer;
};

Weights load_weights(const std::string &st_path, const std::string &cfg_path) {
    Weights w;
    json cfg;
    {
        std::ifstream in(cfg_path);
        if (!in.good()) {
            die("cannot open " + cfg_path);
        }
        in >> cfg;
    }
    w.hidden = cfg.value("hidden_size", 0u);
    w.inter = cfg.value("intermediate_size", 0u);
    w.layers = cfg.value("num_hidden_layers", 0u);
    w.heads = cfg.value("num_attention_heads", 0u);
    w.eps = cfg.value("layer_norm_eps", 1e-12);
    SafeTensors st;
    st.open(st_path);
    const Tensor word = st.get({"embeddings.word_embeddings.weight", "bert.embeddings.word_embeddings.weight"}, 2);
    const Tensor pos = st.get({"embeddings.position_embeddings.weight", "bert.embeddings.position_embeddings.weight"}, 2);
    const Tensor type = st.get({"embeddings.token_type_embeddings.weight", "bert.embeddings.token_type_embeddings.weight"}, 2);
    w.word_rows = static_cast<uint32_t>(word.rows());
    w.pos_rows = static_cast<uint32_t>(pos.rows());
    w.type_rows = static_cast<uint32_t>(type.rows());
    w.word = upload(word);
    w.pos = upload(pos);
    w.type = upload(type);
    w.emb_ln_w = upload(st.get({"embeddings.LayerNorm.weight", "bert.embeddings.LayerNorm.weight"}, 1));
    w.emb_ln_b = upload(st.get({"embeddings.LayerNorm.bias", "bert.embeddings.LayerNorm.bias"}, 1));
    for (uint32_t i = 0; i < w.layers; ++i) {
        const std::string p = "encoder.layer." + std::to_string(i) + ".", q = "bert." + p;
        auto get = [&](const char *suffix, size_t rank) {
            const std::string a = p + suffix, b = q + suffix;
            return upload(st.get({a.c_str(), b.c_str()}, rank));
        };
        Layer L;
        L.q_w = get("attention.self.query.weight", 2);
        L.q_b = get("attention.self.query.bias", 1);
        L.k_w = get("attention.self.key.weight", 2);
        L.k_b = get("attention.self.key.bias", 1);
        L.v_w = get("attention.self.value.weight", 2);
        L.v_b = get("attention.self.value.bias", 1);
        L.o_w = get("attention.output.dense.weight", 2);
        L.o_b = get("attention.output.dense.bias", 1);
        L.ln1_w = get("attention.output.LayerNorm.weight", 1);
        L.ln1_b = get("attention.output.LayerNorm.bias", 1);
        L.ff_i_w = get("intermediate.dense.weight", 2);
        L.ff_i_b = get("intermediate.dense.bias", 1);
        L.ff_o_w = get("output.dense.weight", 2);
        L.ff_o_b = get("output.dense.bias", 1);
        L.ln2_w = get("output.LayerNorm.weight", 1);
        L.ln2_b = get("output.LayerNorm.bias", 1);
        w.layer.push_back(L);
    }
    w.dummy = shared(4, nullptr);
    return w;
}

struct Pipes {
    id<MTLComputePipelineState> embed, ln, linear, gelu, add, residual, copyf, scores, softmax, ctx, pool, l2, linear_simd, add_bias;
};

id<MTLComputePipelineState> pipe(id<MTLLibrary> lib, const char *name) {
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (fn == nil) {
        die(std::string("kernel ") + name + " missing");
    }
    NSError *err = nil;
    id<MTLComputePipelineState> p = [g_device newComputePipelineStateWithFunction:fn error:&err];
    if (p == nil) {
        die(std::string("pipeline ") + name + ": " + (err ? err.localizedDescription.UTF8String : "?"));
    }
    return p;
}

void dispatch(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, NSUInteger x, NSUInteger y = 1, NSUInteger z = 1) {
    [e setComputePipelineState:p];
    const NSUInteger tw = std::min<NSUInteger>(x, p.maxTotalThreadsPerThreadgroup);
    [e dispatchThreads:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
}

json latency(std::vector<double> &s, double rows, double tokens) {
    std::sort(s.begin(), s.end());
    const size_t n = s.size();
    auto pct = [&](double p) { return s[std::min<size_t>(n - 1, static_cast<size_t>(std::llround((n - 1) * p)))]; };
    double total = 0;
    for (double v : s) {
        total += v;
    }
    const double mean = total / static_cast<double>(n);
    json l;
    l["p50_ms"] = pct(0.5);
    if (n >= 100) {
        l["p99_ms"] = pct(0.99);
    }
    l["mean_ms"] = mean;
    l["min_ms"] = s.front();
    l["max_ms"] = s.back();
    l["rows_per_s"] = rows / (mean / 1e3);
    l["tokens_per_s"] = tokens / (mean / 1e3);
    l["iters"] = n;
    return l;
}

} // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        const Options o = parse(argc, argv);
        std::ifstream in(o.tokens);
        if (!in.good()) {
            die("cannot open " + o.tokens);
        }
        json dump;
        in >> dump;
        if (!dump["artifacts"].contains("safetensors") || !dump["artifacts"].contains("hf_config")) {
            die("the token dump names no safetensors and hf_config artifacts");
        }
        const std::string pooling = dump.value("pooling", "");
        const uint32_t mode = pooling == "mean" ? 1u : pooling == "cls" ? 2u : pooling == "last" ? 3u : 0u;
        if (mode == 0) {
            die("pooling `" + pooling + "` is not mean, cls or last");
        }
        const bool normalize = dump.value("normalize", "") == "l2";

        g_device = MTLCreateSystemDefaultDevice();
        if (g_device == nil) {
            die("no Metal device");
        }
        NSError *err = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [g_device newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:opts error:&err];
        if (lib == nil) {
            die(std::string("shader compile: ") + (err ? err.localizedDescription.UTF8String : "?"));
        }
        Pipes P;
        P.embed = pipe(lib, "embed_kernel");
        P.ln = pipe(lib, "layer_norm_kernel");
        P.linear = pipe(lib, "linear_nt_kernel");
        if ([g_device supportsFamily:MTLGPUFamilyApple7]) {
            P.linear_simd = pipe(lib, "linear_nt_simd_kernel");
            P.add_bias = pipe(lib, "add_bias_rows_kernel");
        }
        P.gelu = pipe(lib, "gelu_erf_kernel");
        P.add = pipe(lib, "add_inplace_kernel");
        P.residual = pipe(lib, "residual_from_ctx_kernel");
        P.copyf = pipe(lib, "copy_f32_kernel");
        P.scores = pipe(lib, "attention_scores_kernel");
        P.softmax = pipe(lib, "softmax_rows_kernel");
        P.ctx = pipe(lib, "attention_ctx_kernel");
        P.pool = pipe(lib, "sentence_pool_kernel");
        P.l2 = pipe(lib, "l2_normalize_kernel");
        const Weights W = load_weights(dump["artifacts"]["safetensors"], dump["artifacts"]["hf_config"]);
        id<MTLCommandQueue> queue = [g_device newCommandQueue];

        json cells = json::array();
        for (const auto &cell : dump["cells"]) {
            const uint32_t b = cell["batch"].get<uint32_t>(), s = cell["seq"].get<uint32_t>();
            const std::vector<int32_t> ids = cell["ids"].get<std::vector<int32_t>>();
            const std::vector<int32_t> mask = cell["mask"].get<std::vector<int32_t>>();
            if (ids.size() != static_cast<size_t>(b) * s || mask.size() != ids.size()) {
                die("cell ids/mask do not hold batch*seq elements");
            }
            double live = 0;
            for (const auto &l : cell["lengths"]) {
                live += l.get<double>();
            }
            const uint32_t N = b * s, H = W.hidden, I = W.inter, heads = W.heads, dh = H / heads;
            const uint32_t Npad = (N + 31) / 32 * 32; // the same padding the provider uses
            std::vector<int32_t> posv(N), typev(N, 0);
            for (uint32_t r = 0; r < b; ++r) {
                for (uint32_t t = 0; t < s; ++t) {
                    posv[r * s + t] = static_cast<int32_t>(t);
                }
            }
            id<MTLBuffer> bids = shared(N * 4, ids.data()), bmask = shared(N * 4, mask.data()), bpos = shared(N * 4, posv.data()),
                          btype = shared(N * 4, typev.data());
            id<MTLBuffer> x = shared(uint64_t(Npad) * H * 4, nullptr), res = shared(uint64_t(Npad) * H * 4, nullptr),
                          q = shared(uint64_t(Npad) * H * 4, nullptr), k = shared(uint64_t(Npad) * H * 4, nullptr),
                          v = shared(uint64_t(Npad) * H * 4, nullptr), attn = shared(uint64_t(b) * heads * s * s * 4, nullptr),
                          ctxb = shared(uint64_t(Npad) * H * 4, nullptr), inter = shared(uint64_t(Npad) * I * 4, nullptr),
                          out = shared(uint64_t(b) * H * 4, nullptr);
            std::vector<float> host_out(static_cast<size_t>(b) * H);
            auto run_once = [&]() {
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
                struct { uint32_t seq, hidden, word_rows, pos_rows, type_rows; } ep{N, H, W.word_rows, W.pos_rows, W.type_rows};
                [e setComputePipelineState:P.embed];
                [e setBuffer:x offset:0 atIndex:0];
                [e setBuffer:bids offset:0 atIndex:1];
                [e setBuffer:bpos offset:0 atIndex:2];
                [e setBuffer:btype offset:0 atIndex:3];
                [e setBuffer:W.word offset:0 atIndex:4];
                [e setBuffer:W.pos offset:0 atIndex:5];
                [e setBuffer:W.type offset:0 atIndex:6];
                [e setBytes:&ep length:sizeof(ep) atIndex:7];
                dispatch(e, P.embed, N);
                struct { uint32_t seq, hidden; float eps; } lp{N, H, W.eps};
                auto ln = [&](id<MTLBuffer> w, id<MTLBuffer> bb) {
                    [e setComputePipelineState:P.ln];
                    [e setBuffer:x offset:0 atIndex:0];
                    [e setBuffer:w offset:0 atIndex:1];
                    [e setBuffer:bb offset:0 atIndex:2];
                    [e setBytes:&lp length:sizeof(lp) atIndex:3];
                    dispatch(e, P.ln, N);
                };
                ln(W.emb_ln_w, W.emb_ln_b);
                auto elem = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> a, id<MTLBuffer> b2, id<MTLBuffer> c2, uint32_t n) {
                    struct { uint32_t n; } p{n};
                    [e setComputePipelineState:pso];
                    [e setBuffer:a offset:0 atIndex:0];
                    uint32_t idx = 1;
                    if (b2 != nil) {
                        [e setBuffer:b2 offset:0 atIndex:idx++];
                    }
                    if (c2 != nil) {
                        [e setBuffer:c2 offset:0 atIndex:idx++];
                    }
                    [e setBytes:&p length:sizeof(p) atIndex:idx];
                    dispatch(e, pso, n);
                };
                auto linear = [&](id<MTLBuffer> in_, id<MTLBuffer> w, id<MTLBuffer> bias, id<MTLBuffer> y, uint32_t kdim, uint32_t outdim) {
                    struct { uint32_t seq, k, out, has_bias; } p{N, kdim, outdim, 1u};
                    if (P.linear_simd != nil && outdim % 32 == 0 && kdim % 8 == 0) {
                        struct { uint32_t seq, k, out, has_bias; } ps{Npad, kdim, outdim, 0u};
                        [e setComputePipelineState:P.linear_simd];
                        [e setBuffer:in_ offset:0 atIndex:0];
                        [e setBuffer:w offset:0 atIndex:1];
                        [e setBuffer:y offset:0 atIndex:2];
                        [e setBytes:&ps length:sizeof(ps) atIndex:3];
                        [e dispatchThreadgroups:MTLSizeMake(outdim / 32, Npad / 32, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [e setComputePipelineState:P.add_bias];
                        [e setBuffer:y offset:0 atIndex:0];
                        [e setBuffer:bias offset:0 atIndex:1];
                        [e setBytes:&p length:sizeof(p) atIndex:2];
                        dispatch(e, P.add_bias, outdim, N);
                        return;
                    }
                    [e setComputePipelineState:P.linear];
                    [e setBuffer:in_ offset:0 atIndex:0];
                    [e setBuffer:w offset:0 atIndex:1];
                    [e setBuffer:bias offset:0 atIndex:2];
                    [e setBuffer:y offset:0 atIndex:3];
                    [e setBytes:&p length:sizeof(p) atIndex:4];
                    [e dispatchThreadgroups:MTLSizeMake((N + 15) / 16, (outdim + 15) / 16, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                };
                struct { uint32_t seq, hidden, heads, dh; float scale; uint32_t batch; } ap{s, H, heads, dh, 1.0f / std::sqrt(static_cast<float>(dh)), b};
                for (const Layer &L : W.layer) {
                    elem(P.copyf, res, x, nil, N * H);
                    linear(x, L.q_w, L.q_b, q, H, H);
                    linear(x, L.k_w, L.k_b, k, H, H);
                    linear(x, L.v_w, L.v_b, v, H, H);
                    [e setComputePipelineState:P.scores];
                    [e setBuffer:q offset:0 atIndex:0];
                    [e setBuffer:k offset:0 atIndex:1];
                    [e setBuffer:attn offset:0 atIndex:2];
                    [e setBuffer:bmask offset:0 atIndex:3];
                    [e setBytes:&ap length:sizeof(ap) atIndex:4];
                    dispatch(e, P.scores, s, s, static_cast<NSUInteger>(b) * heads);
                    [e setComputePipelineState:P.softmax];
                    [e setBuffer:attn offset:0 atIndex:0];
                    [e setBytes:&ap length:sizeof(ap) atIndex:1];
                    dispatch(e, P.softmax, s, static_cast<NSUInteger>(b) * heads);
                    [e setComputePipelineState:P.ctx];
                    [e setBuffer:attn offset:0 atIndex:0];
                    [e setBuffer:v offset:0 atIndex:1];
                    [e setBuffer:ctxb offset:0 atIndex:2];
                    [e setBytes:&ap length:sizeof(ap) atIndex:3];
                    dispatch(e, P.ctx, s, H, b);
                    linear(ctxb, L.o_w, L.o_b, q, H, H);
                    elem(P.residual, x, q, res, N * H);
                    ln(L.ln1_w, L.ln1_b);
                    elem(P.copyf, res, x, nil, N * H);
                    linear(x, L.ff_i_w, L.ff_i_b, inter, H, I);
                    elem(P.gelu, inter, nil, nil, N * I);
                    linear(inter, L.ff_o_w, L.ff_o_b, x, I, H);
                    elem(P.add, x, res, nil, N * H);
                    ln(L.ln2_w, L.ln2_b);
                }
                struct { uint32_t seq, hidden, mode, batch; } sp{s, H, mode, b};
                [e setComputePipelineState:P.pool];
                [e setBuffer:x offset:0 atIndex:0];
                [e setBuffer:bmask offset:0 atIndex:1];
                [e setBuffer:out offset:0 atIndex:2];
                [e setBytes:&sp length:sizeof(sp) atIndex:3];
                dispatch(e, P.pool, H, b);
                if (normalize) {
                    struct { uint32_t n; } np{H};
                    [e setComputePipelineState:P.l2];
                    [e setBuffer:out offset:0 atIndex:0];
                    [e setBytes:&np length:sizeof(np) atIndex:1];
                    dispatch(e, P.l2, b);
                }
                [e endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
                if (cmd.error != nil) {
                    die(std::string("command buffer: ") + cmd.error.localizedDescription.UTF8String);
                }
                // The result is already host-visible (unified memory); a
                // user copies it out of the shared buffer.
                std::memcpy(host_out.data(), out.contents, host_out.size() * 4);
            };
            // At least o.warmup iterations and at least 0.5 s, as turbo-bench
            // warms up, so the GPU leaves its idle clock state first.
            const auto w0 = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < o.warmup || std::chrono::steady_clock::now() - w0 < std::chrono::milliseconds(500); ++i) {
                run_once();
            }
            std::vector<double> samples;
            for (unsigned i = 0; i < o.iters; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                run_once();
                samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            json lat = latency(samples, b, live);
            std::fprintf(stderr, "embed batch %2u seq %3u: tokens p50 %.3f ms (%.0f rows/s)\n", b, s, lat["p50_ms"].get<double>(),
                         lat["rows_per_s"].get<double>());
            json c;
            c["batch"] = b;
            c["seq"] = s;
            c["live_tokens_per_row"] = live / b;
            c["token_count_source"] = "tokenizer";
            c["text_path"] = lat;
            c["prepared_tokens_path"] = lat;
            c["per_run"] = {{"host_allocs", nullptr}, {"provider_allocs", nullptr}};
            cells.push_back(c);
        }
        json r;
        r["receipt_version"] = 1;
        r["kind"] = "native";
        {
            const std::time_t t = std::time(nullptr);
            std::tm tm{};
            gmtime_r(&t, &tm);
            char buf[16];
            std::strftime(buf, sizeof buf, "%Y-%m-%d", &tm);
            r["date"] = buf;
        }
        r["machine"] = {{"hostname", run_cmd("uname -n")}, {"os", run_cmd("uname -sr")}, {"arch", run_cmd("uname -m")}};
        r["commit"] = o.commit;
        r["provider"] = {{"id", "metal"}, {"version", "reference"}, {"runtime_version", "Metal, MSL compiled at runtime (the provider's kernels driven directly)"},
                         {"driver_version", run_cmd("sw_vers -productVersion")}};
        r["device"] = {{"name", std::string(g_device.name.UTF8String) + " (Metal)"}, {"kind", "IGpu"}, {"ordinal", 0}, {"caps", "0x0"},
                       {"memory_total", static_cast<uint64_t>(g_device.recommendedMaxWorkingSetSize)}};
        r["bundle"] = dump["bundle_id"];
        r["embed"] = cells;
        r["native_reference"] = "this is the native side: the metal provider's kernels driven by a plain Objective-C++ program on the token rows of " + o.tokens +
                                "; bundle identity copied from that dump; the same pooling and normalization kernels; no session counters";
        const std::string text = r.dump(2) + "\n";
        if (o.out.empty()) {
            std::printf("%s", text.c_str());
        } else {
            std::ofstream f(o.out);
            f << text;
            std::fprintf(stderr, "receipt written to %s\n", o.out.c_str());
        }
    }
    return 0;
}
