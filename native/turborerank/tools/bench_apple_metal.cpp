// SPDX-License-Identifier: Apache-2.0
//
// Machine C FINAL SOLIDIFY bench — TurboRerank Metal slice.
// Measures p50/p99 after warmup. Gates: allocs/forward==0, Berlin in
// band, turbo_buffer Metal SHARED (no CPU fallback). Writes JSON for
// the combined receipt writer (crates/turboembed/tests/apple_solidify_bench.rs).
// Never invents timings. Never embeds lab hostnames.

#include "metal_api.hpp"
#include "reranker.hpp"
#include "turbo_buffer.h"
#include "turborerank.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static uint64_t now_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<uint64_t>(ts.tv_nsec);
}

static uint64_t percentile_us(std::vector<uint64_t> *ns, unsigned p) {
    if (ns->empty()) {
        return 0;
    }
    std::sort(ns->begin(), ns->end());
    const size_t idx = (ns->size() - 1) * static_cast<size_t>(p) / 100;
    return (*ns)[idx] / 1000ull;
}

static uint64_t mean_us(const std::vector<uint64_t> &ns) {
    if (ns.empty()) {
        return 0;
    }
    uint64_t sum = 0;
    for (uint64_t v : ns) {
        sum += v;
    }
    return (sum / static_cast<uint64_t>(ns.size())) / 1000ull;
}

static unsigned env_u(const char *key, unsigned fallback) {
    const char *v = std::getenv(key);
    if (v == nullptr || !v[0]) {
        return fallback;
    }
    char *end = nullptr;
    const unsigned long n = std::strtoul(v, &end, 10);
    if (end == v || n == 0 || n > 100000) {
        return fallback;
    }
    return static_cast<unsigned>(n);
}

static std::string run_cmd(const char *cmd) {
    FILE *p = popen(cmd, "r");
    if (p == nullptr) {
        return {};
    }
    char buf[256];
    std::string out;
    while (fgets(buf, sizeof(buf), p) != nullptr) {
        out += buf;
    }
    pclose(p);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return out;
}

int main() {
    std::string why;
    if (!turborerank::impl::metal_device_present(&why)) {
        std::fprintf(stderr, "bench_apple_metal: Metal missing: %s\n", why.c_str());
        return 1;
    }
    const char *root_c = std::getenv("INFERSTREAM_ROOT");
    const std::string root = root_c && root_c[0] ? root_c : ".";
    const std::string model = root + "/models/rerank/ms-marco-minilm-l6";
    const char *out_env = std::getenv("BENCH_RERANK_JSON");
    const std::string out_path = out_env && out_env[0]
        ? std::string(out_env)
        : (root + "/native/turborerank/build/machine-c-rerank-bench.json");

    const unsigned warmup = env_u("BENCH_WARMUP", 32);
    const unsigned iters = env_u("BENCH_ITERS", 200);

    turborerank_engine *auto_e = nullptr;
    turborerank_status st =
        turborerank_engine_create(TURBORERANK_DEVICE_AUTO, model.c_str(), &auto_e);
    if (st != TURBORERANK_OK || auto_e == nullptr ||
        auto_e->device != TURBORERANK_DEVICE_METAL) {
        std::fprintf(
            stderr,
            "AUTO must resolve to METAL (st=%s device=%s err=%s) — no CPU fallback\n",
            turborerank_status_name(st),
            auto_e ? turborerank_device_name(auto_e->device) : "null",
            turborerank_last_error(auto_e)
        );
        if (auto_e != nullptr) {
            turborerank_engine_destroy(auto_e);
        }
        return 1;
    }
    turborerank_engine_destroy(auto_e);

    turborerank_engine *e = nullptr;
    st = turborerank_engine_create(TURBORERANK_DEVICE_METAL, model.c_str(), &e);
    if (st != TURBORERANK_OK || e == nullptr ||
        e->device != TURBORERANK_DEVICE_METAL) {
        std::fprintf(
            stderr,
            "create METAL failed: st=%s device=%s err=%s\n",
            turborerank_status_name(st),
            e ? turborerank_device_name(e->device) : "null",
            turborerank_last_error(e)
        );
        if (e != nullptr) {
            turborerank_engine_destroy(e);
        }
        return 1;
    }
    st = turborerank_load_model(e, "ms-marco-minilm-l6", 0);
    if (st != TURBORERANK_OK) {
        std::fprintf(stderr, "load failed: %s\n", turborerank_last_error(e));
        turborerank_engine_destroy(e);
        return 1;
    }

    const char *q = "How many people live in Berlin?";
    const char *rel =
        "Berlin has a population of 3,520,031 registered inhabitants in an "
        "area of 891.82 square kilometers.";
    const char *mid = "Berlin is well known for its museums.";
    const char *irrel = "New York City is famous for its pizza and bagels.";
    turborerank_str query{q, std::strlen(q)};
    turborerank_str docs[3] = {
        {rel, std::strlen(rel)},
        {mid, std::strlen(mid)},
        {irrel, std::strlen(irrel)},
    };
    turborerank_score_options opts{};
    opts.truncation = TURBORERANK_TRUNC_LONGEST_FIRST;
    opts.activation = TURBORERANK_ACT_IDENTITY;
    opts.max_length = 512;

    const bool arena_owns =
        e->arena != nullptr && e->work != nullptr &&
        turbo_buffer_arena_owns(e->arena, e->work->input_ids);
    const bool metal_owns =
        e->work != nullptr && turbo_buffer_metal_owns(e->work->input_ids);
    void *mtl = nullptr;
    size_t mtl_off = 0;
    const bool metal_lookup = e->work != nullptr &&
        turbo_buffer_metal_lookup(e->work->input_ids, &mtl, &mtl_off) == 1 &&
        mtl != nullptr;
    if (!arena_owns || !metal_owns || !metal_lookup) {
        std::fprintf(
            stderr,
            "FAKE: tokens are not turbo_buffer Metal SHARED "
            "(arena_owns=%d metal_owns=%d lookup=%d)\n",
            static_cast<int>(arena_owns),
            static_cast<int>(metal_owns),
            static_cast<int>(metal_lookup)
        );
        turborerank_engine_destroy(e);
        return 1;
    }

    float logits[3] = {0, 0, 0};
    for (unsigned i = 0; i < warmup; ++i) {
        st = turborerank_score(e, nullptr, 0, query, docs, 3, &opts, logits);
        if (st != TURBORERANK_OK) {
            std::fprintf(stderr, "warmup score failed: %s\n", turborerank_last_error(e));
            turborerank_engine_destroy(e);
            return 1;
        }
    }

    turborerank::alloc_counter_reset();
    float again[3] = {0, 0, 0};
    st = turborerank_score(e, nullptr, 0, query, docs, 3, &opts, again);
    const uint64_t allocs_after = turbo_buffer_alloc_counter();
    if (st != TURBORERANK_OK || allocs_after != 0) {
        std::fprintf(
            stderr,
            "steady-state score allocs=%llu st=%s err=%s\n",
            static_cast<unsigned long long>(allocs_after),
            turborerank_status_name(st),
            turborerank_last_error(e)
        );
        turborerank_engine_destroy(e);
        return 1;
    }

    std::vector<uint64_t> samples;
    samples.reserve(iters);
    turborerank::alloc_counter_reset();
    for (unsigned i = 0; i < iters; ++i) {
        const uint64_t t0 = now_ns();
        st = turborerank_score(e, nullptr, 0, query, docs, 3, &opts, again);
        const uint64_t t1 = now_ns();
        if (st != TURBORERANK_OK) {
            std::fprintf(stderr, "timed score failed: %s\n", turborerank_last_error(e));
            turborerank_engine_destroy(e);
            return 1;
        }
        samples.push_back(t1 - t0);
    }
    const uint64_t allocs_after_loop = turbo_buffer_alloc_counter();
    if (allocs_after_loop != 0) {
        std::fprintf(
            stderr,
            "FAKE: allocs/forward grew during timed loop: %llu\n",
            static_cast<unsigned long long>(allocs_after_loop)
        );
        turborerank_engine_destroy(e);
        return 1;
    }

    const float gold[3] = {8.84585285f, -4.32007599f, -11.27389431f};
    float max_abs = 0.0f;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < 3; ++i) {
        const float ae = std::fabs(logits[i] - gold[i]);
        if (ae > max_abs) {
            max_abs = ae;
        }
        dot += logits[i] * gold[i];
        na += logits[i] * logits[i];
        nb += gold[i] * gold[i];
    }
    const float cosine = dot / (std::sqrt(na) * std::sqrt(nb));
    const bool berlin = max_abs < 2e-3f && cosine > 0.999f &&
                        logits[0] > logits[1] && logits[1] > logits[2];
    const uint64_t p50 = percentile_us(&samples, 50);
    const uint64_t p99 = percentile_us(&samples, 99);
    const uint64_t mean = mean_us(samples);
    if (p50 == 0 || p99 < p50) {
        std::fprintf(
            stderr,
            "FAKE: p50/p99 not measured honestly (p50=%llu p99=%llu iters=%u)\n",
            static_cast<unsigned long long>(p50),
            static_cast<unsigned long long>(p99),
            iters
        );
        turborerank_engine_destroy(e);
        return 1;
    }
    const bool pass = berlin && allocs_after == 0 && metal_owns && arena_owns &&
                      metal_lookup && p50 > 0 && p99 >= p50;

    std::string gpu;
    turborerank::impl::metal_gpu_name(&gpu);
    const std::string sha = run_cmd("git rev-parse HEAD");

    std::ostringstream js;
    js.setf(std::ios::fixed);
    js << "{\n";
    js << "  \"alias\": \"ms-marco-minilm-l6\",\n";
    js << "  \"device\": \"METAL\",\n";
    js << "  \"memory_path\": \"SHARED\",\n";
    js << "  \"cpu_fallback\": false,\n";
    js << "  \"auto_resolves_metal\": true,\n";
    js << "  \"metal_owns_tokens\": " << (metal_owns ? "true" : "false") << ",\n";
    js << "  \"arena_owns_tokens\": " << (arena_owns ? "true" : "false") << ",\n";
    js << "  \"metal_lookup\": " << (metal_lookup ? "true" : "false") << ",\n";
    js << "  \"gpu\": \"" << gpu << "\",\n";
    js << "  \"warmup\": " << warmup << ",\n";
    js << "  \"iters\": " << iters << ",\n";
    js << "  \"p50_us\": " << p50 << ",\n";
    js << "  \"p99_us\": " << p99 << ",\n";
    js << "  \"mean_us\": " << mean << ",\n";
    js << "  \"allocs_per_forward\": 0,\n";
    js.precision(8);
    js << "  \"logits\": [" << logits[0] << ", " << logits[1] << ", " << logits[2]
       << "],\n";
    js.precision(6);
    js << "  \"max_abs_logit_err\": " << max_abs << ",\n";
    js.precision(10);
    js << "  \"cosine_vs_golden\": " << cosine << ",\n";
    js << "  \"berlin_in_band\": " << (berlin ? "true" : "false") << ",\n";
    js << "  \"git_sha\": \"" << sha << "\",\n";
    js << "  \"pass\": " << (pass ? "true" : "false") << "\n";
    js << "}\n";

    {
        const auto slash = out_path.rfind('/');
        if (slash != std::string::npos) {
            const std::string dir = out_path.substr(0, slash);
            std::string mkdir = "mkdir -p \"" + dir + "\"";
            if (std::system(mkdir.c_str()) != 0) {
                std::fprintf(stderr, "cannot mkdir for %s\n", out_path.c_str());
                turborerank_engine_destroy(e);
                return 1;
            }
        }
    }
    std::ofstream out(out_path);
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
        turborerank_engine_destroy(e);
        return 1;
    }
    out << js.str();
    turborerank_engine_destroy(e);
    std::fprintf(
        stderr,
        "wrote %s pass=%s p50_us=%llu p99_us=%llu mean_us=%llu "
        "max_abs=%.6g cosine=%.10f logits=%.8f %.8f %.8f allocs=0 SHARED\n",
        out_path.c_str(),
        pass ? "true" : "false",
        static_cast<unsigned long long>(p50),
        static_cast<unsigned long long>(p99),
        static_cast<unsigned long long>(mean),
        max_abs,
        cosine,
        logits[0],
        logits[1],
        logits[2]
    );
    return pass ? 0 : 2;
}
