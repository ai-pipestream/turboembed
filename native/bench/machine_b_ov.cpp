// SPDX-License-Identifier: Apache-2.0
//
// FINAL SOLIDIFY bench — Machine B OpenVINO GPU.
// Measures TurboEmbed + TurboRerank. Writes testdata/receipts/bench/machine-b-ov.json.
// Numbers come from this process. Hostnames stay out of the receipt.

#include "genai.hpp"
#include "ov_api.hpp"
#include "turbo_buffer.h"
#include "turboembed.h"
#include "turborerank.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef TURBOEMBED_WORKSPACE_ROOT
#define TURBOEMBED_WORKSPACE_ROOT ""
#endif

extern "C" turboembed_status turboembed_test_genai_arena_info(
    const turboembed_engine *engine,
    uint32_t *arena_device,
    uint32_t *token_placement,
    const void **token_ids,
    const void **hidden,
    int *owns_tokens,
    int *owns_hidden,
    int *hidden_used_arena
);

namespace {

constexpr float kBerlinGold[3] = {8.84585285f, -4.32007599f, -11.27389431f};
constexpr float kBerlinAtol = 2e-3f;
constexpr float kBerlinCosineFloor = 0.999f;
constexpr float kEmbedCosineFloor = 0.99f;
constexpr const char *kHello = "hello world";
constexpr const char *kQuery = "How many people live in Berlin?";
constexpr const char *kRel =
    "Berlin has a population of 3,520,031 registered inhabitants in an "
    "area of 891.82 square kilometers.";
constexpr const char *kMid = "Berlin is well known for its museums.";
constexpr const char *kIrrel = "New York City is famous for its pizza and bagels.";

uint32_t env_u32(const char *name, uint32_t fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const unsigned long n = std::strtoul(v, &end, 10);
    if (end == v || n == 0 || n > 100000) {
        return fallback;
    }
    return static_cast<uint32_t>(n);
}

std::string root_dir() {
    if (const char *e = std::getenv("INFERSTREAM_ROOT")) {
        if (e[0]) {
            return e;
        }
    }
    if (TURBOEMBED_WORKSPACE_ROOT[0]) {
        return TURBOEMBED_WORKSPACE_ROOT;
    }
    return ".";
}

std::string run_cmd(const char *cmd) {
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

std::string utc_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm {};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string json_escape(const std::string &s) {
    std::string o;
    o.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '"':
            o += "\\\"";
            break;
        case '\\':
            o += "\\\\";
            break;
        case '\n':
            o += "\\n";
            break;
        case '\r':
            o += "\\r";
            break;
        default:
            o += c;
        }
    }
    return o;
}

bool load_first_vector(const std::string &path, std::vector<float> *out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::string key = "\"vector\"";
    const size_t k = text.find(key);
    if (k == std::string::npos) {
        return false;
    }
    const size_t lb = text.find('[', k);
    if (lb == std::string::npos) {
        return false;
    }
    size_t i = lb + 1;
    out->clear();
    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == '\n' || text[i] == '\r' ||
                                   text[i] == '\t' || text[i] == ',')) {
            ++i;
        }
        if (i < text.size() && text[i] == ']') {
            return !out->empty();
        }
        char *end = nullptr;
        const float v = std::strtof(text.c_str() + i, &end);
        if (end == text.c_str() + i) {
            return false;
        }
        out->push_back(v);
        i = static_cast<size_t>(end - text.c_str());
    }
    return false;
}

float cosine(const float *a, const float *b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(a[i]);
        const double y = static_cast<double>(b[i]);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if (na <= 0.0 || nb <= 0.0) {
        return 0.0f;
    }
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

double now_ns() {
    timespec ts {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) * 1e9 + static_cast<double>(ts.tv_nsec);
}

double percentile_us(std::vector<double> ns, double p) {
    if (ns.empty()) {
        return 0.0;
    }
    std::sort(ns.begin(), ns.end());
    const double idx = (p / 100.0) * static_cast<double>(ns.size() - 1);
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min(lo + 1, ns.size() - 1);
    const double t = idx - static_cast<double>(lo);
    const double v = ns[lo] * (1.0 - t) + ns[hi] * t;
    return v / 1000.0;
}

struct Latency {
    uint32_t warmup = 0;
    uint32_t iters = 0;
    double p50_us = 0;
    double p99_us = 0;
    double min_us = 0;
    double max_us = 0;
};

Latency summarize(const std::vector<double> &samples_ns, uint32_t warmup, uint32_t iters) {
    Latency L;
    L.warmup = warmup;
    L.iters = iters;
    if (samples_ns.empty()) {
        return L;
    }
    L.p50_us = percentile_us(samples_ns, 50.0);
    L.p99_us = percentile_us(samples_ns, 99.0);
    auto mm = std::minmax_element(samples_ns.begin(), samples_ns.end());
    L.min_us = *mm.first / 1000.0;
    L.max_us = *mm.second / 1000.0;
    return L;
}

void write_latency(std::ostringstream &js, const Latency &L, const char *indent) {
    js.setf(std::ios::fixed);
    js.precision(3);
    js << indent << "\"warmup_iters\": " << L.warmup << ",\n";
    js << indent << "\"measure_iters\": " << L.iters << ",\n";
    js << indent << "\"p50_us\": " << L.p50_us << ",\n";
    js << indent << "\"p99_us\": " << L.p99_us << ",\n";
    js << indent << "\"min_us\": " << L.min_us << ",\n";
    js << indent << "\"max_us\": " << L.max_us << ",\n";
    js.precision(6);
    js << indent << "\"p50_ms\": " << (L.p50_us / 1000.0) << ",\n";
    js << indent << "\"p99_ms\": " << (L.p99_us / 1000.0) << "\n";
}

} // namespace

int main() {
    const uint32_t warmup = env_u32("BENCH_WARMUP", 32);
    const uint32_t iters = env_u32("BENCH_ITERS", 128);
    const std::string root = root_dir();

    std::string why;
    if (!turborerank::impl::ov_gpu_present(&why)) {
        std::fprintf(stderr, "bench-machine-b-ov: OpenVINO GPU missing: %s\n", why.c_str());
        return 1;
    }

    std::string gpu;
    turborerank::impl::ov_gpu_name(&gpu);
    const bool remote_wrap = turborerank::impl::ov_probe_remote_usm_wrap(&why);
    std::string wrap_why = why;

    turborerank_engine *re = nullptr;
    const std::string rerank_dir = root + "/models/ov-rerank/ms-marco-minilm-l6";
    if (turborerank_engine_create(TURBORERANK_DEVICE_OPENVINO_GPU, rerank_dir.c_str(), &re) !=
            TURBORERANK_OK ||
        re == nullptr) {
        std::fprintf(
            stderr, "turborerank create: %s\n", turborerank_last_error(nullptr)
        );
        return 1;
    }
    if (turborerank_load_model(re, "ms-marco-minilm-l6", 0) != TURBORERANK_OK) {
        std::fprintf(stderr, "turborerank load: %s\n", turborerank_last_error(re));
        turborerank_engine_destroy(re);
        return 1;
    }

    turborerank_str query{kQuery, std::strlen(kQuery)};
    turborerank_str docs[3] = {
        {kRel, std::strlen(kRel)},
        {kMid, std::strlen(kMid)},
        {kIrrel, std::strlen(kIrrel)},
    };
    turborerank_score_options opts {};
    opts.truncation = TURBORERANK_TRUNC_LONGEST_FIRST;
    opts.activation = TURBORERANK_ACT_IDENTITY;
    opts.max_length = 512;
    float logits[3] = {0, 0, 0};

    turborerank_buffer *packed = nullptr;
    uint32_t used_tokens[3] = {0, 0, 0};
    uint32_t pack_seq = 0;
    if (turborerank_buffer_alloc(TURBORERANK_DEVICE_OPENVINO_GPU, 3, 512, &packed) ==
            TURBORERANK_OK &&
        packed != nullptr) {
        pack_seq = packed->seq;
        for (uint32_t i = 0; i < 3; ++i) {
            if (turborerank_pack_text(
                    re, packed, i, query, docs[i], TURBORERANK_TRUNC_LONGEST_FIRST, 512
                ) != TURBORERANK_OK) {
                continue;
            }
            uint32_t used = packed->seq;
            const int32_t *mask =
                packed->attention_mask + static_cast<size_t>(i) * packed->row_stride;
            while (used > 0 && mask[used - 1] == 0) {
                --used;
            }
            used_tokens[i] = used;
        }
        turborerank_buffer_free(packed);
    }

    for (uint32_t i = 0; i < warmup; ++i) {
        if (turborerank_score(re, nullptr, 0, query, docs, 3, &opts, logits) !=
            TURBORERANK_OK) {
            std::fprintf(stderr, "turborerank warmup: %s\n", turborerank_last_error(re));
            turborerank_engine_destroy(re);
            return 1;
        }
    }

    turbo_buffer_alloc_counter_reset();
    turbo_buffer_ze_xfer_reset();
    turborerank::impl::ov_xfer_reset();
    if (turborerank_score(re, nullptr, 0, query, docs, 3, &opts, logits) != TURBORERANK_OK) {
        std::fprintf(stderr, "turborerank gate score: %s\n", turborerank_last_error(re));
        turborerank_engine_destroy(re);
        return 1;
    }
    const uint64_t rerank_allocs = turbo_buffer_alloc_counter();
    const uint64_t rerank_ze_h2d = turbo_buffer_ze_xfer_h2d_bytes();
    const uint64_t rerank_ze_d2h = turbo_buffer_ze_xfer_d2h_bytes();
    const uint64_t rerank_ze_h2d_calls = turbo_buffer_ze_xfer_h2d_calls();
    const uint64_t rerank_ze_d2h_calls = turbo_buffer_ze_xfer_d2h_calls();
    const uint64_t rerank_wrap_in = turborerank::impl::ov_xfer_wrap_input_bytes();
    const uint64_t rerank_wrap_out = turborerank::impl::ov_xfer_result_bytes();
    const uint32_t rerank_n = turborerank::impl::ov_xfer_last_n_rows();
    const uint32_t rerank_seq = turborerank::impl::ov_xfer_last_seq();

    float max_abs = 0.0f;
    for (int i = 0; i < 3; ++i) {
        max_abs = std::max(max_abs, std::fabs(logits[i] - kBerlinGold[i]));
    }
    const float berlin_cos = cosine(logits, kBerlinGold, 3);
    const bool berlin_order = logits[0] > logits[1] && logits[1] > logits[2];
    const bool berlin_ok =
        max_abs < kBerlinAtol && berlin_cos > kBerlinCosineFloor && berlin_order;

    turbo_buffer_ze_xfer_reset();
    turborerank::impl::ov_xfer_reset();
    std::vector<double> rerank_ns;
    rerank_ns.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        const double t0 = now_ns();
        if (turborerank_score(re, nullptr, 0, query, docs, 3, &opts, logits) !=
            TURBORERANK_OK) {
            std::fprintf(stderr, "turborerank measure: %s\n", turborerank_last_error(re));
            turborerank_engine_destroy(re);
            return 1;
        }
        rerank_ns.push_back(now_ns() - t0);
    }
    const Latency rerank_lat = summarize(rerank_ns, warmup, iters);
    const uint64_t rerank_wrap_in_loop = turborerank::impl::ov_xfer_wrap_input_bytes();
    const uint64_t rerank_ze_h2d_loop = turbo_buffer_ze_xfer_h2d_bytes();

    turboembed_engine *ee = nullptr;
    if (turboembed_engine_create(TURBOEMBED_DEVICE_OPENVINO_GPU, nullptr, &ee) !=
            TURBOEMBED_OK ||
        ee == nullptr) {
        std::fprintf(stderr, "turboembed create: %s\n", turboembed_last_error(nullptr));
        turborerank_engine_destroy(re);
        return 1;
    }
    if (turboembed_load_model(ee, "minilm", 0) != TURBOEMBED_OK) {
        std::fprintf(stderr, "turboembed load: %s\n", turboembed_last_error(ee));
        turboembed_engine_destroy(ee);
        turborerank_engine_destroy(re);
        return 1;
    }

    turboembed_embed_result *er = nullptr;
    for (uint32_t i = 0; i < warmup; ++i) {
        if (turboembed_embed_one(
                ee, "minilm", 0, kHello, std::strlen(kHello), nullptr, &er
            ) != TURBOEMBED_OK ||
            er == nullptr) {
            std::fprintf(stderr, "turboembed warmup: %s\n", turboembed_last_error(ee));
            turboembed_engine_destroy(ee);
            turborerank_engine_destroy(re);
            return 1;
        }
        turboembed_embed_result_free(er);
        er = nullptr;
    }

    turbo_buffer_alloc_counter_reset();
    turbo_buffer_ze_xfer_reset();
    turboembed_genai::genai_xfer_reset();
    if (turboembed_embed_one(ee, "minilm", 0, kHello, std::strlen(kHello), nullptr, &er) !=
            TURBOEMBED_OK ||
        er == nullptr) {
        std::fprintf(stderr, "turboembed gate: %s\n", turboembed_last_error(ee));
        turboembed_engine_destroy(ee);
        turborerank_engine_destroy(re);
        return 1;
    }
    const uint64_t embed_allocs = turbo_buffer_alloc_counter();
    const uint64_t embed_ze_h2d = turbo_buffer_ze_xfer_h2d_bytes();
    const uint64_t embed_ze_d2h = turbo_buffer_ze_xfer_d2h_bytes();
    const uint64_t embed_ze_h2d_calls = turbo_buffer_ze_xfer_h2d_calls();
    const uint64_t embed_ze_d2h_calls = turbo_buffer_ze_xfer_d2h_calls();
    const uint64_t embed_wrap_in = turboembed_genai::genai_xfer_wrap_input_bytes();
    const uint64_t embed_hidden_wrap = turboembed_genai::genai_xfer_hidden_wrap_bytes();
    const uint64_t embed_hidden_memcpy = turboembed_genai::genai_xfer_hidden_memcpy_bytes();
    const uint32_t embed_n = turboembed_genai::genai_xfer_last_n();
    const uint32_t embed_seq = turboembed_genai::genai_xfer_last_seq();
    const uint32_t embed_inputs = turboembed_genai::genai_xfer_last_inputs();
    const uint32_t embed_dim = er->dim;
    std::vector<float> live(er->values, er->values + er->dim);

    uint32_t arena_dev = 0;
    uint32_t token_place = 0;
    const void *token_ids = nullptr;
    const void *hidden = nullptr;
    int owns_tokens = 0;
    int owns_hidden = 0;
    int hidden_arena = 0;
    (void)turboembed_test_genai_arena_info(
        ee,
        &arena_dev,
        &token_place,
        &token_ids,
        &hidden,
        &owns_tokens,
        &owns_hidden,
        &hidden_arena
    );
    turbo_buffer_placement zq = TURBO_BUFFER_PLACE_HOST;
    const bool tokens_shared =
        token_ids != nullptr &&
        turbo_buffer_ze_query(token_ids, &zq) == TURBO_BUFFER_OK &&
        zq == TURBO_BUFFER_PLACE_SHARED;
    turboembed_embed_result_free(er);
    er = nullptr;

    std::vector<float> intel_gold;
    std::vector<float> nvidia_gold;
    const bool intel_ok = load_first_vector(
        root + "/testdata/e2e/goldens/intel/minilm.json", &intel_gold
    );
    const bool nvidia_ok = load_first_vector(
        root + "/testdata/e2e/goldens/nvidia/minilm.json", &nvidia_gold
    );
    float cos_intel = 0.0f;
    float cos_nvidia = 0.0f;
    if (intel_ok && intel_gold.size() == live.size()) {
        cos_intel = cosine(live.data(), intel_gold.data(), live.size());
    }
    if (nvidia_ok && nvidia_gold.size() == live.size()) {
        cos_nvidia = cosine(live.data(), nvidia_gold.data(), live.size());
    }
    const bool embed_ok = intel_ok && nvidia_ok && embed_dim == 384 &&
                          cos_intel >= kEmbedCosineFloor &&
                          cos_nvidia >= kEmbedCosineFloor;

    turbo_buffer_ze_xfer_reset();
    turboembed_genai::genai_xfer_reset();
    std::vector<double> embed_ns;
    embed_ns.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        const double t0 = now_ns();
        if (turboembed_embed_one(
                ee, "minilm", 0, kHello, std::strlen(kHello), nullptr, &er
            ) != TURBOEMBED_OK ||
            er == nullptr) {
            std::fprintf(stderr, "turboembed measure: %s\n", turboembed_last_error(ee));
            turboembed_engine_destroy(ee);
            turborerank_engine_destroy(re);
            return 1;
        }
        embed_ns.push_back(now_ns() - t0);
        turboembed_embed_result_free(er);
        er = nullptr;
    }
    const Latency embed_lat = summarize(embed_ns, warmup, iters);
    const uint64_t embed_wrap_in_loop = turboembed_genai::genai_xfer_wrap_input_bytes();
    const uint64_t embed_ze_h2d_loop = turbo_buffer_ze_xfer_h2d_bytes();

    const bool allocs_ok = rerank_allocs == 0 && embed_allocs == 0;
    const bool explicit_ze_ok = rerank_ze_h2d == 0 && rerank_ze_d2h == 0 &&
                                embed_ze_h2d == 0 && embed_ze_d2h == 0 &&
                                rerank_ze_h2d_loop == 0 && embed_ze_h2d_loop == 0;
    const bool wrap_reported =
        rerank_wrap_in > 0 && embed_wrap_in > 0 && !remote_wrap;
    const bool xfer_honest = explicit_ze_ok && wrap_reported;
    const bool pass = allocs_ok && berlin_ok && embed_ok && xfer_honest &&
                      tokens_shared && owns_tokens == 1;

    const std::string sha = run_cmd("git rev-parse HEAD");
    const std::string out_path = root + "/testdata/receipts/bench/machine-b-ov.json";

    std::ostringstream js;
    js.setf(std::ios::fixed);
    js << "{\n";
    js << "  \"schema_version\": 1,\n";
    js << "  \"kind\": \"solidify-final-bench\",\n";
    js << "  \"machine\": \"Machine B\",\n";
    js << "  \"device\": \"OPENVINO_GPU\",\n";
    js << "  \"gpu\": \"" << json_escape(gpu) << "\",\n";
    js << "  \"git_sha\": \"" << json_escape(sha) << "\",\n";
    js << "  \"measured_at_utc\": \"" << utc_now() << "\",\n";
    js << "  \"command\": \"make bench-machine-b-ov\",\n";
    js << "  \"pass\": " << (pass ? "true" : "false") << ",\n";
    js << "  \"gates\": {\n";
    js << "    \"allocs_per_forward\": {\"required\": 0, \"embed\": " << embed_allocs
       << ", \"rerank\": " << rerank_allocs << "},\n";
    js << "    \"berlin_in_band\": " << (berlin_ok ? "true" : "false") << ",\n";
    js << "    \"embed_goldens_in_band\": " << (embed_ok ? "true" : "false") << ",\n";
    js << "    \"xfer_honest\": " << (xfer_honest ? "true" : "false") << "\n";
    js << "  },\n";
    js << "  \"turboembed\": {\n";
    js << "    \"alias\": \"minilm\",\n";
    js << "    \"text\": \"hello world\",\n";
    js << "    \"dim\": " << embed_dim << ",\n";
    js << "    \"latency\": {\n";
    write_latency(js, embed_lat, "      ");
    js << "    },\n";
    js << "    \"allocs_per_forward\": " << embed_allocs << ",\n";
    js << "    \"arena\": {\n";
    js << "      \"device\": " << arena_dev << ",\n";
    js << "      \"token_placement\": " << token_place << ",\n";
    js << "      \"tokens_shared\": " << (tokens_shared ? "true" : "false") << ",\n";
    js << "      \"owns_tokens\": " << (owns_tokens ? "true" : "false") << ",\n";
    js << "      \"owns_hidden\": " << (owns_hidden ? "true" : "false") << ",\n";
    js << "      \"hidden_used_arena\": " << (hidden_arena ? "true" : "false") << "\n";
    js << "    },\n";
    js << "    \"xfer\": {\n";
    js << "      \"model\": \"ov_host_tensor_wrap\",\n";
    js << "      \"remote_ocl_usm_wrap\": " << (remote_wrap ? "true" : "false") << ",\n";
    js << "      \"explicit_ze_h2d_bytes\": " << embed_ze_h2d << ",\n";
    js << "      \"explicit_ze_d2h_bytes\": " << embed_ze_d2h << ",\n";
    js << "      \"explicit_ze_h2d_calls\": " << embed_ze_h2d_calls << ",\n";
    js << "      \"explicit_ze_d2h_calls\": " << embed_ze_d2h_calls << ",\n";
    js << "      \"explicit_ze_h2d_bytes_measure_loop\": " << embed_ze_h2d_loop << ",\n";
    js << "      \"wrapped_input_bytes\": " << embed_wrap_in << ",\n";
    js << "      \"wrapped_input_bytes_measure_loop\": " << embed_wrap_in_loop << ",\n";
    js << "      \"hidden_wrap_bytes\": " << embed_hidden_wrap << ",\n";
    js << "      \"hidden_memcpy_bytes\": " << embed_hidden_memcpy << ",\n";
    js << "      \"n\": " << embed_n << ",\n";
    js << "      \"seq\": " << embed_seq << ",\n";
    js << "      \"n_inputs\": " << embed_inputs << ",\n";
    js << "      \"plugin_copy\": \"expected_host_ingest\",\n";
    js << "      \"note\": \"Remote OCL USM_USER_BUFFER wrap of ZE SHARED is "
          "unavailable. ov::Tensor host wrap means the GPU plugin may copy "
          "wrapped_input_bytes. This receipt does not claim H2D=0.\"\n";
    js << "    },\n";
    js.precision(10);
    js << "    \"cosine\": {\n";
    js << "      \"intel_golden\": " << cos_intel << ",\n";
    js << "      \"nvidia_golden\": " << cos_nvidia << ",\n";
    js << "      \"threshold\": " << kEmbedCosineFloor << "\n";
    js << "    }\n";
    js << "  },\n";
    js << "  \"turborerank\": {\n";
    js << "    \"alias\": \"ms-marco-minilm-l6\",\n";
    js << "    \"golden\": \"testdata/reference_rerank/ms_marco_minilm_l6_berlin.json\",\n";
    js << "    \"latency\": {\n";
    write_latency(js, rerank_lat, "      ");
    js << "    },\n";
    js << "    \"allocs_per_forward\": " << rerank_allocs << ",\n";
    js.precision(8);
    js << "    \"logits\": [" << logits[0] << ", " << logits[1] << ", " << logits[2]
       << "],\n";
    js.precision(6);
    js << "    \"max_abs_logit_err\": " << max_abs << ",\n";
    js.precision(10);
    js << "    \"cosine_vs_golden\": " << berlin_cos << ",\n";
    js << "    \"xfer\": {\n";
    js << "      \"model\": \"ov_host_tensor_wrap\",\n";
    js << "      \"remote_ocl_usm_wrap\": " << (remote_wrap ? "true" : "false") << ",\n";
    if (!wrap_why.empty()) {
        js << "      \"remote_ocl_usm_why\": \"" << json_escape(wrap_why) << "\",\n";
    }
    js << "      \"explicit_ze_h2d_bytes\": " << rerank_ze_h2d << ",\n";
    js << "      \"explicit_ze_d2h_bytes\": " << rerank_ze_d2h << ",\n";
    js << "      \"explicit_ze_h2d_calls\": " << rerank_ze_h2d_calls << ",\n";
    js << "      \"explicit_ze_d2h_calls\": " << rerank_ze_d2h_calls << ",\n";
    js << "      \"explicit_ze_h2d_bytes_measure_loop\": " << rerank_ze_h2d_loop << ",\n";
    js << "      \"wrapped_input_bytes\": " << rerank_wrap_in << ",\n";
    js << "      \"wrapped_input_bytes_measure_loop\": " << rerank_wrap_in_loop << ",\n";
    js << "      \"wrapped_result_bytes\": " << rerank_wrap_out << ",\n";
    js << "      \"n_rows\": " << rerank_n << ",\n";
    js << "      \"seq\": " << rerank_seq << ",\n";
    js << "      \"pack_seq\": " << pack_seq << ",\n";
    js << "      \"used_tokens\": [" << used_tokens[0] << ", " << used_tokens[1] << ", "
       << used_tokens[2] << "],\n";
    js << "      \"plugin_copy\": \"expected_host_ingest\",\n";
    js << "      \"note\": \"Work buffer is max_position (512). OV wraps the full "
          "[n, seq] i32 ids/mask/types. used_tokens is the nonzero mask. Plugin "
          "may copy wrapped_input_bytes; explicit ZE memcpy is 0.\"\n";
    js << "    }\n";
    js << "  },\n";
    js << "  \"note\": \"SOLIDIFY final bench Machine B. Live OpenVINO GPU. "
          "p50/p99 from CLOCK_MONOTONIC on this process. H↔D honesty: explicit "
          "turbo_buffer_ze_memcpy is 0; remote OCL wrap is unavailable so OV "
          "host-tensor ingest bytes are reported, not claimed-zero. "
          "allocs/forward is turbo_buffer_alloc_counter after warmup.\"\n";
    js << "}\n";

    {
        std::ofstream out(out_path);
        if (!out) {
            std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
            turboembed_engine_destroy(ee);
            turborerank_engine_destroy(re);
            return 1;
        }
        out << js.str();
    }

    std::fprintf(
        stderr,
        "wrote %s pass=%s embed_p50=%.3f us embed_p99=%.3f us "
        "rerank_p50=%.3f us rerank_p99=%.3f us "
        "allocs embed=%llu rerank=%llu berlin_abs=%.6g "
        "cos_intel=%.10f wrap_in embed=%llu rerank=%llu\n",
        out_path.c_str(),
        pass ? "true" : "false",
        embed_lat.p50_us,
        embed_lat.p99_us,
        rerank_lat.p50_us,
        rerank_lat.p99_us,
        static_cast<unsigned long long>(embed_allocs),
        static_cast<unsigned long long>(rerank_allocs),
        max_abs,
        cos_intel,
        static_cast<unsigned long long>(embed_wrap_in),
        static_cast<unsigned long long>(rerank_wrap_in)
    );

    turboembed_engine_destroy(ee);
    turborerank_engine_destroy(re);
    return pass ? 0 : 2;
}
