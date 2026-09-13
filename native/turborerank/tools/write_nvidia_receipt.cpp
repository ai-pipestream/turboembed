// SPDX-License-Identifier: Apache-2.0
//
// Machine A NVIDIA receipt for TurboRerank MiniLM-L6 CE.
// Writes testdata/receipts/turborerank/nvidia-minilm-l6.json
// Never embeds lab hostnames.

#include "cuda_api.hpp"
#include "reranker.hpp"
#include "turbo_buffer.h"
#include "turborerank.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

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
    if (!turborerank::impl::cuda_device_present(&why)) {
        std::fprintf(stderr, "write_nvidia_receipt: CUDA missing: %s\n", why.c_str());
        return 1;
    }
    const char *root_c = std::getenv("INFERSTREAM_ROOT");
    const std::string root = root_c && root_c[0] ? root_c : ".";
    const std::string model = root + "/models/rerank/ms-marco-minilm-l6";
    const std::string out_path =
        root + "/testdata/receipts/turborerank/nvidia-minilm-l6.json";

    turborerank_engine *e = nullptr;
    turborerank_status st =
        turborerank_engine_create(TURBORERANK_DEVICE_CUDA, model.c_str(), &e);
    if (st != TURBORERANK_OK || e == nullptr) {
        std::fprintf(
            stderr,
            "create CUDA failed: %s\n",
            turborerank_last_error(nullptr)
        );
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
    float logits[3] = {0, 0, 0};
    st = turborerank_score(e, nullptr, 0, query, docs, 3, &opts, logits);
    if (st != TURBORERANK_OK) {
        std::fprintf(stderr, "score failed: %s\n", turborerank_last_error(e));
        turborerank_engine_destroy(e);
        return 1;
    }
    turbo_buffer_alloc_counter_reset();
    turbo_buffer_cuda_forward_allocs_reset();
    turbo_buffer_cuda_forward_h2d_reset();
    float steady[3] = {0, 0, 0};
    st = turborerank_score(e, nullptr, 0, query, docs, 3, &opts, steady);
    if (st != TURBORERANK_OK) {
        std::fprintf(stderr, "steady score failed: %s\n", turborerank_last_error(e));
        turborerank_engine_destroy(e);
        return 1;
    }
    const uint64_t allocs_fwd = turbo_buffer_alloc_counter();
    const uint64_t cuda_mallocs_fwd = turbo_buffer_cuda_forward_allocs();
    const uint64_t h2d_bytes = turbo_buffer_cuda_forward_h2d_bytes();
    const uint64_t h2d_calls = turbo_buffer_cuda_forward_h2d_calls();
    if (allocs_fwd != 0 || cuda_mallocs_fwd != 0 || h2d_bytes != 0 || h2d_calls != 0) {
        std::fprintf(
            stderr,
            "per-forward not zero: arena=%llu cudaMalloc=%llu h2d_bytes=%llu "
            "h2d_calls=%llu\n",
            static_cast<unsigned long long>(allocs_fwd),
            static_cast<unsigned long long>(cuda_mallocs_fwd),
            static_cast<unsigned long long>(h2d_bytes),
            static_cast<unsigned long long>(h2d_calls)
        );
        turborerank_engine_destroy(e);
        return 2;
    }
    for (int i = 0; i < 3; ++i) {
        logits[i] = steady[i];
    }

    const float gold[3] = {8.84585285f, -4.32007599f, -11.27389431f};
    float abs_err[3];
    float max_abs = 0.0f;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < 3; ++i) {
        abs_err[i] = std::fabs(logits[i] - gold[i]);
        if (abs_err[i] > max_abs) {
            max_abs = abs_err[i];
        }
        dot += logits[i] * gold[i];
        na += logits[i] * logits[i];
        nb += gold[i] * gold[i];
    }
    const float cosine = dot / (std::sqrt(na) * std::sqrt(nb));
    const bool pass = max_abs < 2e-3f && cosine > 0.999f &&
                      logits[0] > logits[1] && logits[1] > logits[2] &&
                      h2d_bytes == 0 && h2d_calls == 0;

    std::string gpu;
    turborerank::impl::cuda_gpu_name(&gpu);
    const std::string sha = run_cmd("git rev-parse HEAD");

    std::ostringstream js;
    js.setf(std::ios::fixed);
    js << "{\n";
    js << "  \"alias\": \"ms-marco-minilm-l6\",\n";
    js << "  \"model\": \"cross-encoder/ms-marco-MiniLM-L6-v2\",\n";
    js << "  \"revision\": \"233902d25c440f23af6f7d6e94d2946bac0bee0a\",\n";
    js << "  \"device\": \"CUDA\",\n";
    js << "  \"machine\": \"Machine A\",\n";
    js << "  \"gpu\": \"" << gpu << "\",\n";
    js << "  \"backend\": \"turborerank first-party CUDA MiniLM CE (turbo_buffer "
          "PINNED mapped token rent + DEVICE activation scratch; device "
          "GEMM/attention/LN/GELU/pooler kernels; kernels read mapped int32 "
          "ids — 0 H2D bytes per row)\",\n";
    js << "  \"compute\": {\n";
    js << "    \"token_workspace\": \"turbo_buffer PINNED mapped rent "
          "(cudaHostAllocMapped; host write lands in device-visible pages)\",\n";
    js << "    \"activations\": \"turbo_buffer DEVICE rent at load\",\n";
    js << "    \"weights\": \"cudaMalloc at load (not per-forward)\",\n";
    js << "    \"allocs_per_forward\": 0,\n";
    js << "    \"h2d_per_row\": 0,\n";
    js << "    \"h2d_bytes_steady\": " << h2d_bytes << ",\n";
    js << "    \"h2d_calls_steady\": " << h2d_calls << ",\n";
    js << "    \"gemm\": \"first-party CUDA kernel matching CPU linear_nt\",\n";
    js << "    \"elementwise\": \"first-party CUDA kernels (embed, LayerNorm, "
          "GELU erf, attention, pooler, classifier)\",\n";
    js << "    \"host_interim\": false,\n";
    js << "    \"mock\": false\n";
    js << "  },\n";
    js << "  \"pass\": " << (pass ? "true" : "false") << ",\n";
    js << "  \"golden\": \"testdata/reference_rerank/ms_marco_minilm_l6_berlin.json\",\n";
    js << "  \"reference\": \"HuggingFace AutoModelForSequenceClassification on "
          "the pinned safetensors\",\n";
    js.precision(8);
    js << "  \"logits\": [" << logits[0] << ", " << logits[1] << ", " << logits[2]
       << "],\n";
    js.precision(6);
    js << "  \"max_abs_logit_err\": " << max_abs << ",\n";
    js.precision(10);
    js << "  \"cosine_vs_golden\": " << cosine << ",\n";
    js << "  \"git_sha\": \"" << sha << "\",\n";
    js << "  \"command\": \"make test-turborerank-nvidia\",\n";
    js << "  \"note\": \"SOLIDIFY (2) CUDA token H2D killed on Machine A. "
          "PINNED mapped (cudaHostAllocMapped) token rent; host tokenize/"
          "pack writes those pages; kernels use turbo_buffer_cuda_mapped_"
          "device_ptr. Steady-state h2d_bytes == 0 and allocs/forward == 0. "
          "Unmapped pointers fail loud (no convenience H2D). AUTO resolves "
          "to CUDA. Not a pinned-host CPU interim: the BERT graph including "
          "GEMM runs on device.\"\n";
    js << "}\n";

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
        "wrote %s pass=%s max_abs=%.6g cosine=%.10f logits=%.8f %.8f %.8f\n",
        out_path.c_str(),
        pass ? "true" : "false",
        max_abs,
        cosine,
        logits[0],
        logits[1],
        logits[2]
    );
    return pass ? 0 : 2;
}
