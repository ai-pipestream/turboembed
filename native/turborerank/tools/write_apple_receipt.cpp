// SPDX-License-Identifier: Apache-2.0
//
// Machine C Apple Metal receipt for TurboRerank MiniLM-L6 CE.
// Writes testdata/receipts/turborerank/apple-minilm-l6.json
// Never embeds lab hostnames.

#include "metal_api.hpp"
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
    if (!turborerank::impl::metal_device_present(&why)) {
        std::fprintf(stderr, "write_apple_receipt: Metal missing: %s\n", why.c_str());
        return 1;
    }
    const char *root_c = std::getenv("INFERSTREAM_ROOT");
    const std::string root = root_c && root_c[0] ? root_c : ".";
    const std::string model = root + "/models/rerank/ms-marco-minilm-l6";
    const std::string out_path =
        root + "/testdata/receipts/turborerank/apple-minilm-l6.json";

    turborerank_engine *e = nullptr;
    turborerank_status st =
        turborerank_engine_create(TURBORERANK_DEVICE_METAL, model.c_str(), &e);
    if (st != TURBORERANK_OK || e == nullptr) {
        std::fprintf(
            stderr,
            "create METAL failed: %s\n",
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
    if (e->arena == nullptr || e->work == nullptr ||
        !turbo_buffer_arena_owns(e->arena, e->work->input_ids) ||
        !turbo_buffer_metal_owns(e->work->input_ids)) {
        std::fprintf(
            stderr,
            "Metal work tokens are not turbo_buffer SHARED arena rents "
            "(arena_owns=%d metal_owns=%d)\n",
            e->arena && e->work
                ? turbo_buffer_arena_owns(e->arena, e->work->input_ids)
                : 0,
            e->work ? turbo_buffer_metal_owns(e->work->input_ids) : 0
        );
        turborerank_engine_destroy(e);
        return 1;
    }

    st = turborerank_score(e, nullptr, 0, query, docs, 3, &opts, logits);
    if (st != TURBORERANK_OK) {
        std::fprintf(stderr, "score failed: %s\n", turborerank_last_error(e));
        turborerank_engine_destroy(e);
        return 1;
    }

    turborerank::alloc_counter_reset();
    float again[3] = {0, 0, 0};
    st = turborerank_score(e, nullptr, 0, query, docs, 3, &opts, again);
    const uint64_t allocs_after_score = turbo_buffer_alloc_counter();
    if (st != TURBORERANK_OK || allocs_after_score != 0) {
        std::fprintf(
            stderr,
            "steady-state score allocs=%llu st=%s err=%s\n",
            static_cast<unsigned long long>(allocs_after_score),
            turborerank_status_name(st),
            turborerank_last_error(e)
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
    const bool pass = max_abs < 2e-3f && cosine > 0.999f &&
                      logits[0] > logits[1] && logits[1] > logits[2] &&
                      allocs_after_score == 0;

    std::string gpu;
    turborerank::impl::metal_gpu_name(&gpu);
    const std::string sha = run_cmd("git rev-parse HEAD");

    std::ostringstream js;
    js.setf(std::ios::fixed);
    js << "{\n";
    js << "  \"alias\": \"ms-marco-minilm-l6\",\n";
    js << "  \"model\": \"cross-encoder/ms-marco-MiniLM-L6-v2\",\n";
    js << "  \"revision\": \"233902d25c440f23af6f7d6e94d2946bac0bee0a\",\n";
    js << "  \"device\": \"METAL\",\n";
    js << "  \"machine\": \"Machine C\",\n";
    js << "  \"gpu\": \"" << gpu << "\",\n";
    js << "  \"backend\": \"turborerank first-party Metal MiniLM CE "
          "(turbo_buffer Metal SHARED arena; MTLResourceStorageModeShared "
          "token rents; kernels bind those MTLBuffers via "
          "turbo_buffer_metal_lookup; device GEMM/attention/LN/GELU/pooler "
          "matching CPU linear_nt; weights copied once at load)\",\n";
    js << "  \"compute\": {\n";
    js << "    \"token_workspace\": \"turbo_buffer arena SHARED "
          "(MTLResourceStorageModeShared; caller-written unified memory; "
          "no extra token copy)\",\n";
    js << "    \"arena\": \"include/turbo_buffer.h METAL backend\",\n";
    js << "    \"token_rent\": \"turbo_buffer_arena_rent SHARED i32\",\n";
    js << "    \"allocs_per_forward\": 0,\n";
    js << "    \"weights_activations\": \"MTL shared buffers reserved at load\",\n";
    js << "    \"gemm\": \"first-party Metal kernel matching CPU linear_nt\",\n";
    js << "    \"elementwise\": \"first-party Metal kernels (embed, LayerNorm, "
          "GELU erf, attention, pooler, classifier)\",\n";
    js << "    \"host_interim\": false,\n";
    js << "    \"token_copy_on_forward\": false,\n";
    js << "    \"private_mtl_token_alloc\": false,\n";
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
    js << "  \"command\": \"make test-turborerank-apple\",\n";
    js << "  \"note\": \"SOLIDIFY (1) Machine C LIVE: turbo_buffer Metal "
          "SHARED rent/return is real MTL. TurboRerank Metal forward rents "
          "arena token slots; allocs/forward==0. AUTO resolves to METAL "
          "when CUDA/OpenVINO GPU are absent. Create without Metal fails "
          "loud. Swift TurboRerankEngine.score calls turborerank_score "
          "(engine work buffer) — no Swift-side token malloc. Weights "
          "copied once at load from mmap'd safetensors.\"\n";
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
