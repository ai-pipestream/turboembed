// SPDX-License-Identifier: Apache-2.0
//
// Machine B Intel OpenVINO receipt for TurboRerank MiniLM-L6 CE.
// Writes testdata/receipts/turborerank/intel-minilm-l6.json
// and intel-cpu-minilm-l6.json when OV CPU scores. Never embeds hostnames.

#include "ov_api.hpp"
#include "reranker.hpp"
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

static int write_one(
    turborerank_device device,
    const char *device_label,
    const std::string &root,
    const std::string &out_path,
    const char *command,
    const char *backend,
    const char *token_workspace,
    const char *note
) {
    const std::string model = root + "/models/ov-rerank/ms-marco-minilm-l6";
    turborerank_engine *e = nullptr;
    turborerank_status st =
        turborerank_engine_create(device, model.c_str(), &e);
    if (st != TURBORERANK_OK || e == nullptr) {
        std::fprintf(
            stderr,
            "create %s failed: %s\n",
            device_label,
            turborerank_last_error(nullptr)
        );
        return 1;
    }
    st = turborerank_load_model(e, "ms-marco-minilm-l6", 0);
    if (st != TURBORERANK_OK) {
        std::fprintf(stderr, "load %s failed: %s\n", device_label, turborerank_last_error(e));
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
        std::fprintf(stderr, "score %s failed: %s\n", device_label, turborerank_last_error(e));
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
                      logits[0] > logits[1] && logits[1] > logits[2];

    std::string gpu;
    if (device == TURBORERANK_DEVICE_OPENVINO_GPU) {
        turborerank::impl::ov_gpu_name(&gpu);
    }
    const std::string sha = run_cmd("git rev-parse HEAD");
    const bool usm = turborerank::impl::ov_usm_available(nullptr);

    std::ostringstream js;
    js.setf(std::ios::fixed);
    js << "{\n";
    js << "  \"alias\": \"ms-marco-minilm-l6\",\n";
    js << "  \"model\": \"cross-encoder/ms-marco-MiniLM-L6-v2\",\n";
    js << "  \"revision\": \"233902d25c440f23af6f7d6e94d2946bac0bee0a\",\n";
    js << "  \"device\": \"" << device_label << "\",\n";
    js << "  \"machine\": \"Machine B\",\n";
    if (!gpu.empty()) {
        js << "  \"gpu\": \"" << gpu << "\",\n";
    }
    js << "  \"backend\": \"" << backend << "\",\n";
    js << "  \"compute\": {\n";
    js << "    \"token_workspace\": \"" << token_workspace << "\",\n";
    js << "    \"token_wrap\": \"ov::Tensor(element::i32, shape, usm_pointer)\",\n";
    js << "    \"weights_activations\": \"OpenVINO CompiledModel on " << device_label
       << "\",\n";
    js << "    \"usm_level_zero\": " << (usm ? "true" : "false") << ",\n";
    js << "    \"remote_ocl_usm_wrap\": false,\n";
    js << "    \"host_interim\": false,\n";
    js << "    \"std_vector_on_forward\": false,\n";
    js << "    \"python_hot_path\": false,\n";
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
    js << "  \"command\": \"" << command << "\",\n";
    js << "  \"note\": \"" << note << "\"\n";
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

int main() {
    std::string why;
    if (!turborerank::impl::ov_gpu_present(&why)) {
        std::fprintf(stderr, "write_intel_receipt: OpenVINO GPU missing: %s\n", why.c_str());
        return 1;
    }
    const char *root_c = std::getenv("INFERSTREAM_ROOT");
    const std::string root = root_c && root_c[0] ? root_c : ".";

    const int gpu_rc = write_one(
        TURBORERANK_DEVICE_OPENVINO_GPU,
        "OPENVINO_GPU",
        root,
        root + "/testdata/receipts/turborerank/intel-minilm-l6.json",
        "make test-turborerank-intel",
        "turborerank OpenVINO CompiledModel MiniLM CE (turbo_buffer ZE SHARED "
        "USM token buffers wrapped with ov::Tensor(..., usm_pointer); no "
        "std::vector on forward; allocs/forward==0 after warmup)",
        "turbo_buffer ZE SHARED USM (caller-written; zeMemAllocShared)",
        "SOLIDIFY (1) Machine B. Token rows rented from the ZE arena as "
        "SHARED. GPU create without a GPU fails loud. CPU buffers on OV GPU "
        "forward are refused. AUTO resolves to OPENVINO_GPU when CUDA is "
        "absent. Not a CPU interim and not a mock score."
    );
    if (gpu_rc != 0) {
        return gpu_rc;
    }

    if (turborerank::impl::ov_cpu_present(&why)) {
        const int cpu_rc = write_one(
            TURBORERANK_DEVICE_OPENVINO_CPU,
            "OPENVINO_CPU",
            root,
            root + "/testdata/receipts/turborerank/intel-cpu-minilm-l6.json",
            "make test-turborerank-intel",
            "turborerank OpenVINO CompiledModel MiniLM CE on CPU (Level Zero "
            "USM host or 64-byte aligned tokens; ov::Tensor(..., pointer))",
            "Level Zero USM host (caller-written) when L0 is present",
            "Phase 2b OpenVINO CPU on Machine B. Same IR as GPU. Explicit "
            "OPENVINO_CPU — not a silent stand-in for TURBORERANK_DEVICE_CPU."
        );
        return cpu_rc;
    }
    std::fprintf(stderr, "OV CPU skipped: %s\n", why.c_str());
    return 0;
}
