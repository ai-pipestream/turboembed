// SPDX-License-Identifier: Apache-2.0
//
// Fetch-tooling: convert MiniLM CE ONNX → SHA-pinned OpenVINO IR.
// Not on the score path.

#include <openvino/openvino.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>

static bool is_file(const char *p) {
    struct stat st {};
    return stat(p, &st) == 0 && S_ISREG(st.st_mode);
}

int main(int argc, char **argv) {
    const char *onnx = argc > 1 ? argv[1]
                                : "models/ov-rerank/ms-marco-minilm-l6/model.onnx";
    const char *xml = argc > 2 ? argv[2]
                               : "models/ov-rerank/ms-marco-minilm-l6/openvino_model.xml";
    if (!is_file(onnx)) {
        std::fprintf(stderr, "onnx_to_ir: missing %s\n", onnx);
        return 1;
    }
    try {
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(onnx);
        ov::save_model(model, xml, true);
        std::fprintf(stderr, "wrote %s\n", xml);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "onnx_to_ir failed: %s\n", e.what());
        return 2;
    }
}
