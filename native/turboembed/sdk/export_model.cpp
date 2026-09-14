// SPDX-License-Identifier: Apache-2.0

#include <openvino/core/graph_util.hpp>
#include <openvino/core/version.hpp>
#include <openvino/openvino.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("usage: turboembed-export-model ONNX_PATH OUT_DIR");
        }
        const std::filesystem::path source(argv[1]);
        const std::filesystem::path output(argv[2]);
        if (!std::filesystem::is_regular_file(source)) {
            throw std::runtime_error("ONNX source is not a regular file");
        }
        if (!std::filesystem::is_directory(output)) {
            throw std::runtime_error("output directory does not exist");
        }

        ov::Core core;
        const auto model = core.read_model(source.string());
        ov::serialize(
            model,
            (output / "openvino_model.xml").string(),
            (output / "openvino_model.bin").string()
        );
        const ov::Version version = ov::get_openvino_version();
        if (version.buildNumber == nullptr || version.buildNumber[0] == '\0') {
            throw std::runtime_error("OpenVINO returned an empty build version");
        }
        std::cout << version.buildNumber << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "turboembed-export-model: " << error.what() << '\n';
        return 1;
    }
}
