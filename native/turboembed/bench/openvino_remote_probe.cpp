// SPDX-License-Identifier: Apache-2.0
// Direct-native correctness probe: remote tensors, GPU pooling, and a following
// OpenCL consumer. This is not a native/ABI performance comparison.
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

namespace op = ov::opset13;

std::shared_ptr<ov::Model> pooled_model(ov::Core &core, const std::string &dir) {
    auto model = core.read_model(dir + "/openvino_model.xml");
    std::map<std::string, ov::PartialShape> shapes;
    ov::Output<ov::Node> mask;
    for (const auto &input : model->inputs()) {
        const auto name = input.get_any_name();
        shapes[name] = ov::PartialShape{1, 32};
        if (name == "attention_mask") { mask = input; }
    }
    if (!mask.get_node_shared_ptr()) { throw std::runtime_error("attention_mask input missing"); }
    model->reshape(shapes);
    const auto hidden = model->get_results().at(0)->input_value(0);
    if (hidden.get_shape() != ov::Shape{1, 32, 384}) { throw std::runtime_error("unexpected MiniLM output shape"); }
    const auto axis1 = op::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    const auto axis2 = op::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto mask_f = std::make_shared<op::Convert>(mask, ov::element::f32);
    auto mask_3d = std::make_shared<op::Unsqueeze>(mask_f, axis2);
    auto sum = std::make_shared<op::ReduceSum>(std::make_shared<op::Multiply>(hidden, mask_3d), axis1, false);
    auto count = std::make_shared<op::ReduceSum>(mask_f, axis1, true);
    auto divisor = std::make_shared<op::Maximum>(count, op::Constant::create(ov::element::f32, ov::Shape{}, {1.0f}));
    auto mean = std::make_shared<op::Divide>(sum, divisor);
    auto squared = std::make_shared<op::Multiply>(mean, mean);
    auto norm = std::make_shared<op::Sqrt>(std::make_shared<op::ReduceSum>(squared, axis1, true));
    auto safe_norm = std::make_shared<op::Maximum>(norm, op::Constant::create(ov::element::f32, ov::Shape{}, {1e-12f}));
    auto normalized = std::make_shared<op::Divide>(mean, safe_norm);
    normalized->output(0).set_names({"embeddings"});
    model = std::make_shared<ov::Model>(ov::OutputVector{normalized}, model->get_parameters());
    ov::preprocess::PrePostProcessor prep(model);
    for (size_t i = 0; i < model->inputs().size(); ++i) { prep.input(i).tensor().set_element_type(ov::element::i32); }
    return prep.build();
}

int main(int argc, char **argv) {
    try {
        if (argc != 2) { throw std::runtime_error("usage: openvino_remote_probe MODEL_BUNDLE_DIR"); }
        const std::string dir = argv[1];
        std::ifstream input(dir + "/tokenizer.json");
        const auto tokenizer = nlohmann::json::parse(input);
        const auto &vocab = tokenizer.at("model").at("vocab");
        std::vector<int32_t> ids(32, vocab.at("[PAD]").get<int32_t>()), masks(32, 0), types(32, 0);
        ids[0] = vocab.at("[CLS]"); ids[1] = vocab.at("hello");
        ids[2] = vocab.at("world"); ids[3] = vocab.at("[SEP]");
        std::fill_n(masks.begin(), 4, 1);
        std::map<std::string, std::vector<int32_t> *> rows = {
            {"input_ids", &ids}, {"attention_mask", &masks}, {"token_type_ids", &types}};
        ov::Core core;
        auto graph = pooled_model(core, dir);
        ov::AnyMap properties = {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                                 ov::hint::inference_precision(ov::element::f32)};
        auto cpu = core.compile_model(graph, "CPU", properties);
        auto cpu_request = cpu.create_infer_request();
        for (const auto &port : cpu.inputs()) {
            auto &values = *rows.at(port.get_any_name());
            cpu_request.set_tensor(port, ov::Tensor(ov::element::i32, ov::Shape{1, 32}, values.data()));
        }
        cpu_request.infer();
        auto cpu_result = cpu_request.get_output_tensor();
        if (cpu_result.get_element_type() != ov::element::f32 ||
            cpu_result.get_shape() != ov::Shape{1, 384}) {
            throw std::runtime_error("unexpected CPU embedding layout");
        }
        const auto *expected = cpu_result.data<const float>();

        auto default_context = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
        cl::Context context(default_context.get(), true);
        const auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty()) { throw std::runtime_error("OpenCL context has no device"); }
        cl::CommandQueue queue(context, devices.front());
        ov::intel_gpu::ocl::ClContext shared_context(core, queue.get());
        auto gpu = core.compile_model(graph, shared_context, properties);
        std::vector<cl::Buffer> buffers;
        cl::Buffer output(context, CL_MEM_READ_WRITE, 384 * sizeof(float));
        auto request = gpu.create_infer_request();
        for (const auto &port : gpu.inputs()) {
            auto &values = *rows.at(port.get_any_name());
            buffers.emplace_back(context, CL_MEM_READ_WRITE, values.size() * sizeof(int32_t));
            queue.enqueueWriteBuffer(buffers.back(), CL_TRUE, 0, values.size() * sizeof(int32_t), values.data());
            auto tensor = shared_context.create_tensor(ov::element::i32, ov::Shape{1, 32}, buffers.back());
            request.set_tensor(port, tensor);
        }
        auto remote_output = shared_context.create_tensor(ov::element::f32, ov::Shape{1, 384}, output);
        if (remote_output.get() != output.get()) { throw std::runtime_error("remote output buffer identity changed"); }
        request.set_output_tensor(remote_output);
        request.infer();

        // Consume the embedding on the same GPU queue before any host readback.
        cl::Program program(context, R"CLC(
            __kernel void twice(__global const float *embedding, __global float *result) {
                size_t i = get_global_id(0); result[i] = 2.0f * embedding[i];
            }
        )CLC");
        program.build({devices.front()});
        cl::Kernel consumer(program, "twice");
        cl::Buffer consumed(context, CL_MEM_READ_WRITE, 384 * sizeof(float));
        consumer.setArg(0, output); consumer.setArg(1, consumed);
        queue.enqueueNDRangeKernel(consumer, cl::NullRange, cl::NDRange(384), cl::NullRange);
        std::vector<float> host(384);
        queue.enqueueReadBuffer(consumed, CL_TRUE, 0, host.size() * sizeof(float), host.data());
        double squared_error = 0.0;
        float max_error = 0.0f;
        for (size_t i = 0; i < host.size(); ++i) {
            const float value = host[i] / 2.0f;
            if (!std::isfinite(value) || !std::isfinite(expected[i])) { throw std::runtime_error("nonfinite GPU output"); }
            const float error = std::abs(value - expected[i]);
            max_error = std::max(max_error, error);
            squared_error += static_cast<double>(error) * error;
        }
        const double rmse = std::sqrt(squared_error / host.size());
        std::cout << "gpu=" << core.get_property("GPU", ov::device::full_name)
                  << " max_abs_error=" << max_error << " rmse=" << rmse
                  << " host_input_bytes=" << buffers.size() * 32 * sizeof(int32_t)
                  << " explicit_readback_bytes=" << host.size() * sizeof(float) << '\n';
        if (max_error > 5e-4f || rmse > 1e-4) { throw std::runtime_error("GPU/CPU numerical gate failed"); }
        std::cout << "PASS: remote inputs, GPU pooled output, downstream OpenCL consumer, explicit readback\n";
        return 0;
    } catch (const cl::Error &e) {
        std::cerr << "OpenCL failure: " << e.what() << " (" << e.err() << ")\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
    }
    return 1;
}
