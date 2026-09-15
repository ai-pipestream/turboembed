// SPDX-License-Identifier: Apache-2.0
#pragma once
// Independent direct OpenVINO reference. Deliberately does not call SDK graph
// construction or the public ABI; keep its numerical contract matched in tests.
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <map>
#include <stdexcept>
namespace op = ov::opset13;
inline std::shared_ptr<ov::Model> pooled_model(ov::Core &core, const std::string &dir, uint32_t batch, uint32_t sequence, uint32_t dimension = 384) {
    auto model = core.read_model(dir + "/openvino_model.xml");
    std::map<std::string, ov::PartialShape> shapes;
    ov::Output<ov::Node> mask;
    for (const auto &input : model->inputs()) {
        const auto name = input.get_any_name();
        shapes[name] = ov::PartialShape{batch, sequence};
        if (name == "attention_mask") { mask = input; }
    }
    if (!mask.get_node_shared_ptr()) { throw std::runtime_error("attention_mask input missing"); }
    model->reshape(shapes);
    const auto hidden = model->get_results().at(0)->input_value(0);
    if (hidden.get_shape() != ov::Shape{batch, sequence, dimension}) { throw std::runtime_error("unexpected MiniLM output shape"); }
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
