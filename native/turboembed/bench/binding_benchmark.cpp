// SPDX-License-Identifier: Apache-2.0
// Matched native ABI reference for java/benchmarks/BindingBenchmark.java.
#include <turboembed_prepared.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;
using json = nlohmann::json;
static te_error error{};
static void check(uint32_t code) {
    if (code) throw std::runtime_error(std::to_string(code) + ": " + error.message);
}
template<class T> static T descriptor() {
    T value{}; value.struct_size = sizeof(T); value.version = TE_PREPARED_VERSION; return value;
}
template<class F> static json measure(F call, int maximum, int warmup) {
    for (int i = 0; i < warmup; ++i) call();
    std::vector<int64_t> samples; samples.reserve(maximum);
    const auto start = Clock::now();
    for (int i = 0; i < maximum; ++i) {
        const auto before = Clock::now(); call(); const auto end = Clock::now();
        samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - before).count());
        if (end - start >= std::chrono::seconds(3)) break;
    }
    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    auto sorted = samples; std::sort(sorted.begin(), sorted.end());
    return {{"samples_ns", samples}, {"count", samples.size()}, {"elapsed_seconds", elapsed},
        {"p50_ns", sorted[(sorted.size() - 1) * 50 / 100]},
        {"p99_ns", sorted[(sorted.size() - 1) * 99 / 100]}, {"requests_per_second", samples.size() / elapsed}};
}
int main(int argc, char **argv) try {
    if (argc < 4 || argc > 5) throw std::runtime_error("usage: binding_benchmark BUNDLE BATCH SEQUENCE [gpu|cpu]");
    const int batch = std::stoi(argv[2]), sequence = std::stoi(argv[3]);
    if (batch < 1 || batch > 32 || sequence < 4 || sequence > 256) throw std::runtime_error("invalid shape");
    // GPU is the reference device; CPU must be selected explicitly.
    const std::string device = argc == 5 ? argv[4] : "gpu";
    if (device != "gpu" && device != "cpu") throw std::runtime_error("device must be gpu or cpu");
    auto opts = descriptor<te_context_options>();
    opts.device = device == "cpu" ? TE_DEVICE_OPENVINO_CPU : TE_DEVICE_OPENVINO_GPU;
    te_context *raw_context{}; check(turboembed_prepared_v1_context_create(&opts, &raw_context, &error));
    std::unique_ptr<te_context, decltype(&turboembed_prepared_v1_context_release)> context(raw_context, turboembed_prepared_v1_context_release);
    auto ci = descriptor<te_context_info>(); check(turboembed_prepared_v1_context_info(context.get(), &ci, &error));
    te_model *raw_model{}; check(turboembed_prepared_v1_model_load(context.get(), argv[1], std::strlen(argv[1]), &raw_model, &error));
    std::unique_ptr<te_model, decltype(&turboembed_prepared_v1_model_release)> model(raw_model, turboembed_prepared_v1_model_release);
    auto mi = descriptor<te_model_info>(); check(turboembed_prepared_v1_model_info(model.get(), &mi, &error));
    if (std::strcmp(mi.tokenizer_sha256, "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037") != 0)
        throw std::runtime_error("benchmark token IDs require pinned MiniLM tokenizer");
    auto so = descriptor<te_slot_options>(); so.batch = batch; so.sequence_length = sequence;
    te_slot *raw_slot{}; check(turboembed_prepared_v1_slot_create(model.get(), &so, &raw_slot, &error));
    std::unique_ptr<te_slot, decltype(&turboembed_prepared_v1_slot_release)> slot(raw_slot, turboembed_prepared_v1_slot_release);
    std::vector<int32_t> ids(batch * sequence), mask(ids.size()), types(ids.size());
    for (int row = 0; row < batch; ++row) {
        const int words[] = {101, 7592, 2088, 102};
        for (int i = 0; i < 4; ++i) { ids[row * sequence + i] = words[i]; mask[row * sequence + i] = 1; }
    }
    check(turboembed_prepared_v1_slot_write_tokens(slot.get(), ids.data(), mask.data(), types.data(), ids.size(), &error));
    std::vector<float> output(batch * mi.dimension), prepared(output.size());
    std::vector<te_text> texts(batch, te_text{"hello world", 11});
    auto stats = descriptor<te_slot_stats>();
    auto bridge = [&] { check(turboembed_prepared_v1_slot_stats(slot.get(), &stats, &error)); };
    auto infer = [&] {
        te_result *result{}; check(turboembed_prepared_v1_slot_execute(slot.get(), &result, &error));
        check(turboembed_prepared_v1_result_release(result, &error));
    };
    auto text = [&] {
        check(turboembed_prepared_v1_slot_write_text(slot.get(), texts.data(), texts.size(), &error));
        te_result *result{}; check(turboembed_prepared_v1_slot_execute(slot.get(), &result, &error));
        const auto read_status = turboembed_prepared_v1_result_read(result, output.data(), output.size(), &error);
        if (read_status) { const std::string message = error.message; turboembed_prepared_v1_result_release(result, &error); throw std::runtime_error(message); }
        check(turboembed_prepared_v1_result_release(result, &error));
    };
    te_result *result{}; check(turboembed_prepared_v1_slot_execute(slot.get(), &result, &error));
    check(turboembed_prepared_v1_result_read(result, prepared.data(), prepared.size(), &error));
    check(turboembed_prepared_v1_result_release(result, &error)); text();
    double maximum = 0, squared = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        const double difference = std::abs(static_cast<double>(output[i]) - prepared[i]);
        if (!std::isfinite(difference)) throw std::runtime_error("nonfinite output");
        maximum = std::max(maximum, difference); squared += difference * difference;
    }
    if (maximum > 1e-6 || std::sqrt(squared / output.size()) > 1e-7) throw std::runtime_error("prepared/text parity failed");
    json records = json::array();
    for (int repeat = 0; repeat < 3; ++repeat) {
        records.push_back({{"repeat", repeat}, {"bridge", measure(bridge, 100000, 100000)},
            {"prepared", measure(infer, 3000, 500)}, {"text_host", measure(text, 3000, 500)}});
    }
    std::cout << json{{"path", "native_abi"}, {"batch", batch}, {"sequence", sequence},
        {"device", ci.device_name}, {"runtime", ci.runtime_version}, {"model", mi.model_id},
        {"revision", mi.revision}, {"tokenizer_sha256", mi.tokenizer_sha256},
        {"text", "hello world"}, {"repeats", records}}.dump() << '\n';
} catch (const std::exception &failure) { std::cerr << failure.what() << '\n'; return 1; }
