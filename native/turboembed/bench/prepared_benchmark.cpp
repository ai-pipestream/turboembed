// SPDX-License-Identifier: Apache-2.0
#include "openvino_reference.hpp"
#include "turboembed_prepared.h"
#include "nlohmann/json.hpp"
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <new>
#include <numeric>
#include <sys/resource.h>
#include <vector>

// Diagnostic scope: C++ new/new[] calls resolving through this executable.
// This does not observe malloc, driver allocations, or hidden vendor allocators.
std::atomic<bool> count_new{false};
std::atomic<uint64_t> new_calls{0}, new_bytes{0};
void *operator new(std::size_t size) {
    void *p = std::malloc(size ? size : 1);
    if (!p) { throw std::bad_alloc(); }
    if (count_new.load(std::memory_order_relaxed)) {
        new_calls.fetch_add(1, std::memory_order_relaxed);
        new_bytes.fetch_add(size, std::memory_order_relaxed);
    }
    return p;
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
double micros(Clock::duration time) { return std::chrono::duration<double, std::micro>(time).count(); }
void check(bool ok, const char *why) { if (!ok) { throw std::runtime_error(why); } }
void status(uint32_t code, const te_error &error) {
    if (code != TE_OK) { throw std::runtime_error(std::to_string(code) + ": " + error.message); }
}
template<typename T> T descriptor() {
    T value{}; value.struct_size = sizeof(T); value.version = 1; return value;
}
struct Inputs {
    uint32_t batch, sequence;
    std::array<std::vector<int32_t>, 3> rows;
    std::vector<uint32_t> lengths;
    Inputs(const std::string &dir, uint32_t b, uint32_t s, bool mixed) : batch(b), sequence(s) {
        Json tokenizer; std::ifstream(dir + "/tokenizer.json") >> tokenizer;
        const auto &vocab = tokenizer.at("model").at("vocab");
        rows[0].resize(b * s, vocab.at("[PAD]").get<int32_t>());
        rows[1].resize(b * s, 0); rows[2].resize(b * s, 0);
        for (uint32_t i = 0; i < b; ++i) {
            const uint32_t length = mixed ? 2 + ((i * 29 + 11) % (s - 1)) : s;
            lengths.push_back(length);
            for (uint32_t j = 0; j < length; ++j) {
                rows[0][i * s + j] = vocab.at(j % 2 ? "hello" : "world").get<int32_t>();
                rows[1][i * s + j] = 1;
            }
            rows[0][i * s] = vocab.at("[CLS]");
            rows[0][i * s + length - 1] = vocab.at("[SEP]");
        }
    }
};
struct Direct {
    ov::Core core;
    cl::Context context;
    cl::CommandQueue queue;
    std::array<cl::Buffer, 3> inputs;
    cl::Buffer output;
    ov::CompiledModel compiled;
    ov::InferRequest request;
    uint32_t batch;
    Direct(const std::string &dir, const Inputs &data, bool profiling = false) : batch(data.batch) {
        auto graph = pooled_model(core, dir, data.batch, data.sequence);
        auto original = core.get_default_context("GPU.0").as<ov::intel_gpu::ocl::ClContext>();
        context = cl::Context(original.get(), true);
        auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
        check(devices.size() == 1, "direct reference requires one OpenCL device");
        queue = cl::CommandQueue(context, devices.front(), profiling ? CL_QUEUE_PROFILING_ENABLE : 0);
        ov::intel_gpu::ocl::ClContext remote(core, queue.get());
        ov::AnyMap properties = {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                                ov::hint::inference_precision(ov::element::f32)};
        if (profiling) { properties.emplace(ov::enable_profiling(true)); }
        compiled = core.compile_model(graph, remote, properties);
        request = compiled.create_infer_request();
        for (const auto &port : compiled.inputs()) {
            const auto name = port.get_any_name();
            const size_t index = name == "input_ids" ? 0 : name == "attention_mask" ? 1 : 2;
            auto &buffer = inputs[index]; const auto &row = data.rows[index];
            buffer = cl::Buffer(context, CL_MEM_READ_WRITE, row.size() * sizeof(int32_t));
            queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, row.size() * sizeof(int32_t), row.data());
            request.set_tensor(port, remote.create_tensor(ov::element::i32,
                ov::Shape{data.batch, data.sequence}, buffer));
        }
        output = cl::Buffer(context, CL_MEM_READ_WRITE, batch * 384 * sizeof(float));
        request.set_output_tensor(remote.create_tensor(ov::element::f32, ov::Shape{batch, 384}, output));
    }
    void execute() { request.infer(); }
    std::vector<float> read() {
        std::vector<float> values(batch * 384);
        queue.enqueueReadBuffer(output, CL_TRUE, 0, values.size() * sizeof(float), values.data());
        return values;
    }
    Json device_profile() {
        cl::Event start, end;
        queue.enqueueMarkerWithWaitList(nullptr, &start);
        execute();
        queue.enqueueMarkerWithWaitList(nullptr, &end); end.wait();
        Json nodes = Json::array();
        for (const auto &item : request.get_profiling_info()) {
            if (item.status == ov::ProfilingInfo::Status::EXECUTED) {
                nodes.push_back({{"node", item.node_name}, {"type", item.exec_type},
                    {"real_us", item.real_time.count()}, {"cpu_us", item.cpu_time.count()}});
            }
        }
        return {{"queue_device_elapsed_us", (end.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
            start.getProfilingInfo<CL_PROFILING_COMMAND_END>()) / 1000.0}, {"nodes", nodes}};
    }
};
struct Abi {
    te_context *context = nullptr;
    te_model *model = nullptr;
    te_slot *slot = nullptr;
    te_error error{};
    uint32_t batch;
    Abi(const std::string &dir, const Inputs &data) : batch(data.batch) {
        try {
            auto options = descriptor<te_context_options>(); options.device = TE_DEVICE_OPENVINO_GPU;
            status(turboembed_prepared_v1_context_create(&options, &context, &error), error);
            status(turboembed_prepared_v1_model_load(context, dir.data(), dir.size(), &model, &error), error);
            auto shape = descriptor<te_slot_options>(); shape.batch = data.batch; shape.sequence_length = data.sequence;
            status(turboembed_prepared_v1_slot_create(model, &shape, &slot, &error), error);
            status(turboembed_prepared_v1_slot_write_tokens(slot, data.rows[0].data(), data.rows[1].data(),
                data.rows[2].data(), data.rows[0].size(), &error), error);
        } catch (...) { release(); throw; }
    }
    void release() noexcept {
        turboembed_prepared_v1_slot_release(slot); turboembed_prepared_v1_model_release(model);
        turboembed_prepared_v1_context_release(context);
    }
    ~Abi() { release(); }
    void execute() {
        te_result *result = nullptr;
        status(turboembed_prepared_v1_slot_execute(slot, &result, &error), error);
        status(turboembed_prepared_v1_result_release(result, &error), error);
    }
    std::vector<float> read() {
        te_result *result = nullptr;
        status(turboembed_prepared_v1_slot_execute(slot, &result, &error), error);
        std::vector<float> values(batch * 384);
        const auto code = turboembed_prepared_v1_result_read(result, values.data(), values.size(), &error);
        te_error release_error{};
        const auto released = turboembed_prepared_v1_result_release(result, &release_error);
        status(code, error); status(released, release_error); return values;
    }
    Json stats() {
        auto s = descriptor<te_slot_stats>();
        status(turboembed_prepared_v1_slot_stats(slot, &s, &error), error);
        return {{"executions", s.executions}, {"input_write_bytes", s.input_write_bytes},
            {"output_read_bytes", s.output_read_bytes}, {"owned_input_tensor_bytes", s.owned_input_bytes},
            {"owned_output_tensor_bytes", s.owned_output_bytes}};
    }
};
double quantile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    return values.at(static_cast<size_t>(q * (values.size() - 1)));
}
Json describe(const std::vector<double> &samples, double elapsed_seconds) {
    return {{"n", samples.size()}, {"p50_us", quantile(samples, .5)}, {"p99_us", quantile(samples, .99)},
        {"elapsed_s", elapsed_seconds}, {"requests_s", samples.size() / elapsed_seconds},
        {"p99_claim", samples.size() >= 1000 ? "descriptive_only" : "insufficient_samples"}, {"samples_us", samples}};
}
template<typename Engine> Json measure(Engine &engine, double seconds) {
    std::vector<double> samples; samples.reserve(10000);
    const auto begin = Clock::now(); auto end = begin;
    while (samples.size() < 10000 && std::chrono::duration<double>(end - begin).count() < seconds) {
        const auto start = Clock::now(); engine.execute(); end = Clock::now();
        samples.push_back(micros(end - start));
    }
    return describe(samples, std::chrono::duration<double>(end - begin).count());
}
template<typename Engine> Json allocations(Engine &engine) {
    new_calls = 0; new_bytes = 0; count_new = true;
    for (int i = 0; i < 10; ++i) { engine.execute(); }
    count_new = false;
    return {{"executions", 10}, {"observed_cxx_new_calls", new_calls.load()}, {"observed_cxx_new_bytes", new_bytes.load()}};
}
Json parity(Direct &native, Abi &abi) {
    native.execute(); const auto expected = native.read(); const auto actual = abi.read();
    double sum = 0; float maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        check(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite embedding");
        const float delta = std::abs(actual[i] - expected[i]); maximum = std::max(maximum, delta); sum += delta * delta;
    }
    const auto rmse = std::sqrt(sum / actual.size());
    check(maximum <= 5e-4f && rmse <= 1e-4, "direct/ABI numerical parity failed");
    return {{"max_abs", maximum}, {"rmse", rmse}, {"max_abs_gate", 5e-4}, {"rmse_gate", 1e-4}};
}
int main(int argc, char **argv) {
    try {
        check(argc == 7, "usage: prepared_benchmark BUNDLE BATCH SEQUENCE full|mixed SECONDS OUTPUT_JSON");
        const std::string dir = argv[1], mode = argv[4];
        const uint32_t batch = static_cast<uint32_t>(std::stoul(argv[2]));
        const uint32_t sequence = static_cast<uint32_t>(std::stoul(argv[3]));
        const double seconds = std::stod(argv[5]);
        check((batch == 1 || batch == 8 || batch == 32) && (sequence == 32 || sequence == 128 || sequence == 256), "shape outside pilot");
        check((mode == "full" || mode == "mixed") && std::isfinite(seconds) && seconds > 0 && seconds <= 10, "invalid bounded workload");
        Inputs data(dir, batch, sequence, mode == "mixed");
        Json result = {{"batch", batch}, {"sequence", sequence}, {"pattern", mode}, {"token_lengths", data.lengths},
            {"seconds_cap", seconds}, {"requests_cap", 10000}, {"warmup_each", 20}, {"repeats", Json::array()},
            {"runtime", ov::get_openvino_version().buildNumber}, {"profiling_during_timing", false}};
        const auto nstart = Clock::now(); Direct native(dir, data);
        result["native_load_compile_upload_ms"] = micros(Clock::now() - nstart) / 1000;
        const auto astart = Clock::now(); Abi abi(dir, data);
        result["abi_verify_load_compile_upload_ms"] = micros(Clock::now() - astart) / 1000;
        result["device"] = native.core.get_property("GPU.0", ov::device::full_name);
        result["parity"] = parity(native, abi);
        for (int i = 0; i < 20; ++i) { native.execute(); abi.execute(); }
        result["abi_before_timing"] = abi.stats();
        for (int repeat = 0; repeat < 3; ++repeat) {
            Json n, a;
            if (repeat % 2 == 0) { n = measure(native, seconds); a = measure(abi, seconds); }
            else { a = measure(abi, seconds); n = measure(native, seconds); }
            result["repeats"].push_back({{"native", n}, {"abi", a}, {"first", repeat % 2 ? "abi" : "native"},
                {"p50_ratio", a["p50_us"].get<double>() / n["p50_us"].get<double>()},
                {"throughput_ratio", a["requests_s"].get<double>() / n["requests_s"].get<double>()}});
        }
        result["abi_after_timing"] = abi.stats();
        result["native_cpp_allocations"] = allocations(native);
        result["abi_cpp_allocations"] = allocations(abi);
        // Separate profiling-enabled diagnostic, never mixed into timing samples.
        {
            Direct diagnostic(dir, data, true);
            for (int i = 0; i < 5; ++i) { diagnostic.execute(); }
            result["native_device_profile"] = diagnostic.device_profile();
        }
        if (batch == 8 && sequence == 128 && mode == "mixed") {
            Abi independent(dir, data);
            for (int i = 0; i < 20; ++i) { independent.execute(); }
            result["two_independent_contexts"] = Json::array();
            for (int repeat = 0; repeat < 3; ++repeat) {
                std::promise<void> ready; auto go = ready.get_future().share();
                auto one = std::async(std::launch::async, [&] { go.wait(); return measure(abi, seconds); });
                auto two = std::async(std::launch::async, [&] { go.wait(); return measure(independent, seconds); });
                ready.set_value();
                result["two_independent_contexts"].push_back({one.get(), two.get()});
            }
        }
        struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);
        result["process_peak_rss_kib"] = usage.ru_maxrss;
        std::ofstream output(argv[6]); output << result.dump() << '\n';
        check(static_cast<bool>(output), "cannot write sample receipt");
        std::cout << batch << 'x' << sequence << ' ' << mode << " PASS parity; ratios";
        for (const auto &item : result["repeats"]) { std::cout << ' ' << item["p50_ratio"]; }
        std::cout << '\n';
        return 0;
    } catch (const std::exception &e) { count_new = false; std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
