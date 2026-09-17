// SPDX-License-Identifier: Apache-2.0
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include "turboembed_prepared.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;
void check(bool value, const char *why) {
    if (!value) { throw std::runtime_error(why); }
}
void ok(uint32_t status, const te_error &error) {
    if (status != TE_OK) { throw std::runtime_error(std::to_string(status) + ": " + error.message); }
}
template<typename T> T descriptor() {
    T value{}; value.struct_size = sizeof(T); value.version = TE_PREPARED_VERSION; return value;
}
struct Context {
    te_context *p = nullptr;
    explicit Context(uint32_t device) {
        auto options = descriptor<te_context_options>(); options.device = device;
        te_error error{}; ok(turboembed_prepared_v1_context_create(&options, &p, &error), error);
    }
    ~Context() { turboembed_prepared_v1_context_release(p); }
};
struct Model {
    te_model *p = nullptr;
    Model(const Context &ctx, const std::string &path) {
        te_error error{}; ok(turboembed_prepared_v1_model_load(ctx.p, path.data(), path.size(), &p, &error), error);
    }
    ~Model() { turboembed_prepared_v1_model_release(p); }
};
struct Slot {
    te_slot *p = nullptr;
    explicit Slot(const Model &model, uint32_t batch = 1) {
        auto options = descriptor<te_slot_options>(); options.batch = batch; options.sequence_length = 32;
        te_error error{}; ok(turboembed_prepared_v1_slot_create(model.p, &options, &p, &error), error);
    }
    ~Slot() { turboembed_prepared_v1_slot_release(p); }
};
struct Result {
    te_result *p = nullptr;
    explicit Result(const Slot &slot) {
        te_error error{}; ok(turboembed_prepared_v1_slot_execute(slot.p, &p, &error), error);
    }
    ~Result() { turboembed_prepared_v1_result_release(p, nullptr); }
    std::vector<float> read() const {
        auto info = descriptor<te_result_info>(); te_error error{};
        ok(turboembed_prepared_v1_result_info(p, &info, &error), error);
        std::vector<float> values(info.batch * info.dimension);
        ok(turboembed_prepared_v1_result_read(p, values.data(), values.size(), &error), error);
        return values;
    }
};
void parity(const std::vector<float> &a, const std::vector<float> &b, float max_limit = 5e-4f, double rmse_limit = 1e-4) {
    check(a.size() == b.size(), "vector size mismatch");
    float maximum = 0; double squared = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        check(std::isfinite(a[i]) && std::isfinite(b[i]), "non-finite output");
        const float delta = std::abs(a[i] - b[i]); maximum = std::max(maximum, delta); squared += delta * delta;
    }
    const double rmse = std::sqrt(squared / a.size());
    check(maximum <= max_limit && rmse <= rmse_limit, "numerical parity failed");
    std::cout << "parity max_abs=" << maximum << " rmse=" << rmse << '\n';
}
void text(Slot &slot, const std::string &value) {
    te_text input{value.data(), value.size()}; te_error error{};
    ok(turboembed_prepared_v1_slot_write_text(slot.p, &input, 1, &error), error);
}
std::vector<float> gpu_consumer(const Result &result) {
    auto view = descriptor<te_opencl_view>(); te_error error{};
    ok(turboembed_prepared_v1_result_opencl(result.p, &view, &error), error);
    cl::Context context(reinterpret_cast<cl_context>(view.context), true);
    cl::CommandQueue queue(reinterpret_cast<cl_command_queue>(view.queue), true);
    cl::Buffer input(reinterpret_cast<cl_mem>(view.buffer), true);
    cl::Buffer output(context, CL_MEM_READ_WRITE, view.byte_size);
    cl::Program program(context, "__kernel void twice(__global const float* x, __global float* y) { size_t i=get_global_id(0); y[i]=2*x[i]; }");
    program.build(); cl::Kernel kernel(program, "twice"); kernel.setArg(0, input); kernel.setArg(1, output);
    const size_t count = view.byte_size / sizeof(float);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(count));
    // Release must wait for the queued consumer before allowing slot reuse.
    ok(turboembed_prepared_v1_result_release(result.p, &error), error);
    std::vector<float> values(count);
    queue.enqueueReadBuffer(output, CL_TRUE, 0, view.byte_size, values.data());
    for (auto &value : values) { value /= 2; }
    return values;
}
void discovery() {
    te_error error{};
    uint32_t count = 0;
    check(turboembed_prepared_v1_device_count(nullptr, &error) == TE_INVALID_ARGUMENT, "null count output accepted");
    ok(turboembed_prepared_v1_device_count(&count, &error), error);
    check(count >= 2, "this host must enumerate both the Intel GPU and CPU");
    auto bad = descriptor<te_device_info>(); --bad.struct_size;
    check(turboembed_prepared_v1_device_info(0, &bad, &error) == TE_ABI_MISMATCH, "bad device descriptor accepted");
    auto out_of_range = descriptor<te_device_info>();
    check(turboembed_prepared_v1_device_info(count, &out_of_range, &error) == TE_NOT_FOUND, "out-of-range device index accepted");
    bool saw_gpu = false;
    for (uint32_t index = 0; index < count; ++index) {
        auto info = descriptor<te_device_info>();
        ok(turboembed_prepared_v1_device_info(index, &info, &error), error);
        check(info.device == TE_DEVICE_OPENVINO_GPU || info.device == TE_DEVICE_OPENVINO_CPU, "unknown discovered device");
        check(info.device_name[0] != 0 && info.runtime_version[0] != 0, "discovered device is missing identity");
        const bool gpu = info.device == TE_DEVICE_OPENVINO_GPU;
        saw_gpu |= gpu;
        check((index + 1 == count) == !gpu, "GPUs must precede the trailing CPU entry");
        const uint64_t expected = TE_CAP_TEXT | TE_CAP_PREPARED_I32 | TE_CAP_HOST_READ | (gpu ? TE_CAP_OPENCL_RESULT : 0);
        check(info.capabilities == expected, "wrong discovered capabilities");
        check(gpu == (info.driver_version[0] != 0), "driver version must be reported for GPUs only");
        // A context selected from the discovered identity must match it.
        auto options = descriptor<te_context_options>();
        options.device = info.device; options.ordinal = info.ordinal;
        te_context *raw = nullptr;
        ok(turboembed_prepared_v1_context_create(&options, &raw, &error), error);
        auto resolved = descriptor<te_context_info>();
        const auto status = turboembed_prepared_v1_context_info(raw, &resolved, &error);
        turboembed_prepared_v1_context_release(raw);
        ok(status, error);
        check(resolved.device == info.device && resolved.ordinal == info.ordinal &&
              std::strcmp(resolved.device_name, info.device_name) == 0 &&
              std::strcmp(resolved.runtime_version, info.runtime_version) == 0 &&
              std::strcmp(resolved.driver_version, info.driver_version) == 0 &&
              resolved.capabilities == info.capabilities, "context identity disagrees with discovery");
    }
    check(saw_gpu, "discovery listed no Intel GPU");
    std::cout << "PASS device discovery identity and selection agreement\n";
}

void run(const std::string &bundle, uint32_t device, const std::vector<float> &reference) {
    Context ctx(device); Model model(ctx, bundle); Slot slot(model);
    te_error error{};
    auto info = descriptor<te_context_info>();
    ok(turboembed_prepared_v1_context_info(ctx.p, &info, &error), error);
    std::cout << "device=" << info.device_name << " runtime=" << info.runtime_version << '\n';
    auto model_info = descriptor<te_model_info>();
    ok(turboembed_prepared_v1_model_info(model.p, &model_info, &error), error);
    check(model_info.dimension == 384 && model_info.normalized == 1, "wrong model metadata");
    auto bad_shape = descriptor<te_slot_options>(); bad_shape.batch = UINT32_MAX; bad_shape.sequence_length = UINT32_MAX;
    te_slot *bad_slot = reinterpret_cast<te_slot *>(1);
    check(turboembed_prepared_v1_slot_create(model.p, &bad_shape, &bad_slot, &error) == TE_INVALID_ARGUMENT && !bad_slot,
          "oversized shape must fail and clear output");
    te_result *invalid = reinterpret_cast<te_result *>(1);
    check(turboembed_prepared_v1_slot_execute(slot.p, &invalid, &error) == TE_INVALID_ARGUMENT && !invalid,
          "execution before write must fail and clear output");

    nlohmann::json tokenizer; std::ifstream(bundle + "/tokenizer.json") >> tokenizer;
    const auto &vocab = tokenizer.at("model").at("vocab");
    std::vector<int32_t> ids(32, vocab.at("[PAD]").get<int32_t>()), masks(32, 0);
    ids[0] = vocab.at("[CLS]"); ids[1] = vocab.at("hello"); ids[2] = vocab.at("world"); ids[3] = vocab.at("[SEP]");
    std::fill_n(masks.begin(), 4, 1);
    ok(turboembed_prepared_v1_slot_write_tokens(slot.p, ids.data(), masks.data(), nullptr, ids.size(), &error), error);
    std::vector<float> prepared;
    {
        Result result(slot); prepared = result.read(); parity(reference, prepared);
        check(turboembed_prepared_v1_slot_execute(slot.p, &invalid, &error) == TE_BUSY && !invalid, "live result must prevent reuse");
        check(turboembed_prepared_v1_slot_write_tokens(slot.p, nullptr, nullptr, nullptr, 0, &error) == TE_BUSY,
              "live result must prevent input mutation");
        check(turboembed_prepared_v1_result_read(result.p, prepared.data(), 1, &error) == TE_INVALID_ARGUMENT, "small output accepted");
        auto bad = descriptor<te_result_info>(); --bad.struct_size;
        check(turboembed_prepared_v1_result_info(result.p, &bad, &error) == TE_ABI_MISMATCH, "bad descriptor accepted");
    }
    // Repeated execution reuses the uploaded input without another write.
    for (int i = 0; i < 3; ++i) { Result repeated(slot); parity(prepared, repeated.read(), 1e-6f, 1e-7); }
    text(slot, "hello world");
    { Result result(slot); parity(prepared, result.read(), 1e-6f, 1e-7); }

    ids[4] = static_cast<int32_t>(model_info.vocab_size);
    check(turboembed_prepared_v1_slot_write_tokens(slot.p, ids.data(), masks.data(), nullptr, ids.size(), &error) == TE_INVALID_ARGUMENT,
          "out-of-range token accepted");
    check(turboembed_prepared_v1_slot_execute(slot.p, &invalid, &error) == TE_INVALID_ARGUMENT, "invalid write left old input live");
    const std::string malformed(1, '\xff'); te_text bad_text{malformed.data(), malformed.size()};
    check(turboembed_prepared_v1_slot_write_text(slot.p, &bad_text, 1, &error) == TE_INVALID_ARGUMENT, "invalid UTF-8 accepted");
    for (const std::string &value : {std::string{}, std::string("hello\0world", 11), std::string(u8"Café 日本語")}) {
        text(slot, value); Result result(slot); const auto values = result.read();
        check(std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }), "text produced non-finite output");
    }
    te_text empty{nullptr, 0}; ok(turboembed_prepared_v1_slot_write_text(slot.p, &empty, 1, &error), error);
    { Result result(slot); (void)result.read(); }

    Slot other(model); text(slot, "hello world"); text(other, "hello world");
    auto first = std::async(std::launch::async, [&] { Result result(slot); return result.read(); });
    auto second = std::async(std::launch::async, [&] { Result result(other); return result.read(); });
    parity(first.get(), second.get());

    Slot batch(model, 2);
    const te_text rows[] = {{"hello world", 11}, {nullptr, 0}};
    ok(turboembed_prepared_v1_slot_write_text(batch.p, rows, 2, &error), error);
    {
        Result result(batch); const auto values = result.read();
        check(values.size() == 768, "wrong batch output layout");
        parity(prepared, std::vector<float>(values.begin(), values.begin() + 384));
        text(other, ""); Result empty_result(other);
        parity(empty_result.read(), std::vector<float>(values.begin() + 384, values.end()));
    }

    if (device == TE_DEVICE_OPENVINO_GPU) {
        Result result(slot); auto stats = descriptor<te_slot_stats>();
        ok(turboembed_prepared_v1_slot_stats(slot.p, &stats, &error), error);
        const auto reads = stats.output_read_bytes;
        parity(reference, gpu_consumer(result)); result.p = nullptr;
        ok(turboembed_prepared_v1_slot_stats(slot.p, &stats, &error), error);
        check(stats.output_read_bytes == reads, "GPU consumer forced ABI readback");
        { Result reused(slot); parity(prepared, reused.read()); }
    } else {
        Result result(slot); auto view = descriptor<te_opencl_view>();
        check(turboembed_prepared_v1_result_opencl(result.p, &view, &error) == TE_NOT_IMPLEMENTED, "CPU exposed a fake GPU resource");
    }

    // A live result independently retains slot, model, context and verified bytes.
    Result retained(slot);
    turboembed_prepared_v1_slot_release(slot.p); slot.p = nullptr;
    turboembed_prepared_v1_model_release(model.p); model.p = nullptr;
    turboembed_prepared_v1_context_release(ctx.p); ctx.p = nullptr;
    parity(prepared, retained.read());
    std::cout << "PASS input, lease, text, concurrent slots, retained result contracts\n";
}

void integrity(const std::string &bundle) {
    Context ctx(TE_DEVICE_OPENVINO_CPU); te_error error{}; te_model *out = nullptr;
    const auto path = fs::temp_directory_path() / ("turboembed-integrity-" + std::to_string(getpid()));
    check(!fs::exists(path), "temporary test path already exists");
    fs::copy(bundle, path, fs::copy_options::recursive);
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); } } cleanup{path};
    const auto name = path.string();
    Model captured(ctx, name);
    { std::ofstream(path / "tokenizer.json", std::ios::app) << " "; }
    check(turboembed_prepared_v1_model_load(ctx.p, name.data(), name.size(), &out, &error) == TE_INTEGRITY_ERROR && !out,
          "corrupt bundle accepted");
    Slot slot(captured); text(slot, "hello world"); { Result result(slot); (void)result.read(); }
    fs::copy_file(fs::path(bundle) / "tokenizer.json", path / "tokenizer.json", fs::copy_options::overwrite_existing);
    nlohmann::json manifest; std::ifstream(fs::path(bundle) / "bundle.json") >> manifest;
    auto modified = manifest; modified["contract"]["unknown_option"] = true;
    { std::ofstream(path / "bundle.json") << modified; }
    check(turboembed_prepared_v1_model_load(ctx.p, name.data(), name.size(), &out, &error) == TE_INTEGRITY_ERROR,
          "unknown contract option accepted");
    modified = manifest; modified["model"]["id"] = std::string("model\0suffix", 12);
    { std::ofstream(path / "bundle.json") << modified; }
    check(turboembed_prepared_v1_model_load(ctx.p, name.data(), name.size(), &out, &error) == TE_INVALID_ARGUMENT,
          "NUL in metadata accepted");
    { std::ofstream(path / "bundle.json") << "{\"schema_version\":1,\"schema_version\":1}"; }
    check(turboembed_prepared_v1_model_load(ctx.p, name.data(), name.size(), &out, &error) == TE_INTEGRITY_ERROR,
          "duplicate manifest keys accepted");
    const std::string missing = name + "/missing";
    check(turboembed_prepared_v1_model_load(ctx.p, missing.data(), missing.size(), &out, &error) == TE_NOT_FOUND,
          "missing bundle returned wrong error");
    std::cout << "PASS bundle integrity and captured-file ownership\n";
}
int main(int argc, char **argv) {
    try {
        check(argc == 2, "usage: prepared_contract_test BUNDLE_DIR (requires Intel GPU and CPU)");
        const std::string bundle = argv[1];
        discovery();
        Context cpu(TE_DEVICE_OPENVINO_CPU); Model model(cpu, bundle); Slot slot(model);
        text(slot, "hello world"); Result reference(slot); const auto expected = reference.read();
        run(bundle, TE_DEVICE_OPENVINO_CPU, expected);
        run(bundle, TE_DEVICE_OPENVINO_GPU, expected);
        integrity(bundle);
        std::cout << "PASS prepared SDK contracts on explicit CPU and Intel GPU\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
