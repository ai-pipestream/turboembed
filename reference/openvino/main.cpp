// SPDX-License-Identifier: Apache-2.0
//
// The direct-native half of the openvino provider's matched benchmark
// (PLAN.md section 11): OpenVINO's C++ API alone, with no libturbo in the
// timed path. It reads the ONNX model the bundle names, runs the token rows
// `turbo-bench embed --dump-tokens` wrote (the same ids, masks and shapes
// libturbo's prepared-token path ran) through an infer request the way an
// OpenVINO user would (input tensors set per run, the hidden state read
// back, mean or CLS pooling and L2 on the host), and writes a receipt of
// kind `native` that `turbo-bench compare` reads.

#include <openvino/openvino.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;

struct Options {
    std::string tokens;
    std::string device = "CPU";
    std::string out;
    std::string commit;
    unsigned iters = 30;
    unsigned warmup = 5;
};

[[noreturn]] void die(const std::string &m) {
    std::fprintf(stderr, "error: %s\n", m.c_str());
    std::exit(2);
}

Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                die(std::string(name) + " needs a value");
            }
            return argv[++i];
        };
        if (a == "--tokens") {
            o.tokens = value("--tokens");
        } else if (a == "--device") {
            o.device = value("--device");
        } else if (a == "--out") {
            o.out = value("--out");
        } else if (a == "--commit") {
            o.commit = value("--commit");
        } else if (a == "--iters") {
            o.iters = static_cast<unsigned>(std::stoul(value("--iters")));
        } else if (a == "--warmup") {
            o.warmup = static_cast<unsigned>(std::stoul(value("--warmup")));
        } else {
            die("unknown flag " + a);
        }
    }
    if (o.tokens.empty()) {
        die("--tokens is required");
    }
    if (o.iters == 0) {
        die("--iters must be at least 1");
    }
    return o;
}

std::string run_cmd(const std::string &cmd) {
    std::string out;
    FILE *p = popen(cmd.c_str(), "r");
    if (p == nullptr) {
        die("cannot run " + cmd);
    }
    char buf[256];
    while (fgets(buf, sizeof buf, p) != nullptr) {
        out += buf;
    }
    if (pclose(p) != 0) {
        die(cmd + " failed");
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) {
        out.pop_back();
    }
    return out;
}

std::string today() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[16];
    std::strftime(buf, sizeof buf, "%Y-%m-%d", &tm);
    return buf;
}

json latency(std::vector<double> &samples, double rows, double tokens) {
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    auto pct = [&](double p) { return samples[std::min<size_t>(n - 1, static_cast<size_t>(std::llround((n - 1) * p)))]; };
    double total = 0;
    for (double s : samples) {
        total += s;
    }
    const double mean = total / static_cast<double>(n);
    json l;
    l["p50_ms"] = pct(0.5);
    if (n >= 100) {
        l["p99_ms"] = pct(0.99);
    }
    l["mean_ms"] = mean;
    l["min_ms"] = samples.front();
    l["max_ms"] = samples.back();
    l["rows_per_s"] = rows / (mean / 1e3);
    l["tokens_per_s"] = tokens / (mean / 1e3);
    l["iters"] = n;
    return l;
}

} // namespace

int main(int argc, char **argv) {
    const Options o = parse(argc, argv);
    std::ifstream in(o.tokens);
    if (!in.good()) {
        die("cannot open " + o.tokens);
    }
    json dump;
    in >> dump;
    if (!dump["artifacts"].contains("onnx")) {
        die("the token dump names no `onnx` artifact");
    }
    const std::string onnx = dump["artifacts"]["onnx"];
    const std::string pooling = dump.value("pooling", "");
    if (pooling != "mean" && pooling != "cls") {
        die("pooling `" + pooling + "` is not implemented by this reference (mean, cls)");
    }
    const std::string norm = dump.value("normalize", "");
    const bool normalize = norm == "l2";
    if (!normalize && norm != "none" && !norm.empty()) {
        die("normalize `" + norm + "` is not l2 or none");
    }

    ov::Core core;
    const std::string version = ov::get_openvino_version().buildNumber;
    std::string device_name;
    try {
        device_name = core.get_property(o.device, ov::device::full_name);
    } catch (const std::exception &e) {
        die("device " + o.device + ": " + e.what());
    }
    auto model = core.read_model(onnx);
    // The same properties the provider compiles with.
    const ov::AnyMap props = {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                              ov::hint::inference_precision(ov::element::f32)};
    auto compiled = core.compile_model(model, o.device, props);
    auto request = compiled.create_infer_request();
    bool has_types = false;
    for (const auto &input : compiled.inputs()) {
        if (input.get_any_name() == "token_type_ids") {
            has_types = true;
        }
    }
    const auto input_type = compiled.input("input_ids").get_element_type();

    json cells = json::array();
    for (const auto &cell : dump["cells"]) {
        const size_t b = cell["batch"].get<size_t>();
        const size_t s = cell["seq"].get<size_t>();
        const std::vector<int32_t> ids32 = cell["ids"].get<std::vector<int32_t>>();
        const std::vector<int32_t> mask32 = cell["mask"].get<std::vector<int32_t>>();
        if (ids32.size() != b * s || mask32.size() != b * s) {
            die("cell " + std::to_string(b) + "x" + std::to_string(s) + ": ids/mask do not hold batch*seq elements");
        }
        double live = 0;
        for (const auto &l : cell["lengths"]) {
            live += l.get<double>();
        }
        std::vector<int64_t> ids64(ids32.begin(), ids32.end());
        std::vector<int64_t> mask64(mask32.begin(), mask32.end());
        std::vector<int64_t> types64(b * s, 0);
        std::vector<int32_t> types32(b * s, 0);
        const ov::Shape shape{b, s};
        std::vector<float> out;
        auto run_once = [&]() {
            // An OpenVINO user's loop: set the input tensors, infer, read
            // the hidden state, pool and normalize on the host.
            if (input_type == ov::element::i64) {
                request.set_tensor("input_ids", ov::Tensor(ov::element::i64, shape, ids64.data()));
                request.set_tensor("attention_mask", ov::Tensor(ov::element::i64, shape, mask64.data()));
                if (has_types) {
                    request.set_tensor("token_type_ids", ov::Tensor(ov::element::i64, shape, types64.data()));
                }
            } else {
                std::vector<int32_t> ids_copy(ids32), mask_copy(mask32);
                request.set_tensor("input_ids", ov::Tensor(ov::element::i32, shape, ids_copy.data()));
                request.set_tensor("attention_mask", ov::Tensor(ov::element::i32, shape, mask_copy.data()));
                if (has_types) {
                    request.set_tensor("token_type_ids", ov::Tensor(ov::element::i32, shape, types32.data()));
                }
            }
            request.infer();
            const ov::Tensor hidden = request.get_output_tensor(0);
            const auto hs = hidden.get_shape();
            if (hs.size() != 3 || hs[0] != b || hs[1] != s) {
                die("output shape is not [batch, seq, hidden]");
            }
            const size_t h = hs[2];
            const float *data = hidden.data<const float>();
            out.assign(b * h, 0.0f);
            for (size_t r = 0; r < b; ++r) {
                float *row = out.data() + r * h;
                if (pooling == "cls") {
                    std::copy(data + r * s * h, data + r * s * h + h, row);
                } else {
                    float n = 0;
                    for (size_t t = 0; t < s; ++t) {
                        if (mask32[r * s + t] != 0) {
                            n += 1;
                            const float *src = data + (r * s + t) * h;
                            for (size_t k = 0; k < h; ++k) {
                                row[k] += src[k];
                            }
                        }
                    }
                    if (n > 0) {
                        for (size_t k = 0; k < h; ++k) {
                            row[k] /= n;
                        }
                    }
                }
                if (normalize) {
                    float sum = 0;
                    for (size_t k = 0; k < h; ++k) {
                        sum += row[k] * row[k];
                    }
                    const float nrm = std::sqrt(sum);
                    if (nrm > 0) {
                        for (size_t k = 0; k < h; ++k) {
                            row[k] /= nrm;
                        }
                    }
                }
            }
        };
        // At least o.warmup iterations and at least 0.5 s, as turbo-bench
        // warms up, so the device leaves its idle clock state first.
        const auto w0 = clock_type::now();
        for (unsigned i = 0; i < o.warmup || clock_type::now() - w0 < std::chrono::milliseconds(500); ++i) {
            run_once();
        }
        std::vector<double> samples;
        samples.reserve(o.iters);
        for (unsigned i = 0; i < o.iters; ++i) {
            const auto t0 = clock_type::now();
            run_once();
            samples.push_back(std::chrono::duration<double, std::milli>(clock_type::now() - t0).count());
        }
        json lat = latency(samples, static_cast<double>(b), live);
        std::fprintf(stderr, "embed batch %2zu seq %3zu: tokens p50 %.3f ms (%.0f rows/s)\n", b, s,
                     lat["p50_ms"].get<double>(), lat["rows_per_s"].get<double>());
        json c;
        c["batch"] = b;
        c["seq"] = s;
        c["live_tokens_per_row"] = live / static_cast<double>(b);
        c["token_count_source"] = "tokenizer";
        c["text_path"] = lat;
        c["prepared_tokens_path"] = lat;
        c["per_run"] = {{"host_allocs", nullptr}, {"provider_allocs", nullptr}};
        cells.push_back(c);
    }

    std::string commit = o.commit;
    if (commit.empty()) {
        die("pass --commit <sha>: this program is built by cmake and does not capture the commit itself");
    }
    json r;
    r["receipt_version"] = 1;
    r["kind"] = "native";
    r["date"] = today();
    r["machine"] = {{"hostname", run_cmd("uname -n")}, {"os", run_cmd("uname -sr")}, {"arch", run_cmd("uname -m")}};
    r["commit"] = commit;
    r["provider"] = {{"id", "openvino"}, {"version", "reference"}, {"runtime_version", "OpenVINO " + version + " C++ API"}, {"driver_version", ""}};
    r["device"] = {{"name", device_name}, {"kind", o.device == "CPU" ? "Cpu" : "Gpu"}, {"ordinal", 0}, {"caps", "0x0"}, {"memory_total", 0}};
    r["bundle"] = dump["bundle_id"];
    r["embed"] = cells;
    r["native_reference"] = "this is the native side: OpenVINO C++ API (" + o.device + ") on the token rows of " + o.tokens +
                            "; bundle identity copied from that dump; host-side " + pooling + " pooling and " +
                            (normalize ? "l2" : "no") + " normalization; no session counters";
    const std::string text = r.dump(2) + "\n";
    if (o.out.empty()) {
        std::cout << text;
    } else {
        std::ofstream f(o.out);
        f << text;
        std::fprintf(stderr, "receipt written to %s\n", o.out.c_str());
    }
    return 0;
}
