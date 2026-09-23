// SPDX-License-Identifier: Apache-2.0
//
// The direct-native half of the openvino provider's matched benchmark
// (PLAN.md section 11): OpenVINO's C++ API alone, with no libturbo in the
// timed path. It reads the ONNX model the bundle names, runs the token rows
// `turbo-bench embed|rerank|classify|token-classify --dump-tokens` wrote
// (the same ids, masks, segment ids and shapes libturbo's prepared-token
// path ran) through an infer request the way an OpenVINO user would (input
// tensors set per run, the model's output read back, and the rest done on
// the host: mean or CLS pooling and L2 for embeddings, the bundle's
// activation on the reranker logit, softmax or sigmoid over a classifier's
// logits, softmax per token for token classification), and writes a
// receipt of kind `native` that `turbo-bench compare` reads.

#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/opsets/opset13.hpp>

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
    // Attribution knobs, off by default (the plain user loop): compile the
    // model reshaped to the cell's static [batch, seq], and pool and
    // normalize inside the graph, the two choices the provider makes.
    bool static_shape = false;
    bool fuse = false;
    // A third: the inputs declared i32 through the pre-processor, as the
    // provider declares them (the ONNX graph's are i64).
    bool i32 = false;
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
        } else if (a == "--static") {
            o.static_shape = true;
        } else if (a == "--fuse") {
            o.fuse = true;
        } else if (a == "--i32") {
            o.i32 = true;
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

/// What a plain OpenVINO user computes from a task model's logits on the
/// host, into `out`: the reranker's `[batch]` scores (sigmoid or the logit
/// itself), a classifier's `[batch, labels]` scores (softmax, sigmoid, or
/// the logits), a token classifier's `[batch, seq, labels]` per-token
/// softmax.
void scores(const std::string &task, const std::string &activation, const ov::Tensor &logits, size_t b, size_t s,
            std::vector<float> &out) {
    const auto shape = logits.get_shape();
    const float *x = logits.data<const float>();
    auto softmax = [](const float *in, float *dst, size_t n) {
        float m = in[0];
        for (size_t k = 1; k < n; ++k) {
            m = std::max(m, in[k]);
        }
        float sum = 0;
        for (size_t k = 0; k < n; ++k) {
            dst[k] = std::exp(in[k] - m);
            sum += dst[k];
        }
        for (size_t k = 0; k < n; ++k) {
            dst[k] /= sum;
        }
    };
    auto sigmoid = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
    if (task == "rerank") {
        if (shape.empty() || shape[0] != b || logits.get_size() != b) {
            die("reranker output is not one logit per row");
        }
        out.assign(x, x + b);
        if (activation == "sigmoid") {
            for (float &v : out) {
                v = sigmoid(v);
            }
        }
        return;
    }
    if (task == "classify") {
        if (shape.size() != 2 || shape[0] != b) {
            die("classifier output shape is not [batch, labels]");
        }
        const size_t n = shape[1];
        out.resize(b * n);
        for (size_t r = 0; r < b; ++r) {
            if (activation == "softmax") {
                softmax(x + r * n, out.data() + r * n, n);
            } else {
                for (size_t k = 0; k < n; ++k) {
                    out[r * n + k] = activation == "sigmoid" ? sigmoid(x[r * n + k]) : x[r * n + k];
                }
            }
        }
        return;
    }
    if (shape.size() != 3 || shape[0] != b || shape[1] != s) {
        die("token classifier output shape is not [batch, seq, labels]");
    }
    const size_t n = shape[2];
    out.resize(b * s * n);
    for (size_t t = 0; t < b * s; ++t) {
        softmax(x + t * n, out.data() + t * n, n);
    }
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
    const std::string task = dump.value("task", "embed");
    if (task != "embed" && task != "rerank" && task != "classify" && task != "token_classify") {
        die("task `" + task + "` is not embed, rerank, classify or token_classify");
    }
    const bool embed = task == "embed";
    const std::string pooling = dump.value("pooling", "");
    if (embed && pooling != "mean" && pooling != "cls") {
        die("pooling `" + pooling + "` is not implemented by this reference (mean, cls)");
    }
    const std::string norm = dump.value("normalize", "");
    const bool normalize = embed && norm == "l2";
    if (embed && !normalize && norm != "none" && !norm.empty()) {
        die("normalize `" + norm + "` is not l2 or none");
    }
    // The activation the provider fuses, applied here on the host: the
    // bundle's, or sigmoid for a reranker and softmax for a classifier when
    // it names none; token classification is always a per-token softmax.
    std::string activation = dump.value("activation", "");
    if (task == "rerank" && activation.empty()) {
        activation = "sigmoid";
    } else if (task == "classify" && activation.empty()) {
        activation = "softmax";
    } else if (task == "token_classify") {
        activation = "softmax";
    }
    if (task == "rerank" && activation != "sigmoid" && activation != "none") {
        die("reranker activation `" + activation + "` is not sigmoid or none");
    }
    if (task == "classify" && activation != "softmax" && activation != "sigmoid" && activation != "none") {
        die("classifier activation `" + activation + "` is not softmax, sigmoid or none");
    }
    if (!embed && o.fuse) {
        die("--fuse fuses pooling and normalization, which only an embedding model has");
    }
    if (task == "rerank" && dump["cells"].size() != 1) {
        die("a rerank dump holds one cell");
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
    // The provider's graph: reshaped to the cell and with the pooling and
    // the normalization appended; used only with --static or --fuse.
    auto shaped = [&](size_t b, size_t s) {
        auto m = model->clone();
        if (o.i32) {
            ov::preprocess::PrePostProcessor ppp(m);
            for (const auto &input : m->inputs()) {
                ppp.input(input.get_any_name()).tensor().set_element_type(ov::element::i32);
            }
            m = ppp.build();
        }
        if (o.static_shape) {
            std::map<std::string, ov::PartialShape> shapes;
            for (const auto &input : m->inputs()) {
                shapes[input.get_any_name()] = ov::PartialShape{static_cast<int64_t>(b), static_cast<int64_t>(s)};
            }
            m->reshape(shapes);
        }
        if (o.fuse) {
            namespace op = ov::opset13;
            ov::Output<ov::Node> mask;
            for (const auto &input : m->inputs()) {
                if (input.get_any_name() == "attention_mask") {
                    mask = input;
                }
            }
            const auto out = m->get_results().at(0)->input_value(0);
            const auto axis1 = op::Constant::create(ov::element::i64, ov::Shape{1}, {1});
            const auto axis2 = op::Constant::create(ov::element::i64, ov::Shape{1}, {2});
            const auto fmask = std::make_shared<op::Convert>(mask, ov::element::f32);
            ov::Output<ov::Node> y;
            if (pooling == "cls") {
                y = std::make_shared<op::Gather>(out, op::Constant::create(ov::element::i64, ov::Shape{}, {0}), axis1);
            } else {
                const auto expanded = std::make_shared<op::Unsqueeze>(fmask, axis2);
                const auto sum = std::make_shared<op::ReduceSum>(std::make_shared<op::Multiply>(out, expanded), axis1, false);
                const auto count = std::make_shared<op::ReduceSum>(fmask, axis1, true);
                const auto denom = std::make_shared<op::Maximum>(count, op::Constant::create(ov::element::f32, ov::Shape{}, {1.0f}));
                y = std::make_shared<op::Divide>(sum, denom);
            }
            if (normalize) {
                const auto sq = std::make_shared<op::Multiply>(y, y);
                const auto norm = std::make_shared<op::Sqrt>(std::make_shared<op::ReduceSum>(sq, axis1, true));
                const auto d = std::make_shared<op::Maximum>(norm, op::Constant::create(ov::element::f32, ov::Shape{}, {1e-12f}));
                y = std::make_shared<op::Divide>(y, d);
            }
            m = std::make_shared<ov::Model>(ov::OutputVector{y}, m->get_parameters(), "fused");
        }
        return m;
    };
    auto compiled = core.compile_model(model, o.device, props);
    auto request = compiled.create_infer_request();
    bool has_types = false;
    for (const auto &input : compiled.inputs()) {
        if (input.get_any_name() == "token_type_ids") {
            has_types = true;
        }
    }

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
        std::vector<int32_t> types32(b * s, 0);
        if (cell.contains("types")) {
            types32 = cell["types"].get<std::vector<int32_t>>();
            if (types32.size() != b * s) {
                die("cell " + std::to_string(b) + "x" + std::to_string(s) + ": types do not hold batch*seq elements");
            }
        }
        std::vector<int64_t> ids64(ids32.begin(), ids32.end());
        std::vector<int64_t> mask64(mask32.begin(), mask32.end());
        std::vector<int64_t> types64(types32.begin(), types32.end());
        const ov::Shape shape{b, s};
        std::vector<float> out;
        if (o.static_shape || o.fuse || o.i32) {
            compiled = core.compile_model(shaped(b, s), o.device, props);
            request = compiled.create_infer_request();
        }
        const auto input_type = compiled.input("input_ids").get_element_type();
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
            if (!embed) {
                scores(task, activation, hidden, b, s, out);
                return;
            }
            if (o.fuse) {
                // Pooled and normalized in the graph: the result is [batch, hidden].
                if (hs.size() != 2 || hs[0] != b) {
                    die("fused output shape is not [batch, hidden]");
                }
                out.assign(hidden.data<const float>(), hidden.data<const float>() + b * hs[1]);
                return;
            }
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
        std::fprintf(stderr, "%s batch %2zu seq %3zu: tokens p50 %.3f ms (%.0f rows/s)\n", task.c_str(), b, s,
                     lat["p50_ms"].get<double>(), lat["rows_per_s"].get<double>());
        if (task == "rerank") {
            // A rerank cell counts documents, not tokens, as turbo-bench's does.
            lat.erase("tokens_per_s");
            json c;
            c["docs"] = b;
            c["seq"] = s;
            c["text_path"] = lat;
            c["prepared_tokens_path"] = lat;
            c["per_run"] = {{"host_allocs", nullptr}, {"provider_allocs", nullptr}};
            cells.push_back(c);
            continue;
        }
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
    if (task == "rerank") {
        r["rerank"] = cells.at(0);
    } else {
        r[task] = cells;
    }
    const std::string host_work =
        embed ? std::string(o.fuse ? "pooling and normalization in the graph; " : "host-side ") + pooling + " pooling and " +
                    (normalize ? "l2" : "no") + " normalization"
        : task == "token_classify" ? std::string("host-side softmax over each token's label logits")
        : task == "rerank"         ? "host-side " + activation + " activation on the logit"
                                   : "host-side " + activation + " over the label logits";
    r["native_reference"] = "this is the native side: OpenVINO C++ API (" + o.device + ") on the token rows of " + o.tokens +
                            "; bundle identity copied from that dump; " + std::string(o.static_shape ? "static [batch, seq] shape; " : "") +
                            std::string(o.i32 ? "i32 inputs; " : "") + host_work + "; no session counters";
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
