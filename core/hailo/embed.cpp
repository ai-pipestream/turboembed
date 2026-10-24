/* SPDX-License-Identifier: Apache-2.0
 *
 * embed.h's model and session. The attention bias is the HEF's own
 * convention and never leaves this file: 0 for a key the mask keeps and
 * MASKED for one it drops, the same value for every head and every query,
 * laid out [query, head * key] as the HEF takes it.
 */

#include "embed.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <cerrno>

namespace turbo_hailo {

namespace {

/* The bias for a dropped key: the value the HEF was calibrated with.
 * exp(-100) underflows to 0 in the softmax, as the upstream model's
 * dtype minimum does. */
constexpr float MASKED = -100.f;

/* The batch the model is configured with: on a HEF of several contexts the
 * device runs up to this many frames through each context before it loads
 * the next, so the cost of a switch is shared by the burst. With HailoRT's
 * scheduler it is the most frames run at once, not a number a run waits
 * for: a single frame runs alone. 32 is the largest HailoRT configures
 * all-MiniLM-L6-v2's HEF with on the Hailo-10H; docs/hailo.md has what
 * each burst gives. */
constexpr uint16_t BURST = 32;

/* How long a frame may take before the run gives up on the device. */
constexpr std::chrono::milliseconds FRAME_TIMEOUT{10000};

Failure fail(int32_t code, std::string message, uint32_t field = 0) {
    Failure f;
    f.code = code;
    f.message = std::move(message);
    f.field = field;
    return f;
}

Failure hailort_failed(hailo_status s, const std::string &what) {
    const char *m = hailo_get_status_message(s);
    return fail(TURBO_E_RUNTIME, what + ": " + (m ? m : "unknown status") + " (" + std::to_string((int)s) + ")");
}

/* The longest frame timeout TURBO_HAILO_FRAME_TIMEOUT_MS may set: ten
 * minutes, far inside what a deadline on the steady clock can hold. */
constexpr unsigned long MAX_FRAME_TIMEOUT_MS = 600000;

/* FRAME_TIMEOUT, or TURBO_HAILO_FRAME_TIMEOUT_MS, read when a session is
 * made: a whole number of ms from 0 to MAX_FRAME_TIMEOUT_MS, and anything
 * else refuses the session. A test sets 0, so a frame genuinely does not
 * come back in time on a device that answers, to reach what a device that
 * stopped answering leaves (core/tests/hailo_release.rs). */
Failure frame_timeout(std::chrono::milliseconds &out) {
    const char *v = std::getenv("TURBO_HAILO_FRAME_TIMEOUT_MS");
    out = FRAME_TIMEOUT;
    if (!v) return {};
    bool digits = *v != '\0';
    for (const char *c = v; *c; c++) digits = digits && *c >= '0' && *c <= '9';
    errno = 0;
    char *end = nullptr;
    const unsigned long ms = digits ? std::strtoul(v, &end, 10) : 0;
    if (!digits || errno == ERANGE || *end || ms > MAX_FRAME_TIMEOUT_MS)
        return fail(TURBO_E_INVALID_ARGUMENT, std::string("TURBO_HAILO_FRAME_TIMEOUT_MS is ") + v +
                                                  ": a whole number of ms from 0 to " +
                                                  std::to_string(MAX_FRAME_TIMEOUT_MS));
    out = std::chrono::milliseconds{(long long)ms};
    return {};
}

/* Frame sets kept by a model and not yet reclaimed (retired_frames). */
std::atomic<uint32_t> retired_count{0};

/* A stream's layout and its one quantization, or why the HEF's stream is
 * not one this model can use. */
template <typename S> Failure describe(S &stream, const std::string &name, size_t elements, Stream &out) {
    out.name = name;
    out.frame_bytes = stream.get_frame_size();
    if (out.frame_bytes != elements && out.frame_bytes != 2 * elements)
        return fail(TURBO_E_BUNDLE_INVALID, "hef stream " + name + ": " + std::to_string(out.frame_bytes) +
                                                " bytes a frame, not " + std::to_string(elements) +
                                                " elements of 8 or 16 bits");
    out.element_bytes = out.frame_bytes / elements;
    const auto q = stream.get_quant_infos();
    if (q.size() != 1)
        return fail(TURBO_E_BUNDLE_INVALID,
                    "hef stream " + name + ": " + std::to_string(q.size()) + " quantizations, not one per stream");
    out.scale = q[0].qp_scale;
    out.zero_point = q[0].qp_zp;
    if (!(out.scale > 0.f))
        return fail(TURBO_E_BUNDLE_INVALID, "hef stream " + name + ": quantization scale " + std::to_string(out.scale));
    return {};
}

/* x in the stream's integer type, rounded to nearest and held to its range. */
inline void put(const Stream &s, uint8_t *frame, size_t i, float x) {
    const float hi = s.element_bytes == 2 ? 65535.f : 255.f;
    const float v = std::fmin(hi, std::fmax(0.f, std::nearbyint(x / s.scale + s.zero_point)));
    if (s.element_bytes == 2) {
        const uint16_t u = (uint16_t)v;
        std::memcpy(frame + 2 * i, &u, 2);
    } else {
        frame[i] = (uint8_t)v;
    }
}

inline float get(const Stream &s, const uint8_t *frame, size_t i) {
    float q;
    if (s.element_bytes == 2) {
        uint16_t u;
        std::memcpy(&u, frame + 2 * i, 2);
        q = u;
    } else {
        q = frame[i];
    }
    return (q - s.zero_point) * s.scale;
}

} // namespace

Failure Model::load(hailort::VDevice &vdevice, const ModelDesc &desc, std::unique_ptr<Model> &out) {
    std::unique_ptr<Model> m(new Model());
    m->desc_ = desc;
    auto infer = vdevice.create_infer_model(hailort::MemoryView::create_const(desc.hef, desc.hef_bytes));
    if (!infer) return hailort_failed(infer.status(), "create_infer_model");
    m->infer_ = infer.release();
    const auto &inputs = m->infer_->get_input_names();
    const auto &outputs = m->infer_->get_output_names();
    if (inputs.size() != 2 || outputs.size() != 1)
        return fail(TURBO_E_BUNDLE_INVALID, "the hef has " + std::to_string(inputs.size()) + " inputs and " +
                                                std::to_string(outputs.size()) +
                                                " outputs; an embed hef has the rows and the bias in and the "
                                                "hidden states out");
    const size_t rows = (size_t)desc.seq * desc.hidden, bias = (size_t)desc.seq * desc.heads * desc.seq;
    // The inputs are told apart by the names the compiler gives them in the
    // order the graph declares them (bundle/hailo/hef_compile.py): the rows
    // [seq, hidden] first, then the bias [seq, heads * seq]. Their sizes
    // alone do not tell them apart when heads * seq is hidden (MiniLM at 32
    // tokens), so each is checked against the size its name implies.
    for (const auto &name : inputs) {
        const auto ends = [&](const std::string &tail) {
            return name.size() >= tail.size() && name.compare(name.size() - tail.size(), tail.size(), tail) == 0;
        };
        const bool is_rows = ends("/input_layer1"), is_bias = ends("/input_layer2");
        if (!is_rows && !is_bias)
            return fail(TURBO_E_BUNDLE_INVALID, "hef input " + name +
                                                    ": an embed hef's inputs are input_layer1, the rows, and "
                                                    "input_layer2, the bias");
        auto in = m->infer_->input(name);
        if (!in) return hailort_failed(in.status(), "input " + name);
        Failure e = describe(*in, name, is_rows ? rows : bias, is_rows ? m->rows_ : m->bias_);
        if (e) return e;
    }
    if (m->rows_.name.empty() || m->bias_.name.empty())
        return fail(TURBO_E_BUNDLE_INVALID, "the hef's inputs are not input_layer1 and input_layer2");
    auto o = m->infer_->output(outputs[0]);
    if (!o) return hailort_failed(o.status(), "output " + outputs[0]);
    Failure e = describe(*o, outputs[0], rows, m->hidden_);
    if (e) return e;
    m->burst_ = BURST;
    m->infer_->set_batch_size(BURST);
    auto configured = m->infer_->configure();
    if (!configured) return hailort_failed(configured.status(), "configure");
    m->configured_.reset(new hailort::ConfiguredInferModel(configured.release()));
    auto queue = m->configured_->get_async_queue_size();
    if (!queue) return hailort_failed(queue.status(), "get_async_queue_size");
    m->queue_ = (uint32_t)*queue;
    out = std::move(m);
    return {};
}

uint32_t retired_frames() { return retired_count.load(); }

void Model::retire(std::unique_ptr<Slot> slot) {
    std::lock_guard<std::mutex> g(retired_mu_);
    retired_.push_back(std::move(slot));
    retired_count++;
}

void Model::reap() {
    std::lock_guard<std::mutex> g(retired_mu_);
    auto done = [](const std::unique_ptr<Slot> &s) { return s->job.wait(std::chrono::milliseconds{0}) != HAILO_TIMEOUT; };
    const auto kept = std::remove_if(retired_.begin(), retired_.end(), done);
    retired_count -= (uint32_t)(retired_.end() - kept);
    retired_.erase(kept, retired_.end());
}

Model::~Model() {
    reap();
    for (auto &s : retired_) {
        // Its job was waited on, so its destructor does not wait again;
        // the detach says so whatever HailoRT does. The memory is not
        // freed: the device may still own it.
        s->job.detach();
        (void)s.release();
    }
}

// The alignment HailoRT asks of a frame it reads or writes in place.
constexpr size_t FRAME_ALIGN = 16384;

bool Frame::resize(size_t bytes) {
    const size_t len = (bytes + FRAME_ALIGN - 1) / FRAME_ALIGN * FRAME_ALIGN;
    p.reset(static_cast<uint8_t *>(std::aligned_alloc(FRAME_ALIGN, len ? len : FRAME_ALIGN)));
    if (!p) return false;
    std::memset(p.get(), 0, len ? len : FRAME_ALIGN);
    n = bytes;
    return true;
}

Failure Session::create(Model &model, uint32_t max_batch, uint32_t max_seq, uint32_t max_ranges,
                        std::unique_ptr<Session> &out) {
    if (max_seq > model.desc().seq)
        return fail(TURBO_E_UNSUPPORTED_OPTION,
                    "max_seq " + std::to_string(max_seq) + " is over the hef's " + std::to_string(model.desc().seq),
                    2);
    model.reap();
    std::unique_ptr<Session> s(new Session());
    // Before anything can refuse: a refused session is released, and its
    // release reaps the model's frames.
    s->model_ = &model;
    if (Failure f = frame_timeout(s->frame_timeout_)) return f;
    s->max_batch_ = max_batch;
    s->max_seq_ = max_seq;
    s->max_ranges_ = max_ranges;
    s->ranges_.reserve(max_ranges);
    s->ids_.resize((size_t)max_batch * max_seq);
    s->mask_.resize((size_t)max_batch * max_seq);
    s->sum_.resize(model.desc().hidden);
    // Two bursts in flight, so one fills on the host while the other runs,
    // and no more than HailoRT queues or the session's rows need.
    const uint32_t slots = std::min({2 * model.burst(), model.queue(), std::max(max_batch, 2u)});
    for (uint32_t i = 0; i < std::max(slots, 1u); i++) {
        s->slots_.emplace_back(new Slot());
        Slot &slot = *s->slots_.back();
        if (!slot.rows.resize(model.rows().frame_bytes) || !slot.bias.resize(model.bias().frame_bytes) ||
            !slot.hidden.resize(model.hidden().frame_bytes))
            return fail(TURBO_E_OUT_OF_MEMORY, "a frame's memory", 0);
        auto b = model.configured().create_bindings();
        if (!b) return hailort_failed(b.status(), "create_bindings");
        slot.bindings.emplace(b.release());
        hailo_status st = slot.bindings->input(model.rows().name)->set_buffer(
            hailort::MemoryView(slot.rows.data(), slot.rows.size()));
        if (st == HAILO_SUCCESS)
            st = slot.bindings->input(model.bias().name)->set_buffer(
                hailort::MemoryView(slot.bias.data(), slot.bias.size()));
        if (st == HAILO_SUCCESS)
            st = slot.bindings->output(model.hidden().name)->set_buffer(
                hailort::MemoryView(slot.hidden.data(), slot.hidden.size()));
        if (st != HAILO_SUCCESS) return hailort_failed(st, "set_buffer");
    }
    out = std::move(s);
    return {};
}

Failure Session::write(const Rows &r) {
    if (!broken_.empty()) return fail(TURBO_E_INVALID_STATE, broken_);
    written_ = false;
    by_ranges_ = false;
    ranges_.clear();
    if (r.types) {
        for (uint32_t b = 0; b < r.batch; b++)
            for (uint32_t t = 0; t < r.seq; t++)
                if (r.types[(size_t)b * r.row_stride + t] != 0)
                    return fail(TURBO_E_UNSUPPORTED_OPTION,
                                "token type " + std::to_string(r.types[(size_t)b * r.row_stride + t]) + " in row " +
                                    std::to_string(b) + ": this hef computes token type 0 only");
    }
    rows_ = r;
    for (uint32_t b = 0; b < r.batch; b++) {
        std::memcpy(&ids_[(size_t)b * max_seq_], r.ids + (size_t)b * r.row_stride, (size_t)r.seq * 4);
        std::memcpy(&mask_[(size_t)b * max_seq_], r.mask + (size_t)b * r.row_stride, (size_t)r.seq * 4);
    }
    rows_.ids = rows_.mask = rows_.types = nullptr;   // the copies are read from here on
    written_ = true;
    return {};
}

Failure Session::write_ranges(const Rows &r, const turbo_token_range *ranges, uint32_t count) {
    Failure f = write(r);
    if (f) return f;
    // The rows are kept; the write stands only once the ranges are.
    written_ = false;
    if (r.pooling != TURBO_POOLING_MEAN)
        return fail(TURBO_E_UNSUPPORTED_OPTION,
                    "pooling " + std::to_string(r.pooling) + ": a range is pooled by the mean alone", 5);
    if (count == 0) return fail(TURBO_E_INVALID_ARGUMENT, "a write of ranges holds at least one");
    if (count > max_ranges_)
        return fail(TURBO_E_CAPACITY, std::to_string(count) + " ranges are over the session's max_ranges " +
                                          std::to_string(max_ranges_));
    // The core checks each range as turbo.h says. They are checked again
    // here because pooling indexes the session's rows by them, and a range
    // without a live token would be a mean over nothing.
    for (uint32_t i = 0; i < count; i++) {
        const turbo_token_range &g = ranges[i];
        if (g.row >= r.batch || g.start >= g.end || g.end > r.seq)
            return fail(TURBO_E_INVALID_ARGUMENT, "range " + std::to_string(i) + " is row " + std::to_string(g.row) +
                                                      " columns [" + std::to_string(g.start) + ", " +
                                                      std::to_string(g.end) + "), outside " +
                                                      std::to_string(r.batch) + " rows of " + std::to_string(r.seq) +
                                                      " tokens");
        const int32_t *mask = &mask_[(size_t)g.row * max_seq_];
        if (std::find(mask + g.start, mask + g.end, 1) == mask + g.end)
            return fail(TURBO_E_INVALID_ARGUMENT, "range " + std::to_string(i) + " has no token with mask 1");
    }
    ranges_.assign(ranges, ranges + count);
    by_ranges_ = true;
    written_ = true;
    return {};
}

/* Row row's frame: each token's word row, quantized as it is written, the
 * pad id's row past the written length, and the bias from the mask. */
void Session::fill(Slot &slot, uint32_t row) {
    const ModelDesc &d = model_->desc();
    const Stream &rs = model_->rows(), &bs = model_->bias();
    const int32_t *ids = &ids_[(size_t)row * max_seq_];
    const int32_t *mask = &mask_[(size_t)row * max_seq_];
    for (uint32_t t = 0; t < d.seq; t++) {
        // Past the written length the frame repeats the first id: the bias
        // drops those keys, and their outputs are not pooled.
        const float *w = d.word_table + (size_t)(t < rows_.seq ? ids[t] : ids[0]) * d.hidden;
        for (uint32_t h = 0; h < d.hidden; h++) put(rs, slot.rows.data(), (size_t)t * d.hidden + h, w[h]);
    }
    const size_t keys = (size_t)d.heads * d.seq;
    for (uint32_t k = 0; k < d.seq; k++) {
        const float v = (k < rows_.seq && mask[k]) ? 0.f : MASKED;
        for (uint32_t head = 0; head < d.heads; head++) put(bs, slot.bias.data(), (size_t)head * d.seq + k, v);
    }
    // Every query sees the same keys: the first query's bias, copied.
    const size_t line = keys * bs.element_bytes;
    for (uint32_t q = 1; q < d.seq; q++) std::memcpy(slot.bias.data() + q * line, slot.bias.data(), line);
    slot.row = row;
}

/* The vectors the slot's frame holds, pooled from its hidden states, cut to
 * output_dim, and normalized, into out. After write, the frame's row's
 * vector at the row's index. After write_ranges, the vector of every range
 * of the frame's row at the range's index: frames come back in any order,
 * so a range's place in the output is its place in the list, never its
 * row. */
void Session::pool(const Slot &slot, float *out) {
    const ModelDesc &d = model_->desc();
    const Stream &hs = model_->hidden();
    const int32_t *mask = &mask_[(size_t)slot.row * max_seq_];
    const uint32_t dim = rows_.output_dim;
    if (by_ranges_) {
        for (size_t i = 0; i < ranges_.size(); i++) {
            const turbo_token_range &g = ranges_[i];
            if (g.row != slot.row) continue;
            mean(slot, mask, g.start, g.end);
            put_vector(out + i * dim);
        }
        return;
    }
    if (rows_.pooling == TURBO_POOLING_MEAN) {
        mean(slot, mask, 0, rows_.seq);
    } else {
        uint32_t t = 0;
        if (rows_.pooling == TURBO_POOLING_LAST)
            for (uint32_t i = 0; i < rows_.seq; i++)
                if (mask[i]) t = i;
        for (uint32_t h = 0; h < d.hidden; h++) sum_[h] = get(hs, slot.hidden.data(), (size_t)t * d.hidden + h);
    }
    put_vector(out + (size_t)slot.row * dim);
}

/* The mean of the frame's hidden states over the columns [start, end) whose
 * mask is 1, into sum_. The tokens are summed in column order, so a row and
 * a range over the same columns give the same bits. */
void Session::mean(const Slot &slot, const int32_t *mask, uint32_t start, uint32_t end) {
    const ModelDesc &d = model_->desc();
    const Stream &hs = model_->hidden();
    std::fill(sum_.begin(), sum_.end(), 0.0);
    uint32_t live = 0;
    for (uint32_t t = start; t < end; t++) {
        if (!mask[t]) continue;
        live++;
        for (uint32_t h = 0; h < d.hidden; h++) sum_[h] += get(hs, slot.hidden.data(), (size_t)t * d.hidden + h);
    }
    for (double &v : sum_) v /= live;
}

/* sum_ cut to output_dim and normalized as the rows say, into dst as F32. */
void Session::put_vector(float *dst) {
    const uint32_t dim = rows_.output_dim;
    double scale = 1.0;
    if (rows_.normalize == TURBO_NORMALIZE_L2) {
        double n = 0;
        for (uint32_t h = 0; h < dim; h++) n += sum_[h] * sum_[h];
        scale = 1.0 / std::fmax(std::sqrt(n), 1e-12);
    }
    for (uint32_t h = 0; h < dim; h++) dst[h] = (float)(sum_[h] * scale);
}

Session::~Session() {
    // Once a frame has not come back, in a run or here, the device is
    // taken as gone and the rest are not waited for.
    std::chrono::milliseconds timeout = broken_.empty() ? frame_timeout_ : std::chrono::milliseconds{0};
    for (auto &s : slots_) {
        if (!s || !s->busy) continue;
        if (s->job.wait(timeout) == HAILO_SUCCESS) continue;
        timeout = std::chrono::milliseconds{0};
        // The device may still write into the slot: the model keeps it
        // until its job finishes.
        model_->retire(std::move(s));
    }
    model_->reap();
}

Failure Session::run(float *out, uint64_t &h2d, uint64_t &d2h) {
    if (!broken_.empty()) return fail(TURBO_E_INVALID_STATE, broken_);
    if (!written_) return fail(TURBO_E_INVALID_STATE, "the hailo session has no rows written since its last run");
    written_ = false;
    h2d = d2h = 0;
    std::lock_guard<std::mutex> g(model_->runs());
    auto &cm = model_->configured();
    // Up to slots_.size() frames in flight: each slot is filled again as
    // soon as its frame comes back.
    Failure failed;
    auto fault = [&](Failure f) {
        if (!failed) failed = f;
        broken_ = "a frame failed on the device (" + f.message + "); release the session";
    };
    auto finish = [&](Slot &s) {
        if (!s.busy) return;
        const hailo_status st = s.job.wait(frame_timeout_);
        if (st != HAILO_SUCCESS) {
            // The device may still own this slot's memory: it stays busy,
            // and the session is not used again.
            fault(hailort_failed(st, "frame " + std::to_string(s.row)));
            return;
        }
        s.busy = false;
        d2h += s.hidden.size();
        if (!failed) pool(s, out);
    };
    for (uint32_t row = 0; row < rows_.batch && !failed; row++) {
        Slot &s = *slots_[row % slots_.size()];
        finish(s);
        if (failed) break;
        fill(s, row);
        hailo_status st = cm.wait_for_async_ready(FRAME_TIMEOUT);
        if (st != HAILO_SUCCESS) {
            fault(hailort_failed(st, "wait_for_async_ready"));
            break;
        }
        auto job = cm.run_async(*s.bindings);
        if (!job) {
            fault(hailort_failed(job.status(), "run_async"));
            break;
        }
        s.job = job.release();
        s.busy = true;
        h2d += s.rows.size() + s.bias.size();
    }
    // Every frame still in flight is waited for, until one fails: then the
    // rest stay busy, the run returns after one FRAME_TIMEOUT at most, and
    // releasing the session does not wait for them again (~Session,
    // docs/hailo.md).
    for (auto &s : slots_)
        if (!failed) finish(*s);
    return failed;
}

} // namespace turbo_hailo
