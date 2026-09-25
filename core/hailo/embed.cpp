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
#include <cstring>

namespace turbo_hailo {

namespace {

/* The bias for a dropped key: the value the HEF was calibrated with.
 * exp(-100) underflows to 0 in the softmax, as the upstream model's
 * dtype minimum does. */
constexpr float MASKED = -100.f;

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
    // The inputs are told apart by size: the rows are [seq, hidden], the
    // bias [seq, heads * seq].
    for (const auto &name : inputs) {
        auto in = m->infer_->input(name);
        if (!in) return hailort_failed(in.status(), "input " + name);
        const size_t f = in->get_frame_size();
        Stream &s = (f == rows || f == 2 * rows) ? m->rows_ : m->bias_;
        if (!s.name.empty())
            return fail(TURBO_E_BUNDLE_INVALID, "hef inputs " + s.name + " and " + name + " are the same size");
        Failure e = describe(*in, name, &s == &m->rows_ ? rows : bias, s);
        if (e) return e;
    }
    if (m->rows_.name.empty() || m->bias_.name.empty())
        return fail(TURBO_E_BUNDLE_INVALID,
                    "the hef's inputs are not [" + std::to_string(desc.seq) + ", " + std::to_string(desc.hidden) +
                        "] rows and a [" + std::to_string(desc.seq) + ", " + std::to_string(desc.heads * desc.seq) +
                        "] bias");
    auto o = m->infer_->output(outputs[0]);
    if (!o) return hailort_failed(o.status(), "output " + outputs[0]);
    Failure e = describe(*o, outputs[0], rows, m->hidden_);
    if (e) return e;
    auto configured = m->infer_->configure();
    if (!configured) return hailort_failed(configured.status(), "configure");
    m->configured_.reset(new hailort::ConfiguredInferModel(configured.release()));
    out = std::move(m);
    return {};
}

Failure Session::create(Model &model, uint32_t max_batch, uint32_t max_seq, std::unique_ptr<Session> &out) {
    if (max_seq > model.desc().seq)
        return fail(TURBO_E_UNSUPPORTED_OPTION,
                    "max_seq " + std::to_string(max_seq) + " is over the hef's " + std::to_string(model.desc().seq),
                    2);
    std::unique_ptr<Session> s(new Session());
    s->model_ = &model;
    s->max_batch_ = max_batch;
    s->max_seq_ = max_seq;
    s->ids_.resize((size_t)max_batch * max_seq);
    s->mask_.resize((size_t)max_batch * max_seq);
    s->sum_.resize(model.desc().hidden);
    for (Slot &slot : s->slots_) {
        slot.rows.resize(model.rows().frame_bytes);
        slot.bias.resize(model.bias().frame_bytes);
        slot.hidden.resize(model.hidden().frame_bytes);
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

/* The frame's hidden states pooled over the row's mask, cut to output_dim,
 * and normalized, into out. */
void Session::pool(const Slot &slot, float *out) {
    const ModelDesc &d = model_->desc();
    const Stream &hs = model_->hidden();
    const int32_t *mask = &mask_[(size_t)slot.row * max_seq_];
    std::fill(sum_.begin(), sum_.end(), 0.0);
    if (rows_.pooling == TURBO_POOLING_MEAN) {
        uint32_t live = 0;
        for (uint32_t t = 0; t < rows_.seq; t++) {
            if (!mask[t]) continue;
            live++;
            for (uint32_t h = 0; h < d.hidden; h++) sum_[h] += get(hs, slot.hidden.data(), (size_t)t * d.hidden + h);
        }
        for (double &v : sum_) v /= live;
    } else {
        uint32_t t = 0;
        if (rows_.pooling == TURBO_POOLING_LAST)
            for (uint32_t i = 0; i < rows_.seq; i++)
                if (mask[i]) t = i;
        for (uint32_t h = 0; h < d.hidden; h++) sum_[h] = get(hs, slot.hidden.data(), (size_t)t * d.hidden + h);
    }
    const uint32_t dim = rows_.output_dim;
    double scale = 1.0;
    if (rows_.normalize == TURBO_NORMALIZE_L2) {
        double n = 0;
        for (uint32_t h = 0; h < dim; h++) n += sum_[h] * sum_[h];
        scale = 1.0 / std::fmax(std::sqrt(n), 1e-12);
    }
    for (uint32_t h = 0; h < dim; h++) out[(size_t)slot.row * dim + h] = (float)(sum_[h] * scale);
}

Failure Session::run(float *out, uint64_t &h2d, uint64_t &d2h) {
    if (!broken_.empty()) return fail(TURBO_E_INVALID_STATE, broken_);
    if (!written_) return fail(TURBO_E_INVALID_STATE, "the hailo session has no rows written since its last run");
    written_ = false;
    h2d = d2h = 0;
    std::lock_guard<std::mutex> g(model_->runs());
    auto &cm = model_->configured();
    // Two frames in flight: one fills on the host while the other runs.
    Failure failed;
    auto fault = [&](Failure f) {
        if (!failed) failed = f;
        broken_ = "a frame failed on the device (" + f.message + "); release the session";
    };
    auto finish = [&](Slot &s) {
        if (!s.busy) return;
        const hailo_status st = s.job.wait(FRAME_TIMEOUT);
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
        Slot &s = slots_[row % 2];
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
    // Every frame still in flight is waited for. After a frame timed out,
    // this waits on the same slot once more, so a device that stopped
    // answering costs up to two FRAME_TIMEOUTs in the run, and releasing
    // the session then waits without a timeout (docs/hailo.md).
    for (Slot &s : slots_) finish(s);
    return failed;
}

} // namespace turbo_hailo
