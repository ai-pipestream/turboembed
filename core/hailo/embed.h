/* SPDX-License-Identifier: Apache-2.0
 *
 * Embed on a Hailo device through HailoRT's InferModel: a model is a HEF
 * that takes the word-embedding rows of a fixed-length frame and an
 * additive attention bias per head, and gives the encoder's hidden states;
 * a session looks the rows up on the host, quantizing each value as it
 * writes it into the frame HailoRT sends, keeps two frames in flight, and
 * pools and normalizes each frame's hidden states on the host as it comes
 * back. backend.cpp puts these behind the turbo_backend table.
 */

#ifndef TURBO_HAILO_EMBED_H
#define TURBO_HAILO_EMBED_H

#include <turbo/turbo.h>

#include <hailo/hailort.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace turbo_hailo {

/* A failure, as a status and a message for turbo_error. */
struct Failure {
    int32_t code = TURBO_OK;
    std::string message;
    uint32_t field = 0;
    explicit operator bool() const { return code != TURBO_OK; }
};

/* One element of a frame, as HailoRT lays it out: its width and the
 * quantization that maps a float onto it. */
struct Stream {
    std::string name;
    size_t frame_bytes = 0;
    size_t element_bytes = 0;   // 1 or 2
    float scale = 1.f;
    float zero_point = 0.f;
};

/* What model_load hands over: the HEF's bytes and the word table, both
 * the core's, unchanged until the model is released. */
struct ModelDesc {
    const void *hef = nullptr;
    uint64_t hef_bytes = 0;
    const float *word_table = nullptr;   // [vocab, hidden] F32
    uint32_t vocab = 0;
    uint32_t hidden = 0;
    uint32_t heads = 0;
    uint32_t seq = 0;   // the HEF's fixed frame length
};

class Model {
  public:
    /* The HEF configured on vdevice, checked against desc: two inputs, the
     * rows [seq, hidden] and the bias [seq, heads * seq], and one output
     * [seq, hidden]. */
    static Failure load(hailort::VDevice &vdevice, const ModelDesc &desc, std::unique_ptr<Model> &out);

    const ModelDesc &desc() const { return desc_; }
    const Stream &rows() const { return rows_; }
    const Stream &bias() const { return bias_; }
    const Stream &hidden() const { return hidden_; }

    /* The configured model runs one frame set at a time from whichever
     * session holds this lock; HailoRT's pipeline is shared. */
    std::mutex &runs() { return runs_; }
    hailort::ConfiguredInferModel &configured() { return *configured_; }

  private:
    ModelDesc desc_;
    std::shared_ptr<hailort::InferModel> infer_;
    std::unique_ptr<hailort::ConfiguredInferModel> configured_;
    Stream rows_, bias_, hidden_;
    std::mutex runs_;
};

/* The rows of one run, as embed_write passes them. */
struct Rows {
    uint32_t batch = 0, seq = 0, row_stride = 0;
    const int32_t *ids = nullptr;
    const int32_t *mask = nullptr;
    const int32_t *types = nullptr;   // NULL for all zero
    uint32_t pooling = 0, normalize = 0, output_dim = 0;
};

class Session {
  public:
    /* Frames for max_batch rows of up to max_seq tokens: two in flight,
     * and the session's copy of the rows. Everything a run touches is
     * allocated here. */
    static Failure create(Model &model, uint32_t max_batch, uint32_t max_seq, std::unique_ptr<Session> &out);

    /* The rows and options of the next run, copied. A token type other
     * than 0 is refused: the HEF folds type 0 in as a constant. */
    Failure write(const Rows &rows);

    /* Runs every row, pooling and normalizing each into out, [batch,
     * output_dim] F32 packed. h2d and d2h count the frames' bytes. */
    Failure run(float *out, uint64_t &h2d, uint64_t &d2h);

    uint32_t normalize() const { return rows_.normalize; }

  private:
    struct Slot {
        std::vector<uint8_t> rows, bias, hidden;
        std::optional<hailort::ConfiguredInferModel::Bindings> bindings;
        hailort::AsyncInferJob job;
        bool busy = false;
        uint32_t row = 0;
    };

    void fill(Slot &slot, uint32_t row);
    void pool(const Slot &slot, float *out);

    Model *model_ = nullptr;
    uint32_t max_batch_ = 0, max_seq_ = 0;
    Rows rows_;
    std::vector<int32_t> ids_, mask_;
    std::vector<double> sum_;   // one row's pooled values, before the cut and the norm
    Slot slots_[2];
    bool written_ = false;
};

} // namespace turbo_hailo

#endif
