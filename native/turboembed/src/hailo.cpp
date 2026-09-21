/* SPDX-License-Identifier: Apache-2.0
 *
 * TurboEmbed Hailo provider — Raspberry Pi AI HAT+ (Hailo-8 / Hailo-8L,
 * Hailo-10H later). Compiled only with -DTURBOEMBED_HAILO (cargo feature
 * `hailo`); requires hailo/hailort.h from the Pi apt packages
 * (`hailo-all` for Hailo-8/8L, `hailo-h10-all` for Hailo-10H).
 *
 * Split host/NPU architecture (see docs/hailo-embed.md):
 *   host: WordPiece tokenize (native/wordpiece), then per `front_end`:
 *         the raw word-embedding gather for the official Model Zoo HEFs
 *         (position/token-type/LayerNorm live inside those HEFs), or the
 *         full word+position+token_type+LayerNorm for community
 *         single-input HEFs; plus padding, pooling, L2 normalize.
 *   NPU:  the fixed-shape encoder body from model.hef via the HailoRT C
 *         API (vdevice → hef → configure → vstreams → write/read frame);
 *         the official HEFs take a second input with the [seq,seq]
 *         additive attention bias (0 = attend, -10000 = masked).
 *
 * The vdevice is created with the scheduler OFF (single model per engine —
 * with the default round-robin scheduler, HailoRT owns activation and
 * explicit hailo_activate_network_group fails INVALID_OPERATION).
 *
 * Known v1 caveat: INT8 quantization compresses absolute cosine vs FP32
 * (mean ≈ 0.54 on the reference texts) while ranking stays near-parity
 * (Spearman 0.937 vs 0.9438 FP32) — see testdata/receipts/turboembed/
 * pi5ai1-hailo8.json. The live suite gates ranking, not cosine.
 */

#include "hailo.hpp"

#include <hailo/hailort.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nlohmann/json.hpp"
#include "wordpiece.h"

namespace turboembed_hailo {
namespace {

/* embedding_tables.bin: little-endian header followed by fp32 arrays.
 * u32 magic 'TEMB' | u32 version(1) | u32 vocab_rows | u32 max_pos |
 * u32 dim | f32 layer_norm_eps |
 * float word[vocab_rows][dim] | float pos[max_pos][dim] |
 * float token_type[2][dim] | float ln_gamma[dim] | float ln_beta[dim] */
constexpr uint32_t kTablesMagic = 0x54454d42u; // "TEMB"
constexpr uint32_t kTablesVersion = 1u;

constexpr size_t kMaxScanDevices = 8;
constexpr size_t kMaxNetworkGroups = 8; // HAILO_MAX_NETWORK_GROUPS
constexpr size_t kMaxStreams = 40;      // HAILO_MAX_STREAMS_COUNT

turboembed_status hailo_fail(hailo_status st, const char *what, std::string *err) {
    *err = std::string(what) + ": " + hailo_get_status_message(st);
    return TURBOEMBED_ERR_UNAVAILABLE;
}

bool read_exact(FILE *f, void *dst, size_t bytes) {
    return bytes == 0 || std::fread(dst, 1, bytes, f) == bytes;
}

bool file_exists(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    std::fclose(f);
    return true;
}

std::string join_path(const std::string &dir, const char *name) {
    return dir + "/" + name;
}

} // namespace

struct Pipeline::Impl {
    std::string workspace_root;
    std::string alias;
    std::string model_dir;

    hailo_vdevice vdevice = nullptr;
    hailo_hef hef = nullptr;
    hailo_configured_network_group network_group = nullptr;
    hailo_activated_network_group activated = nullptr;
    // One encoder-body input (hidden states); the official Model Zoo HEFs
    // add a second input for the [seq,seq] attention mask. mask_stream is
    // nullptr on single-input (unmasked) HEFs.
    hailo_input_vstream input_streams[2] = {nullptr, nullptr};
    size_t input_stream_count = 0;
    hailo_input_vstream hidden_stream = nullptr;
    hailo_input_vstream mask_stream = nullptr;
    hailo_output_vstream output_stream = nullptr;
    size_t hidden_frame_bytes = 0;
    size_t mask_frame_bytes = 0;
    size_t output_frame_bytes = 0;

    uint32_t dim = 0;
    uint32_t seq_len = 0;
    uint8_t default_pooling = TURBOEMBED_POOLING_MEAN;
    bool default_normalize = true;
    /* Front-end contract, from config.json `front_end`:
     * true  = "word": input_layer1 is the raw word-embedding row for every
     *         position (padding positions carry the PAD id's row); the HEF
     *         contains the position/token-type add + embeddings LayerNorm.
     *         This is the official Hailo Model Zoo MiniLM contract.
     * false = "bert_embeddings": host computes word+position+token_type +
     *         LayerNorm and zero-pads (community single-input HEFs). */
    bool word_only_frontend = true;

    wordpiece_vocab *vocab = nullptr;
    std::vector<float> word_emb;
    std::vector<float> pos_emb;
    std::vector<float> token_type_emb;
    std::vector<float> ln_gamma;
    std::vector<float> ln_beta;
    float ln_eps = 1e-12f;
    uint32_t vocab_rows = 0;
    uint32_t max_pos = 0;

    std::vector<int32_t> ids;     // token scratch, reused per text
    std::vector<float> in_frame;  // [seq_len * dim] host-format input
    std::vector<float> out_frame; // [seq_len * dim] host-format output
    std::vector<float> mask_frame; // [seq_len * seq_len] attention mask (2-input HEFs)

    ~Impl() { teardown(); }

    void teardown() {
        if (input_stream_count > 0) {
            (void)hailo_release_input_vstreams(input_streams, input_stream_count);
            input_stream_count = 0;
            hidden_stream = nullptr;
            mask_stream = nullptr;
        }
        if (output_stream != nullptr) {
            (void)hailo_release_output_vstreams(&output_stream, 1);
            output_stream = nullptr;
        }
        if (activated != nullptr) {
            (void)hailo_deactivate_network_group(activated);
            activated = nullptr;
        }
        network_group = nullptr; // owned by the configured vdevice scope
        if (hef != nullptr) {
            (void)hailo_release_hef(hef);
            hef = nullptr;
        }
        if (vdevice != nullptr) {
            (void)hailo_release_vdevice(vdevice);
            vdevice = nullptr;
        }
        if (vocab != nullptr) {
            wordpiece_vocab_destroy(vocab);
            vocab = nullptr;
        }
    }
};

Pipeline::Pipeline() : impl_(new (std::nothrow) Impl()) {}
Pipeline::~Pipeline() = default;

uint32_t Pipeline::embedding_dim() const {
    return impl_ == nullptr ? 0 : impl_->dim;
}

const std::string &Pipeline::model_alias() const {
    static const std::string kEmpty;
    return impl_ == nullptr ? kEmpty : impl_->alias;
}

bool Pipeline::is_loaded() const {
    return impl_ != nullptr && impl_->activated != nullptr;
}

turboembed_status Pipeline::scan_count(size_t *count, std::string *err) {
    hailo_pcie_device_info_t infos[kMaxScanDevices];
    size_t found = 0;
    const hailo_status st =
        hailo_scan_pcie_devices(infos, kMaxScanDevices, &found);
    if (st != HAILO_SUCCESS) {
        *err = std::string("hailo_scan_pcie_devices failed: ") +
               hailo_get_status_message(st) +
               " (is the hailo PCIe driver loaded? see docs/hailo-embed.md)";
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    *count = found;
    return TURBOEMBED_OK;
}

turboembed_status Pipeline::open(
    const char *workspace_root,
    std::unique_ptr<Pipeline> *out,
    std::string *err
) {
    size_t count = 0;
    turboembed_status st = scan_count(&count, err);
    if (st != TURBOEMBED_OK) {
        return st;
    }
    if (count == 0) {
        *err =
            "no Hailo device found — expected a Raspberry Pi AI HAT+ on "
            "PCIe (/dev/hailo0 for Hailo-8/8L, /dev/h1x-0 for Hailo-10H). "
            "Install the driver/runtime with `sudo apt install dkms "
            "hailo-all` (Hailo-8/8L) or `hailo-h10-all` (Hailo-10H); "
            "refusing any CPU/mock substitute";
        return TURBOEMBED_ERR_UNSUPPORTED_DEVICE;
    }

    auto pipe = std::unique_ptr<Pipeline>(new (std::nothrow) Pipeline());
    if (pipe == nullptr || pipe->impl_ == nullptr) {
        *err = "hailo pipeline allocation failed";
        return TURBOEMBED_ERR_OUT_OF_MEMORY;
    }
    pipe->impl_->workspace_root =
        workspace_root != nullptr ? workspace_root : "";

    hailo_vdevice_params_t params;
    hailo_status hst = hailo_init_vdevice_params(&params);
    if (hst != HAILO_SUCCESS) {
        return hailo_fail(hst, "hailo_init_vdevice_params", err);
    }
    // One engine serves one model: disable the round-robin scheduler (the
    // default) so we own activation explicitly — with the scheduler on,
    // HailoRT owns activation and hailo_activate_network_group fails with
    // HAILO_INVALID_OPERATION.
    params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_NONE;
    hst = hailo_create_vdevice(&params, &pipe->impl_->vdevice);
    if (hst != HAILO_SUCCESS) {
        return hailo_fail(hst, "hailo_create_vdevice", err);
    }
    *out = std::move(pipe);
    return TURBOEMBED_OK;
}

namespace {

turboembed_status load_tables(const std::string &path, Pipeline::Impl *impl, std::string *err) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        *err = "cannot open " + path +
               " (run scripts/export-minilm-hailo-tables.py)";
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    uint32_t header[5] = {0, 0, 0, 0, 0};
    float eps = 0.0f;
    bool ok = read_exact(f, header, sizeof(header)) && read_exact(f, &eps, sizeof(eps)) &&
              header[0] == kTablesMagic && header[1] == kTablesVersion;
    if (!ok) {
        std::fclose(f);
        *err = path + ": bad embedding_tables.bin header (magic/version)";
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    const uint32_t vocab = header[2];
    const uint32_t max_pos = header[3];
    const uint32_t dim = header[4];
    if (vocab == 0 || max_pos == 0 || dim == 0 || vocab > (1u << 22) ||
        dim > 8192 || max_pos > 65536) {
        std::fclose(f);
        *err = path + ": implausible table shape (vocab/pos/dim)";
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    impl->word_emb.resize(static_cast<size_t>(vocab) * dim);
    impl->pos_emb.resize(static_cast<size_t>(max_pos) * dim);
    impl->token_type_emb.resize(2u * dim);
    impl->ln_gamma.resize(dim);
    impl->ln_beta.resize(dim);
    ok = read_exact(f, impl->word_emb.data(), impl->word_emb.size() * sizeof(float)) &&
         read_exact(f, impl->pos_emb.data(), impl->pos_emb.size() * sizeof(float)) &&
         read_exact(f, impl->token_type_emb.data(), impl->token_type_emb.size() * sizeof(float)) &&
         read_exact(f, impl->ln_gamma.data(), dim * sizeof(float)) &&
         read_exact(f, impl->ln_beta.data(), dim * sizeof(float));
    std::fclose(f);
    if (!ok) {
        *err = path + ": truncated embedding_tables.bin";
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    impl->vocab_rows = vocab;
    impl->max_pos = max_pos;
    impl->dim = dim;
    impl->ln_eps = eps;
    return TURBOEMBED_OK;
}

turboembed_status load_config(const std::string &path, Pipeline::Impl *impl, std::string *err) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        *err = "cannot open " + path;
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size <= 0 || size > (1 << 20)) {
        std::fclose(f);
        *err = path + ": empty or implausibly large config.json";
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    std::string text(static_cast<size_t>(size), '\0');
    const bool ok = read_exact(f, text.data(), text.size());
    std::fclose(f);
    if (!ok) {
        *err = path + ": short read on config.json";
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    try {
        const nlohmann::json cfg = nlohmann::json::parse(text);
        impl->seq_len = cfg.value("seq_len", 128u);
        const std::string pooling = cfg.value("pooling", std::string("mean"));
        if (pooling == "mean") {
            impl->default_pooling = TURBOEMBED_POOLING_MEAN;
        } else if (pooling == "cls") {
            impl->default_pooling = TURBOEMBED_POOLING_CLS;
        } else if (pooling == "last") {
            impl->default_pooling = TURBOEMBED_POOLING_LAST;
        } else {
            *err = path + ": pooling must be mean|cls|last";
            return TURBOEMBED_ERR_UNAVAILABLE;
        }
        impl->default_normalize = cfg.value("normalize", true);
        const std::string front_end = cfg.value("front_end", std::string("word"));
        if (front_end == "word") {
            impl->word_only_frontend = true;
        } else if (front_end == "bert_embeddings") {
            impl->word_only_frontend = false;
        } else {
            *err = path + ": front_end must be \"word\" or \"bert_embeddings\"";
            return TURBOEMBED_ERR_UNAVAILABLE;
        }
    } catch (const std::exception &e) {
        *err = path + ": invalid config.json: " + e.what();
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    if (impl->seq_len < 3 || impl->seq_len > impl->max_pos) {
        *err = path + ": seq_len must be in [3, max_pos]";
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    return TURBOEMBED_OK;
}

} // namespace

turboembed_status Pipeline::load(
    const std::string &alias,
    const std::string &config_path,
    std::string *err
) {
    Impl *impl = impl_.get();
    if (impl == nullptr || impl->vdevice == nullptr) {
        *err = "hailo pipeline is not open";
        return TURBOEMBED_ERR_INTERNAL;
    }
    if (impl->activated != nullptr) {
        *err = "a model is already loaded; destroy the engine to switch";
        return TURBOEMBED_ERR_INVALID_ARGUMENT;
    }

    const std::string dir = !config_path.empty()
                                ? config_path
                                : impl->workspace_root + "/models/hailo/" + alias;
    static const char *kRequired[] = {
        "model.hef", "config.json", "embedding_tables.bin", "vocab.txt"
    };
    std::string missing;
    for (const char *name : kRequired) {
        if (!file_exists(join_path(dir, name))) {
            if (!missing.empty()) {
                missing += ", ";
            }
            missing += name;
        }
    }
    if (!missing.empty()) {
        *err = "hailo model dir " + dir + " is missing: " + missing +
               " (run `make fetch-hailo` and "
               "scripts/export-minilm-hailo-tables.py)";
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    impl->model_dir = dir;

    turboembed_status st = load_tables(join_path(dir, "embedding_tables.bin"), impl, err);
    if (st != TURBOEMBED_OK) {
        return st;
    }
    st = load_config(join_path(dir, "config.json"), impl, err);
    if (st != TURBOEMBED_OK) {
        return st;
    }
    const std::string vocab_path = join_path(dir, "vocab.txt");
    if (wordpiece_vocab_load(vocab_path.c_str(), &impl->vocab) != 0 || impl->vocab == nullptr) {
        *err = "failed to load " + vocab_path;
        return TURBOEMBED_ERR_UNAVAILABLE;
    }

    hailo_status hst =
        hailo_create_hef_file(&impl->hef, join_path(dir, "model.hef").c_str());
    if (hst != HAILO_SUCCESS) {
        return hailo_fail(hst, "hailo_create_hef_file (model.hef unreadable)", err);
    }
    hailo_configure_params_t params;
    hst = hailo_init_configure_params_by_vdevice(impl->hef, impl->vdevice, &params);
    if (hst != HAILO_SUCCESS) {
        return hailo_fail(hst, "hailo_init_configure_params_by_vdevice", err);
    }
    hailo_configured_network_group groups[kMaxNetworkGroups];
    size_t group_count = kMaxNetworkGroups;
    hst = hailo_configure_vdevice(impl->vdevice, impl->hef, &params, groups, &group_count);
    if (hst != HAILO_SUCCESS) {
        *err = std::string("hailo_configure_vdevice failed: ") +
               hailo_get_status_message(hst) +
               " (a HEF is arch-locked: hailo8l / hailo8 / hailo10h builds "
               "are not interchangeable — fetch the HEF for this board)";
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    if (group_count != 1) {
        *err = "expected exactly 1 network group in the encoder HEF, got " +
               std::to_string(group_count);
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    impl->network_group = groups[0];

    hailo_input_vstream_params_by_name_t in_params[kMaxStreams];
    size_t in_count = kMaxStreams;
    hst = hailo_hef_make_input_vstream_params(
        impl->hef, nullptr, false, HAILO_FORMAT_TYPE_FLOAT32, in_params, &in_count
    );
    if (hst != HAILO_SUCCESS) {
        return hailo_fail(hst, "hailo_hef_make_input_vstream_params", err);
    }
    hailo_output_vstream_params_by_name_t out_params[kMaxStreams];
    size_t out_count = kMaxStreams;
    hst = hailo_make_output_vstream_params(
        impl->network_group, false, HAILO_FORMAT_TYPE_FLOAT32, out_params, &out_count
    );
    if (hst != HAILO_SUCCESS) {
        return hailo_fail(hst, "hailo_make_output_vstream_params", err);
    }
    // Encoder contract: one hidden-states input [seq,dim]; the official
    // Model Zoo HEFs add a second attention-mask input [seq,seq]. Exactly
    // one token-level [seq,dim] output.
    if (in_count < 1 || in_count > 2 || out_count != 1) {
        *err = "expected 1-2 input vstreams and 1 output in the encoder HEF, got " +
               std::to_string(in_count) + "/" + std::to_string(out_count);
        return TURBOEMBED_ERR_UNAVAILABLE;
    }

    hst = hailo_create_input_vstreams(
        impl->network_group, in_params, in_count, impl->input_streams
    );
    if (hst != HAILO_SUCCESS) {
        return hailo_fail(hst, "hailo_create_input_vstreams", err);
    }
    impl->input_stream_count = in_count;
    hailo_output_vstream outputs[1];
    hst = hailo_create_output_vstreams(impl->network_group, out_params, 1, outputs);
    if (hst != HAILO_SUCCESS) {
        return hailo_fail(hst, "hailo_create_output_vstreams", err);
    }
    impl->output_stream = outputs[0];

    const size_t expect_hidden =
        static_cast<size_t>(impl->seq_len) * impl->dim * sizeof(float);
    const size_t expect_mask =
        static_cast<size_t>(impl->seq_len) * impl->seq_len * sizeof(float);
    if (expect_hidden == expect_mask) {
        *err = "dim == seq_len makes the hidden/mask inputs ambiguous";
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    for (size_t i = 0; i < in_count; ++i) {
        size_t frame = 0;
        hst = hailo_get_input_vstream_frame_size(impl->input_streams[i], &frame);
        if (hst != HAILO_SUCCESS) {
            return hailo_fail(hst, "hailo_get_input_vstream_frame_size", err);
        }
        if (frame == expect_hidden && impl->hidden_stream == nullptr) {
            impl->hidden_stream = impl->input_streams[i];
            impl->hidden_frame_bytes = frame;
        } else if (frame == expect_mask && impl->mask_stream == nullptr) {
            impl->mask_stream = impl->input_streams[i];
            impl->mask_frame_bytes = frame;
        } else {
            *err = "unexpected HEF input frame size " + std::to_string(frame) +
                   " bytes; expected hidden [seq*dim*4 = " +
                   std::to_string(expect_hidden) + "] or mask [seq*seq*4 = " +
                   std::to_string(expect_mask) + "] — see docs/hailo-embed.md";
            return TURBOEMBED_ERR_UNAVAILABLE;
        }
    }
    if (impl->hidden_stream == nullptr) {
        *err = "no hidden-states input of [seq*dim*4] bytes in the HEF";
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
    hst = hailo_get_output_vstream_frame_size(impl->output_stream, &impl->output_frame_bytes);
    if (hst != HAILO_SUCCESS) {
        return hailo_fail(hst, "hailo_get_output_vstream_frame_size", err);
    }
    if (impl->output_frame_bytes != expect_hidden) {
        *err = "HEF output frame is " + std::to_string(impl->output_frame_bytes) +
               " bytes; expected token-level [seq_len,dim] fp32 = " +
               std::to_string(expect_hidden) +
               " (a pooled-output HEF is not supported — compile with the "
               "encoder body ending at last_hidden_state)";
        return TURBOEMBED_ERR_UNAVAILABLE;
    }

    hst = hailo_activate_network_group(impl->network_group, nullptr, &impl->activated);
    if (hst != HAILO_SUCCESS) {
        return hailo_fail(hst, "hailo_activate_network_group", err);
    }

    impl->ids.resize(impl->seq_len);
    impl->in_frame.assign(static_cast<size_t>(impl->seq_len) * impl->dim, 0.0f);
    impl->out_frame.assign(static_cast<size_t>(impl->seq_len) * impl->dim, 0.0f);
    if (impl->mask_stream != nullptr) {
        impl->mask_frame.assign(
            static_cast<size_t>(impl->seq_len) * impl->seq_len, 0.0f
        );
    }
    impl->alias = alias;
    return TURBOEMBED_OK;
}

namespace {

/* Fill the input frame. ids[0] is [CLS], ids[n_real-1] is [SEP].
 * "word" front-end (official zoo HEF): raw word-embedding rows at every
 * position, PAD id beyond n_real — the HEF adds position/token-type and
 * the embeddings LayerNorm itself.
 * "bert_embeddings" front-end (community HEFs): host-side
 * word+pos+token_type + LayerNorm on real tokens, zero padding after. */
void build_frame(Pipeline::Impl *impl, const int32_t *ids, size_t n_real) {
    float *frame = impl->in_frame.data();
    const uint32_t dim = impl->dim;
    if (impl->word_only_frontend) {
        const int32_t pad_id = wordpiece_pad_id(impl->vocab);
        for (size_t t = 0; t < impl->seq_len; ++t) {
            int32_t id = t < n_real ? ids[t] : pad_id;
            const float *word = nullptr;
            if (id >= 0 && static_cast<uint32_t>(id) < impl->vocab_rows) {
                word = impl->word_emb.data() + static_cast<size_t>(id) * dim;
            }
            if (word == nullptr) {
                word = impl->word_emb.data(); // id 0 — never out of bounds
            }
            std::memcpy(frame + t * dim, word, dim * sizeof(float));
        }
        return;
    }
    std::memset(frame, 0, impl->in_frame.size() * sizeof(float));
    for (size_t t = 0; t < n_real; ++t) {
        const int32_t id = ids[t];
        const float *word = nullptr;
        if (id >= 0 && static_cast<uint32_t>(id) < impl->vocab_rows) {
            word = impl->word_emb.data() + static_cast<size_t>(id) * dim;
        }
        const float *pos = impl->pos_emb.data() + t * dim;
        const float *tt = impl->token_type_emb.data();
        float *row = frame + t * dim;
        for (uint32_t j = 0; j < dim; ++j) {
            row[j] = (word == nullptr ? 0.0f : word[j]) + pos[j] + tt[j];
        }
        double mean = 0.0;
        for (uint32_t j = 0; j < dim; ++j) {
            mean += row[j];
        }
        mean /= dim;
        double var = 0.0;
        for (uint32_t j = 0; j < dim; ++j) {
            const double d = row[j] - mean;
            var += d * d;
        }
        var /= dim;
        const float inv =
            static_cast<float>(1.0 / std::sqrt(var + impl->ln_eps));
        for (uint32_t j = 0; j < dim; ++j) {
            row[j] = (row[j] - static_cast<float>(mean)) * inv *
                         impl->ln_gamma[j] +
                     impl->ln_beta[j];
        }
    }
}

} // namespace

turboembed_status Pipeline::embed_into(
    const turboembed_str *texts,
    size_t n_texts,
    const turboembed_embed_options *opts,
    float *out,
    std::string *err
) {
    Impl *impl = impl_.get();
    if (impl == nullptr || impl->activated == nullptr) {
        *err = "hailo model is not loaded; call turboembed_load_model first";
        return TURBOEMBED_ERR_NOT_FOUND;
    }
    uint32_t pooling = impl->default_pooling;
    bool normalize = impl->default_normalize;
    uint32_t budget = impl->seq_len;
    if (opts != nullptr) {
        if (opts->pooling != TURBOEMBED_POOLING_DEFAULT) {
            pooling = static_cast<uint32_t>(opts->pooling);
        }
        if (opts->normalize == 0) {
            normalize = false;
        } else if (opts->normalize == 1) {
            normalize = true;
        }
        if (opts->truncate_to != 0) {
            if (opts->truncate_to < 3) {
                *err = "truncate_to below the 3-special minimum";
                return TURBOEMBED_ERR_INVALID_ARGUMENT;
            }
            budget = opts->truncate_to < impl->seq_len ? opts->truncate_to
                                                       : impl->seq_len;
        }
    }
    const int32_t cls_id = wordpiece_cls_id(impl->vocab);
    const int32_t sep_id = wordpiece_sep_id(impl->vocab);
    const uint32_t dim = impl->dim;

    for (size_t row = 0; row < n_texts; ++row) {
        size_t n_content = 0;
        const int rc = wordpiece_tokenize(
            impl->vocab,
            texts[row].ptr,
            texts[row].len,
            nullptr,
            0,
            4,
            &n_content
        );
        if (rc != 0) {
            *err = "wordpiece tokenize failed (invalid UTF-8 input?)";
            return TURBOEMBED_ERR_INVALID_ARGUMENT;
        }
        const size_t capacity = budget - 2; // [CLS] … [SEP]
        const size_t kept = n_content > capacity ? capacity : n_content;
        size_t written = 0;
        impl->ids[0] = cls_id;
        if (kept > 0) {
            if (wordpiece_tokenize(
                    impl->vocab,
                    texts[row].ptr,
                    texts[row].len,
                    impl->ids.data() + 1,
                    kept,
                    4,
                    &written
                ) != 0 ||
                written != kept) {
                *err = "wordpiece tokenize fill failed";
                return TURBOEMBED_ERR_INTERNAL;
            }
        }
        impl->ids[kept + 1] = sep_id;
        const size_t n_real = kept + 2;

        build_frame(impl, impl->ids.data(), n_real);
        hailo_status hst = hailo_vstream_write_raw_buffer(
            impl->hidden_stream, impl->in_frame.data(), impl->hidden_frame_bytes
        );
        if (hst != HAILO_SUCCESS) {
            return hailo_fail(hst, "hailo_vstream_write_raw_buffer(hidden)", err);
        }
        if (impl->mask_stream != nullptr) {
            // Additive attention bias (the official HEF contract):
            // 0.0 = attend, -10000.0 = masked; mask[q][k] = outer product
            // of the 1-D attention mask with itself.
            float *mask = impl->mask_frame.data();
            for (size_t q = 0; q < impl->seq_len; ++q) {
                float *mrow = mask + q * impl->seq_len;
                for (size_t k = 0; k < impl->seq_len; ++k) {
                    mrow[k] = (q < n_real && k < n_real) ? 0.0f : -10000.0f;
                }
            }
            hst = hailo_vstream_write_raw_buffer(
                impl->mask_stream, impl->mask_frame.data(), impl->mask_frame_bytes
            );
            if (hst != HAILO_SUCCESS) {
                return hailo_fail(hst, "hailo_vstream_write_raw_buffer(mask)", err);
            }
        }
        hst = hailo_vstream_read_raw_buffer(
            impl->output_stream, impl->out_frame.data(), impl->output_frame_bytes
        );
        if (hst != HAILO_SUCCESS) {
            return hailo_fail(hst, "hailo_vstream_read_raw_buffer", err);
        }

        float *dst = out + row * dim;
        if (pooling == TURBOEMBED_POOLING_CLS) {
            std::memcpy(dst, impl->out_frame.data(), dim * sizeof(float));
        } else {
            size_t pick = 0;
            size_t span = n_real;
            if (pooling == TURBOEMBED_POOLING_LAST) {
                pick = n_real - 1;
                span = 1;
            }
            if (span == 1) {
                std::memcpy(dst, impl->out_frame.data() + pick * dim, dim * sizeof(float));
            } else {
                for (uint32_t j = 0; j < dim; ++j) {
                    double sum = 0.0;
                    for (size_t t = 0; t < n_real; ++t) {
                        sum += impl->out_frame[t * dim + j];
                    }
                    dst[j] = static_cast<float>(sum / static_cast<double>(n_real));
                }
            }
        }
        if (normalize) {
            double sum_sq = 0.0;
            for (uint32_t j = 0; j < dim; ++j) {
                sum_sq += static_cast<double>(dst[j]) * dst[j];
            }
            if (sum_sq > 0.0) {
                const float inv = static_cast<float>(1.0 / std::sqrt(sum_sq));
                for (uint32_t j = 0; j < dim; ++j) {
                    dst[j] *= inv;
                }
            }
        }
    }
    return TURBOEMBED_OK;
}

} // namespace turboembed_hailo
