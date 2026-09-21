/* SPDX-License-Identifier: Apache-2.0
 *
 * TurboEmbed Hailo provider — Raspberry Pi AI HAT+ (Hailo-8 / Hailo-8L /
 * Hailo-10H). Internal to the C ABI dispatch; not part of the frozen ABI.
 *
 * Architecture (per docs/hailo-embed.md): the NPU runs only the
 * transformer encoder body from a fixed-shape HEF. The host runs
 * WordPiece tokenization (native/wordpiece), the embedding lookup +
 * embeddings LayerNorm (embedding_tables.bin, exported once from the HF
 * checkpoint by scripts/export-minilm-hailo-tables.py), padding, pooling,
 * and L2 normalization. No gRPC, no server, no Python at inference time.
 *
 * Compiled only with -DTURBOEMBED_HAILO (cargo feature `hailo`); requires
 * hailo/hailort.h (apt install hailo-all on the Pi).
 */

#ifndef TURBOEMBED_HAILO_HPP
#define TURBOEMBED_HAILO_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "turboembed.h"

namespace turboembed_hailo {

/* Live HailoRT session: one vdevice, one configured+activated network
 * group, one hidden-states input vstream (+ an optional [seq,seq]
 * attention-mask input on the official Model Zoo HEFs), one token-level
 * output vstream — plus the host-side front-end (vocab + embedding
 * tables). */
class Pipeline {
  public:
    Pipeline();
    ~Pipeline();
    Pipeline(const Pipeline &) = delete;
    Pipeline &operator=(const Pipeline &) = delete;

    /* Count Hailo devices visible to the driver (PCIe scan). Used at
     * engine-create time to fail loud before any state exists. */
    static turboembed_status scan_count(size_t *count, std::string *err);

    /* Open a vdevice over the scanned devices. Fail-loud: no Hailo
     * hardware/driver → UNAVAILABLE with apt guidance, never a CPU or
     * mock stand-in. */
    static turboembed_status open(
        const char *workspace_root,
        std::unique_ptr<Pipeline> *out,
        std::string *err
    );

    /* Load a catalog alias from <dir> = config_path or
     * <workspace_root>/models/hailo/<alias>, requiring model.hef,
     * config.json, embedding_tables.bin, vocab.txt. */
    turboembed_status load(
        const std::string &alias,
        const std::string &config_path,
        std::string *err
    );

    /* Embed `n_texts` into `out` (n_texts * embedding_dim() floats,
     * row-major). Pooling / normalize / truncate_to are all honored
     * host-side. */
    turboembed_status embed_into(
        const turboembed_str *texts,
        size_t n_texts,
        const turboembed_embed_options *opts,
        float *out,
        std::string *err
    );

    uint32_t embedding_dim() const;
    const std::string &model_alias() const;
    bool is_loaded() const;

    /* Opaque state; defined in hailo.cpp. Public name only so the .cpp's
     * anonymous-namespace helpers can take `Pipeline::Impl *`. */
    struct Impl;

  private:
    std::unique_ptr<Impl> impl_;
};

} // namespace turboembed_hailo

#endif /* TURBOEMBED_HAILO_HPP */
