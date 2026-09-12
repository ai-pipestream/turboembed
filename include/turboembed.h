#ifndef TURBOEMBED_H
#define TURBOEMBED_H

/*
 * Shared C ABI for in-process catalog embeddings.
 *
 *   embed("minilm", text) → FP32 sentence vector
 *
 * Providers (one implementation per arch; never a mock):
 *   nvidia  ORT CUDA EP + IoBinding device buffers  (--features ort-cuda)
 *   intel   OpenVINO GenAI (separate crate feature; not this translation unit)
 *   apple   MLX (separate crate feature; not this translation unit)
 *
 * Catalog aliases error if the real provider feature is not compiled in.
 * Missing CUDA / a CPU-only session is a hard error — no silent CPU fallback.
 * No Python.
 *
 * Caller owns every buffer returned via out-pointers and must free them with
 * turboembed_free / turboembed_free_str.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TurboEmbedEngine TurboEmbedEngine;

/* Create a session for `arch` ("nvidia" / "intel" / "apple").
 * catalog_path may be NULL to use the compiled-in config/catalog.toml.
 * Returns NULL on error and writes a heap string to *err. */
TurboEmbedEngine *turboembed_create(const char *arch, const char *catalog_path, char **err);

void turboembed_destroy(TurboEmbedEngine *engine);

/* embed(alias, text) — catalog alias + UTF-8 text → one FP32 vector.
 * On success returns 0, writes *out (length *out_dim). Caller frees *out.
 * On failure returns non-zero and writes *err. */
int turboembed_embed(
    TurboEmbedEngine *engine,
    const char *alias,
    const char *text,
    float **out,
    size_t *out_dim,
    char **err);

/* Live device string. NVIDIA ORT path is always "CUDA" after a successful
 * create; it is never "CPU" and never "mock". */
const char *turboembed_device(const TurboEmbedEngine *engine);

void turboembed_free(void *ptr);
void turboembed_free_str(char *ptr);

#ifdef __cplusplus
}
#endif

#endif /* TURBOEMBED_H */
