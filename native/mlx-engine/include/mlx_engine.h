#ifndef INFERSTREAM_MLX_ENGINE_H
#define INFERSTREAM_MLX_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* In-process native MLX engine (Swift mlx-swift + mlx-swift-lm, Metal).
 * No Python interpreter. Caller owns every string/buffer returned via out
 * pointers and must free them with mlx_engine_free / mlx_engine_free_str. */

typedef struct MlxEngine MlxEngine;

typedef void (*mlx_token_cb)(const char *token_utf8, void *user);

MlxEngine *mlx_engine_create(char **err);
void mlx_engine_destroy(MlxEngine *engine);

/* JSON: {mlx_version, device, metal_available, matmul_ok, active_memory, peak_memory} */
int mlx_engine_ping(MlxEngine *engine, char **out_json, char **err);

int mlx_engine_embed(
    MlxEngine *engine,
    const char *model_path,
    const char *const *texts,
    size_t n_texts,
    int normalize,
    float **out_vectors,
    size_t *out_rows,
    size_t *out_dim,
    char **err);

int mlx_engine_generate(
    MlxEngine *engine,
    const char *model_path,
    const char *prompt,
    uint32_t max_tokens,
    mlx_token_cb on_token,
    void *user,
    uint32_t *out_tokens,
    double *out_decode_tps,
    char **err);

void mlx_engine_free(void *ptr);
void mlx_engine_free_str(char *ptr);

#ifdef __cplusplus
}
#endif

#endif
