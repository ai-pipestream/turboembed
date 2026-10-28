/* SPDX-License-Identifier: Apache-2.0
 *
 * The rotary families on the CUDA backend (TURBO_FAMILY_MODERNBERT,
 * TURBO_FAMILY_NOMIC_BERT, #1461): what differs from the BERT encoder,
 * held to the CPU encoder (core/src/cpu/encoder/rotary/), which is the
 * reference. The rest is the BERT session's: its packing, GEMMs,
 * attention, LayerNorms and pooling.
 *
 * At load the family's tensors are put in BERT's slots (rotary_slots),
 * so the weights' copies, layouts and F16 narrowing are the BERT
 * model's: Q, K and V from the fused Wqkv, each head's Q and K rows taken
 * pair-adjacent, (i, i + d/2) for i under d/2, so the QKV GEMM's epilogue
 * rotates neighbouring columns (EPI_QKV_ROPE); the gated MLP's gate row j
 * and up row j side by side in FFN_IN, [2 intermediate, hidden], so its
 * epilogue writes act(gate) * up from one product (EPI_GATED_*); a zero
 * vector for every absent bias. ModernBERT's final norm sits in the
 * position table's slot, which no rotary model has; its layer 0 has no
 * attention norm, and that slot is empty. The core's cos/sin table is
 * uploaded as it is: the backend has no trigonometry of its own.
 *
 * ModernBERT is pre-norm: x += attn(xn), x += mlp(xn), xn the LayerNorm
 * of x after each add (add_norm_into), layer 0's attention reading x;
 * the last add takes the final norm in place. nomic-bert is post-norm in
 * BERT's order, with the token-type row added at the lookup. A
 * ModernBERT layer l is global iff l % global_every is 0, with the global
 * base; else each query reads the keys within attention_window / 2.
 */
#ifndef TURBO_CUDA_ROTARY_H
#define TURBO_CUDA_ROTARY_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include <turbo/turbo_backend.h>

#include "kernels.h"

namespace turbo_cuda {

/* Whether family is a rotary one. */
inline bool is_rotary_family(uint32_t family) {
    return family == TURBO_FAMILY_MODERNBERT || family == TURBO_FAMILY_NOMIC_BERT;
}

/* What model_load checks of a rotary description, as the CPU's
 * (core/src/cpu/rotary.rs): an empty string when the backend runs it,
 * else why not, and *code the status. */
const char *rotary_check(const turbo_backend_model &d, int32_t *code, char *why, size_t len);

/* The model's tensors in BERT's slots (the header comment):
 * TURBO_BERT_EMBEDDING_TENSORS + layers * TURBO_BERT_LAYER_TENSORS of
 * them, F32, pointing into d's tensors or into owned, which holds the
 * reordered weights and the zeros and must outlive the copy. */
void rotary_slots(const turbo_backend_model &d, std::vector<turbo_backend_tensor> &slots,
                  std::vector<std::vector<float>> &owned);

/* Where the final norm's weight sits among the slots: ModernBERT's. */
constexpr int ROTARY_FINAL_NORM = TURBO_BERT_POSITION_EMBEDDINGS;

/* x[t] = LayerNorm(word[ids] + type[types]) (rotary.cuh); type NULL for
 * ModernBERT. x or x16 may be NULL. */
cudaError_t rotary_embed(cudaStream_t s, const int32_t *rows, const float *word, const float *type,
                         const float *ln_w, const float *ln_b, float eps, const Packing &p, int hidden, float *x,
                         uint16_t *x16, const Plan &plan);

/* x += y (or y16, the F16 product, when not NULL), and LayerNorm of the
 * sum into xn or xn16. */
cudaError_t add_norm_into(cudaStream_t s, float *x, const float *y, const uint16_t *y16, const float *ln_w,
                          const float *ln_b, float eps, const Info *info, int hidden, float *xn, uint16_t *xn16,
                          const Plan &plan);

} // namespace turbo_cuda

#endif /* TURBO_CUDA_ROTARY_H */
