// SPDX-License-Identifier: Apache-2.0

#include "internal.hpp"

#include <algorithm>
#include <cstring>
#include <string>

namespace turborerank {

uint32_t pack_pair_ids(
    int32_t *input_ids,
    int32_t *attention_mask,
    int32_t *token_type_ids,
    int32_t *position_ids,
    uint32_t seq_capacity,
    const int32_t *query_ids,
    size_t n_query,
    const int32_t *doc_ids,
    size_t n_doc,
    Truncation truncation,
    uint32_t max_length,
    Status *status
) {
    auto fail = [&](Status s) -> uint32_t {
        if (status) {
            *status = s;
        }
        return 0;
    };

    if (input_ids == nullptr || attention_mask == nullptr ||
        token_type_ids == nullptr || position_ids == nullptr) {
        return fail(Status::InvalidArgument);
    }
    if (seq_capacity < kSpecials) {
        return fail(Status::InvalidArgument);
    }
    if (n_query > 0 && query_ids == nullptr) {
        return fail(Status::InvalidArgument);
    }
    if (n_doc > 0 && doc_ids == nullptr) {
        return fail(Status::InvalidArgument);
    }

    uint32_t cap = max_length == 0 ? kDefaultMaxLength : max_length;
    if (cap > seq_capacity) {
        cap = seq_capacity;
    }
    if (cap < kSpecials) {
        return fail(Status::InvalidArgument);
    }

    const uint32_t budget = cap - kSpecials;
    size_t nq = n_query;
    size_t nd = n_doc;

    if (nq + nd > budget) {
        switch (truncation) {
        case Truncation::Error:
            return fail(Status::InvalidArgument);
        case Truncation::QueryPriority:
            if (nq > budget) {
                nq = budget;
                nd = 0;
            } else {
                nd = budget - nq;
            }
            break;
        case Truncation::LongestFirst:
        default:
            while (nq + nd > budget) {
                if (nq >= nd && nq > 0) {
                    --nq;
                } else if (nd > 0) {
                    --nd;
                } else {
                    break;
                }
            }
            break;
        }
    }

    const uint32_t packed =
        kSpecials + static_cast<uint32_t>(nq) + static_cast<uint32_t>(nd);

    // Zero the whole row so leftover tokens cannot leak into attention.
    const size_t bytes = static_cast<size_t>(seq_capacity) * sizeof(int32_t);
    std::memset(input_ids, 0, bytes);
    std::memset(attention_mask, 0, bytes);
    std::memset(token_type_ids, 0, bytes);
    std::memset(position_ids, 0, bytes);

    uint32_t i = 0;
    input_ids[i] = kClsId;
    attention_mask[i] = 1;
    token_type_ids[i] = 0;
    position_ids[i] = static_cast<int32_t>(i);
    ++i;

    for (size_t t = 0; t < nq; ++t, ++i) {
        input_ids[i] = query_ids[t];
        attention_mask[i] = 1;
        token_type_ids[i] = 0;
        position_ids[i] = static_cast<int32_t>(i);
    }

    input_ids[i] = kSepId;
    attention_mask[i] = 1;
    token_type_ids[i] = 0;
    position_ids[i] = static_cast<int32_t>(i);
    ++i;

    for (size_t t = 0; t < nd; ++t, ++i) {
        input_ids[i] = doc_ids[t];
        attention_mask[i] = 1;
        token_type_ids[i] = 1;
        position_ids[i] = static_cast<int32_t>(i);
    }

    input_ids[i] = kSepId;
    attention_mask[i] = 1;
    token_type_ids[i] = 1;
    position_ids[i] = static_cast<int32_t>(i);
    ++i;

    // Remaining positions already zeroed (PAD, mask 0, type 0, pos 0).
    // HF still assigns position_ids = arange(seq) including pad. Match that
    // for the used capacity so a later backend can consume the full row.
    for (uint32_t p = i; p < seq_capacity; ++p) {
        position_ids[p] = static_cast<int32_t>(p);
    }

    if (status) {
        *status = Status::Ok;
    }
    return packed;
}

namespace impl {

Status pack_ids_into(
    turborerank_buffer *buffer,
    uint32_t row,
    const int32_t *query_ids,
    size_t n_query,
    const int32_t *doc_ids,
    size_t n_doc,
    turborerank_truncation truncation,
    uint32_t max_length,
    std::string *err
) {
    if (buffer == nullptr || buffer->input_ids == nullptr) {
        if (err) {
            *err = "pack_ids: null buffer";
        }
        return Status::InvalidArgument;
    }
    if (row >= buffer->batch) {
        if (err) {
            *err = "pack_ids: row out of range";
        }
        return Status::InvalidArgument;
    }
    const size_t off = static_cast<size_t>(row) * buffer->row_stride;
    Status st = Status::Ok;
    const uint32_t n = pack_pair_ids(
        buffer->input_ids + off,
        buffer->attention_mask + off,
        buffer->token_type_ids + off,
        buffer->position_ids + off,
        buffer->seq,
        query_ids,
        n_query,
        doc_ids,
        n_doc,
        static_cast<Truncation>(truncation),
        max_length == 0 ? buffer->seq : max_length,
        &st
    );
    if (st != Status::Ok) {
        if (err) {
            *err = n == 0 && truncation == TURBORERANK_TRUNC_ERROR
                       ? "pack_ids: pair exceeds max_length (TRUNC_ERROR)"
                       : "pack_ids: invalid arguments or max_length < 3";
        }
        return st;
    }
    (void)n;
    return Status::Ok;
}

} // namespace impl
} // namespace turborerank
