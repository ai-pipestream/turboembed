//! Known `struct_size` values per ABI struct.
//!
//! A caller declares the layout it was compiled against through
//! `struct_size`. The library accepts a size only when it is the end of a
//! field the struct has ever had, so every accepted prefix is a layout that
//! could have shipped: fields are only ever appended, and a prefix that ends
//! inside a field (half a pointer, half a count) is never a layout.
//! Everything else is `TURBO_E_INVALID_STRUCT_SIZE`.
//!
//! Generated from the field lists of `lib.rs` and `provider.rs` by
//! `scripts/gen-versioned.py`; regenerate when a struct changes.

use core::mem::{offset_of, size_of};

/// An ABI struct that starts with `uint32_t struct_size`.
pub trait Versioned {
    /// Every accepted `struct_size`, ascending, ending with the current size.
    const SIZES: &'static [usize];

    /// True when `size` is a layout this library understands.
    fn size_is_known(size: u32) -> bool {
        Self::SIZES.contains(&(size as usize))
    }
}

impl Versioned for super::turbo_error {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_error, field),
        offset_of!(super::turbo_error, message),
        size_of::<super::turbo_error>(),
    ];
}

impl Versioned for super::turbo_runtime_desc {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_runtime_desc, n_provider_paths),
        offset_of!(super::turbo_runtime_desc, reserved),
        offset_of!(super::turbo_runtime_desc, provider_paths),
        offset_of!(super::turbo_runtime_desc, log),
        offset_of!(super::turbo_runtime_desc, log_user_data),
        size_of::<super::turbo_runtime_desc>(),
    ];
}

impl Versioned for super::turbo_device_selector {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_device_selector, kind_mask),
        offset_of!(super::turbo_device_selector, ordinal),
        offset_of!(super::turbo_device_selector, provider_id),
        offset_of!(super::turbo_device_selector, vendor),
        size_of::<super::turbo_device_selector>(),
    ];
}

impl Versioned for super::turbo_device_info {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_device_info, ordinal),
        offset_of!(super::turbo_device_info, vendor_id),
        offset_of!(super::turbo_device_info, caps),
        offset_of!(super::turbo_device_info, memory_total),
        offset_of!(super::turbo_device_info, memory_free),
        offset_of!(super::turbo_device_info, name),
        offset_of!(super::turbo_device_info, vendor),
        offset_of!(super::turbo_device_info, provider_id),
        offset_of!(super::turbo_device_info, provider_version),
        offset_of!(super::turbo_device_info, runtime_version),
        offset_of!(super::turbo_device_info, driver_version),
        size_of::<super::turbo_device_info>(),
    ];
}

impl Versioned for super::turbo_capability {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_capability, dtype),
        offset_of!(super::turbo_capability, reference_dtype),
        offset_of!(super::turbo_capability, cosine_floor),
        offset_of!(super::turbo_capability, max_abs_error),
        offset_of!(super::turbo_capability, deterministic),
        offset_of!(super::turbo_capability, reserved),
        offset_of!(super::turbo_capability, notes),
        size_of::<super::turbo_capability>(),
    ];
}

impl Versioned for super::turbo_context_desc {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_context_desc, n_options),
        offset_of!(super::turbo_context_desc, reserved),
        offset_of!(super::turbo_context_desc, options),
        offset_of!(super::turbo_context_desc, next),
        size_of::<super::turbo_context_desc>(),
    ];
}

impl Versioned for super::turbo_buffer_desc {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_buffer_desc, dtype),
        offset_of!(super::turbo_buffer_desc, ndim),
        offset_of!(super::turbo_buffer_desc, shape),
        offset_of!(super::turbo_buffer_desc, strides),
        offset_of!(super::turbo_buffer_desc, bytes),
        offset_of!(super::turbo_buffer_desc, next),
        size_of::<super::turbo_buffer_desc>(),
    ];
}

impl Versioned for super::turbo_native_handle {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_native_handle, handle),
        offset_of!(super::turbo_native_handle, aux),
        offset_of!(super::turbo_native_handle, offset),
        size_of::<super::turbo_native_handle>(),
    ];
}

impl Versioned for super::turbo_model_desc {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_model_desc, options),
        offset_of!(super::turbo_model_desc, next),
        size_of::<super::turbo_model_desc>(),
    ];
}

impl Versioned for super::turbo_model_info {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_model_info, kind),
        offset_of!(super::turbo_model_info, modality),
        offset_of!(super::turbo_model_info, dim),
        offset_of!(super::turbo_model_info, n_labels),
        offset_of!(super::turbo_model_info, pooling),
        offset_of!(super::turbo_model_info, normalize),
        offset_of!(super::turbo_model_info, max_seq),
        offset_of!(super::turbo_model_info, max_batch),
        offset_of!(super::turbo_model_info, dtype_used),
        offset_of!(super::turbo_model_info, fully_accelerated),
        offset_of!(super::turbo_model_info, stage_placement),
        offset_of!(super::turbo_model_info, n_inputs),
        offset_of!(super::turbo_model_info, n_outputs),
        offset_of!(super::turbo_model_info, vocab_size),
        offset_of!(super::turbo_model_info, model_id),
        offset_of!(super::turbo_model_info, revision),
        offset_of!(super::turbo_model_info, tokenizer_sha256),
        offset_of!(super::turbo_model_info, provider_id),
        offset_of!(super::turbo_model_info, prefix_query),
        offset_of!(super::turbo_model_info, prefix_document),
        size_of::<super::turbo_model_info>(),
    ];
}

impl Versioned for super::turbo_tensor_info {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_tensor_info, ndim),
        offset_of!(super::turbo_tensor_info, reserved),
        offset_of!(super::turbo_tensor_info, shape),
        offset_of!(super::turbo_tensor_info, name),
        size_of::<super::turbo_tensor_info>(),
    ];
}

impl Versioned for super::turbo_session_desc {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_session_desc, max_seq),
        offset_of!(super::turbo_session_desc, n_options),
        offset_of!(super::turbo_session_desc, options),
        offset_of!(super::turbo_session_desc, next),
        size_of::<super::turbo_session_desc>(),
    ];
}

impl Versioned for super::turbo_embed_options {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_embed_options, max_tokens),
        offset_of!(super::turbo_embed_options, prompt_role),
        offset_of!(super::turbo_embed_options, normalize),
        offset_of!(super::turbo_embed_options, pooling),
        offset_of!(super::turbo_embed_options, output_dim),
        offset_of!(super::turbo_embed_options, output_dtype),
        size_of::<super::turbo_embed_options>(),
    ];
}

impl Versioned for super::turbo_rerank_options {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_rerank_options, max_tokens),
        offset_of!(super::turbo_rerank_options, top_n),
        offset_of!(super::turbo_rerank_options, return_sorted),
        offset_of!(super::turbo_rerank_options, raw_scores),
        size_of::<super::turbo_rerank_options>(),
    ];
}

impl Versioned for super::turbo_classify_options {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_classify_options, max_tokens),
        offset_of!(super::turbo_classify_options, aggregation),
        offset_of!(super::turbo_classify_options, raw_scores),
        offset_of!(super::turbo_classify_options, reserved),
        size_of::<super::turbo_classify_options>(),
    ];
}

impl Versioned for super::turbo_run_options {
    const SIZES: &'static [usize] =
        &[offset_of!(super::turbo_run_options, params), size_of::<super::turbo_run_options>()];
}

impl Versioned for super::turbo_token_batch {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_token_batch, seq),
        offset_of!(super::turbo_token_batch, row_stride),
        offset_of!(super::turbo_token_batch, ids),
        offset_of!(super::turbo_token_batch, mask),
        offset_of!(super::turbo_token_batch, types),
        size_of::<super::turbo_token_batch>(),
    ];
}

impl Versioned for super::turbo_session_stats {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_session_stats, runs),
        offset_of!(super::turbo_session_stats, host_allocs),
        offset_of!(super::turbo_session_stats, h2d_bytes),
        offset_of!(super::turbo_session_stats, d2h_bytes),
        offset_of!(super::turbo_session_stats, input_bytes),
        offset_of!(super::turbo_session_stats, output_bytes),
        offset_of!(super::turbo_session_stats, provider_allocs),
        size_of::<super::turbo_session_stats>(),
    ];
}

impl Versioned for super::turbo_result_info {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_result_info, batch),
        offset_of!(super::turbo_result_info, dim),
        offset_of!(super::turbo_result_info, dtype),
        offset_of!(super::turbo_result_info, placement),
        offset_of!(super::turbo_result_info, bytes),
        size_of::<super::turbo_result_info>(),
    ];
}

impl Versioned for super::turbo_generate_desc {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_generate_desc, min_new_tokens),
        offset_of!(super::turbo_generate_desc, n_sequences),
        offset_of!(super::turbo_generate_desc, temperature),
        offset_of!(super::turbo_generate_desc, top_k),
        offset_of!(super::turbo_generate_desc, top_p),
        offset_of!(super::turbo_generate_desc, min_p),
        offset_of!(super::turbo_generate_desc, repeat_penalty),
        offset_of!(super::turbo_generate_desc, presence_penalty),
        offset_of!(super::turbo_generate_desc, frequency_penalty),
        offset_of!(super::turbo_generate_desc, has_seed),
        offset_of!(super::turbo_generate_desc, seed),
        offset_of!(super::turbo_generate_desc, n_stop),
        offset_of!(super::turbo_generate_desc, n_stop_tokens),
        offset_of!(super::turbo_generate_desc, stop),
        offset_of!(super::turbo_generate_desc, stop_tokens),
        offset_of!(super::turbo_generate_desc, n_logit_bias),
        offset_of!(super::turbo_generate_desc, logprobs),
        offset_of!(super::turbo_generate_desc, logit_bias),
        offset_of!(super::turbo_generate_desc, structured_kind),
        offset_of!(super::turbo_generate_desc, echo),
        offset_of!(super::turbo_generate_desc, structured),
        offset_of!(super::turbo_generate_desc, n_tools),
        offset_of!(super::turbo_generate_desc, n_options),
        offset_of!(super::turbo_generate_desc, tools),
        offset_of!(super::turbo_generate_desc, options),
        size_of::<super::turbo_generate_desc>(),
    ];
}

impl Versioned for super::turbo_generation_chunk {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_generation_chunk, n_tokens),
        offset_of!(super::turbo_generation_chunk, n_logprobs),
        offset_of!(super::turbo_generation_chunk, tokens),
        offset_of!(super::turbo_generation_chunk, text),
        offset_of!(super::turbo_generation_chunk, logprobs),
        offset_of!(super::turbo_generation_chunk, done),
        offset_of!(super::turbo_generation_chunk, finish_reason),
        offset_of!(super::turbo_generation_chunk, prompt_tokens),
        offset_of!(super::turbo_generation_chunk, generated_tokens),
        size_of::<super::turbo_generation_chunk>(),
    ];
}

impl Versioned for super::turbo_encode_options {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_encode_options, truncate),
        offset_of!(super::turbo_encode_options, max_tokens),
        offset_of!(super::turbo_encode_options, pad_to),
        offset_of!(super::turbo_encode_options, prompt_role),
        size_of::<super::turbo_encode_options>(),
    ];
}

impl Versioned for super::turbo_tokenizer_info {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_tokenizer_info, max_seq),
        offset_of!(super::turbo_tokenizer_info, specials_per_sequence),
        offset_of!(super::turbo_tokenizer_info, pad_id),
        offset_of!(super::turbo_tokenizer_info, bos_id),
        offset_of!(super::turbo_tokenizer_info, eos_id),
        offset_of!(super::turbo_tokenizer_info, unk_id),
        offset_of!(super::turbo_tokenizer_info, kind),
        offset_of!(super::turbo_tokenizer_info, sha256),
        size_of::<super::turbo_tokenizer_info>(),
    ];
}

impl Versioned for super::turbo_chunk_desc {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_chunk_desc, reserved_tokens),
        offset_of!(super::turbo_chunk_desc, overlap_tokens),
        size_of::<super::turbo_chunk_desc>(),
    ];
}

impl Versioned for super::turbo_provider_buffer {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_provider_buffer, handle),
        offset_of!(super::turbo_provider_buffer, host_ptr),
        offset_of!(super::turbo_provider_buffer, desc),
        size_of::<super::turbo_provider_buffer>(),
    ];
}

impl Versioned for super::turbo_provider_output {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_provider_output, name),
        offset_of!(super::turbo_provider_output, buffer),
        offset_of!(super::turbo_provider_output, shape),
        size_of::<super::turbo_provider_output>(),
    ];
}

impl Versioned for super::turbo_provider_result {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_provider_result, outputs),
        offset_of!(super::turbo_provider_result, n_spans),
        offset_of!(super::turbo_provider_result, reserved),
        offset_of!(super::turbo_provider_result, spans),
        size_of::<super::turbo_provider_result>(),
    ];
}

impl Versioned for super::turbo_provider_vtbl {
    const SIZES: &'static [usize] = &[
        offset_of!(super::turbo_provider_vtbl, id),
        offset_of!(super::turbo_provider_vtbl, version),
        offset_of!(super::turbo_provider_vtbl, state),
        offset_of!(super::turbo_provider_vtbl, device_count),
        offset_of!(super::turbo_provider_vtbl, device_info),
        offset_of!(super::turbo_provider_vtbl, capability),
        offset_of!(super::turbo_provider_vtbl, can_run),
        offset_of!(super::turbo_provider_vtbl, context_create),
        offset_of!(super::turbo_provider_vtbl, context_release),
        offset_of!(super::turbo_provider_vtbl, buffer_alloc),
        offset_of!(super::turbo_provider_vtbl, buffer_import),
        offset_of!(super::turbo_provider_vtbl, buffer_read),
        offset_of!(super::turbo_provider_vtbl, buffer_export),
        offset_of!(super::turbo_provider_vtbl, buffer_release),
        offset_of!(super::turbo_provider_vtbl, model_load),
        offset_of!(super::turbo_provider_vtbl, model_info),
        offset_of!(super::turbo_provider_vtbl, model_label),
        offset_of!(super::turbo_provider_vtbl, model_io_info),
        offset_of!(super::turbo_provider_vtbl, model_release),
        offset_of!(super::turbo_provider_vtbl, session_create),
        offset_of!(super::turbo_provider_vtbl, session_write_text),
        offset_of!(super::turbo_provider_vtbl, session_write_tokens),
        offset_of!(super::turbo_provider_vtbl, session_write_pairs),
        offset_of!(super::turbo_provider_vtbl, session_write_text_classify),
        offset_of!(super::turbo_provider_vtbl, session_bind),
        offset_of!(super::turbo_provider_vtbl, session_run),
        offset_of!(super::turbo_provider_vtbl, session_stats),
        offset_of!(super::turbo_provider_vtbl, session_release),
        offset_of!(super::turbo_provider_vtbl, generation_create),
        offset_of!(super::turbo_provider_vtbl, generation_prompt),
        offset_of!(super::turbo_provider_vtbl, generation_prompt_tokens),
        offset_of!(super::turbo_provider_vtbl, generation_step),
        offset_of!(super::turbo_provider_vtbl, generation_cancel),
        offset_of!(super::turbo_provider_vtbl, generation_release),
        size_of::<super::turbo_provider_vtbl>(),
    ];
}
