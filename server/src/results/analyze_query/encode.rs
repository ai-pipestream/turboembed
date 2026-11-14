//! Query-prompted dense/Matryoshka + SPLADE sparse target execution for
//! `AnalyzeQuery` (`#2339`).

use std::sync::atomic::Ordering;

use turbo::sparse::SparseOptions as Cut;
use turbo::{
    TURBO_DTYPE_F32, TURBO_NORMALIZE_L2, TURBO_NORMALIZE_NONE, TURBO_PROMPT_QUERY, TURBO_TASK_EMBED,
    turbo_embed_options, turbo_text,
};
use turbo_analysis::ModelRef;
use turbo_analysis_embed::impacts::ImpactBits;
use turbo_analysis_embed::{
    DEFAULT_IMPACT_CLIP_MAX, ImpactQuantizerConfig, VectorTier, build_tier_from_unnormalized, format_token_key,
    normalize_matryoshka_dims, validate_turboquant_options,
};
use turbo_query::OnePassQueryPlan;

use crate::api::{Failure, Result, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT, TURBO_E_UNSUPPORTED_OPTION, field};
use crate::deadline::Deadline;
use crate::offsets::OffsetMode;
use crate::priority::{Priority, check_before_exec};
use crate::proto::result::analyze_stream_response::Outcome as ProtoOutcome;
use crate::proto::result::{
    QueryDenseResultProto, QueryDenseTargetProto, QuerySpladeResultProto, QuerySpladeTargetProto, QueryVectorSliceProto,
};
use crate::results::{f32_bytes, f32s, model_to_proto};
use crate::state::State;
use crate::stream::run::read_output;

fn default_query_opts(normalize: u32) -> turbo_embed_options {
    turbo_embed_options {
        struct_size: std::mem::size_of::<turbo_embed_options>() as u32,
        truncate: 0,
        max_tokens: 0,
        prompt_role: TURBO_PROMPT_QUERY,
        normalize,
        pooling: 0,
        output_dim: 0,
        quantization: 0,
        int8_ranges: std::ptr::null(),
        layers: 0,
        reserved: 0,
    }
}

fn tier_to_slice_proto(tier: &VectorTier) -> QueryVectorSliceProto {
    QueryVectorSliceProto {
        dim: tier.dim,
        data: f32_bytes(&tier.data),
        inv_norm: tier.inv_norm.first().copied().unwrap_or(0.0),
        sha256: tier.row_sha256.first().map_or_else(|| tier.content_sha256.to_vec(), |d| d.to_vec()),
        turboquant_bits: u32::from(tier.turboquant_bits),
        turboquant_calibrate: tier.turboquant_calibrate,
        turboquant_codes: tier.turboquant_compact.clone(),
    }
}

pub(super) fn encode_dense_targets(
    state: &State,
    targets: &[QueryDenseTargetProto],
    plan: &OnePassQueryPlan,
    priority: Priority,
    deadline: &Deadline,
) -> Result<Vec<QueryDenseResultProto>> {
    let mut out = Vec::with_capacity(targets.len());
    for t in targets {
        let model_name = t.model.trim();
        if model_name.is_empty() {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "dense_targets[].model must be non-empty"));
        }
        let (_, loaded) = state.ready(model_name, "")?;
        if loaded.task != TURBO_TASK_EMBED {
            return Err(Failure::new(
                TURBO_E_UNSUPPORTED_OPTION,
                0,
                format!("model {model_name} is not a TASK_EMBED bundle"),
            ));
        }
        let mi = loaded.model.info()?;
        let full_dim = mi.dim;
        let n_out = (mi.output_dims_count as usize).min(mi.output_dims.len());
        let declared_dims = &mi.output_dims[..n_out];
        let prefixes = normalize_matryoshka_dims("analyze_query", &t.matryoshka_dims, full_dim, declared_dims)
            .map_err(|e| Failure::new(TURBO_E_UNSUPPORTED_OPTION, 0, e.to_string()))?;
        let tq = match t.turboquant_bits {
            0 => {
                if t.turboquant_calibrate {
                    return Err(Failure::new(
                        TURBO_E_INVALID_ARGUMENT,
                        0,
                        "turboquant_calibrate requires turboquant_bits in 2..=4",
                    ));
                }
                None
            }
            2..=4 => Some((t.turboquant_bits as u8, t.turboquant_calibrate)),
            other => {
                return Err(Failure::new(
                    TURBO_E_UNSUPPORTED_OPTION,
                    0,
                    format!("turboquant_bits {other}: supported values are 0, 2, 3, or 4"),
                ));
            }
        };
        validate_turboquant_options("analyze_query", full_dim, tq)
            .map_err(|e| Failure::new(TURBO_E_UNSUPPORTED_OPTION, 0, e.to_string()))?;

        let text = if t.use_residual_text && !plan.free_text.is_empty() { &plan.free_text } else { &plan.query };
        let held = loaded.acquire(priority, deadline)?;
        let slot = held.slot();
        let raw_text = [turbo_text { ptr: text.as_ptr().cast(), len: text.len() as u64 }];
        let mut opts = default_query_opts(TURBO_NORMALIZE_NONE);
        if let Err(f) = slot.session.write_text(&raw_text, &opts) {
            if f.code == TURBO_E_UNSUPPORTED_OPTION {
                opts.normalize = 0;
                slot.session.write_text(&raw_text, &opts)?;
            } else {
                return Err(f);
            }
        }
        check_before_exec(deadline, &loaded.metrics)?;
        state.stream.metrics.record_batch_size(1);
        state.stream.runs.fetch_add(1, Ordering::Relaxed);
        let result = slot.session.run()?;
        let (dim, dtype, bytes) = read_output(&result, 1)?;
        drop(held);
        if dtype != TURBO_DTYPE_F32 || dim != full_dim || bytes.len() != full_dim as usize * 4 {
            return Err(Failure::new(
                TURBO_E_INTERNAL,
                0,
                format!("unexpected dense query output shape: dim={dim}, dtype={dtype}, bytes={}", bytes.len()),
            ));
        }
        let raw_vec = f32s(&bytes);
        let empty: &[u32] = if raw_vec.iter().all(|&v| v == 0.0) { &[0] } else { &[] };
        let mref =
            ModelRef { id: field(&mi.model_id), revision: field(&mi.revision), sha256: field(&mi.manifest_sha256) };
        let l2 = mi.normalize == TURBO_NORMALIZE_L2;
        let full_tier = build_tier_from_unnormalized(
            "analyze_query",
            &mref,
            &raw_vec,
            full_dim as usize,
            full_dim,
            l2,
            empty,
            true,
            tq,
        )
        .map_err(|e| Failure::new(TURBO_E_INTERNAL, 0, e.to_string()))?;
        let mut matryoshka = Vec::with_capacity(prefixes.len());
        for m_dim in prefixes {
            let mtier = build_tier_from_unnormalized(
                "analyze_query",
                &mref,
                &raw_vec,
                full_dim as usize,
                m_dim,
                true,
                empty,
                true,
                None,
            )
            .map_err(|e| Failure::new(TURBO_E_INTERNAL, 0, e.to_string()))?;
            matryoshka.push(tier_to_slice_proto(&mtier));
        }
        let target_id = if t.target_id.is_empty() { model_name.to_owned() } else { t.target_id.clone() };
        out.push(QueryDenseResultProto {
            target_id,
            model: Some(model_to_proto(&mref)),
            full: Some(tier_to_slice_proto(&full_tier)),
            matryoshka,
        });
    }
    Ok(out)
}

pub(super) fn encode_splade_targets(
    state: &State,
    targets: &[QuerySpladeTargetProto],
    plan: &OnePassQueryPlan,
    offsets: OffsetMode,
    priority: Priority,
    deadline: &Deadline,
) -> Result<Vec<QuerySpladeResultProto>> {
    let mut out = Vec::with_capacity(targets.len());
    for t in targets {
        let model_name = t.model.trim();
        if model_name.is_empty() {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "splade_targets[].model must be non-empty"));
        }
        let (_, loaded) = state.ready(model_name, "")?;
        let embed = loaded.embed()?;
        let bits = match t.impact_bits {
            0 => None,
            8 => Some(ImpactBits::U8),
            16 => Some(ImpactBits::U16),
            b => {
                return Err(Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    0,
                    format!("splade_targets[].impact_bits {b}: must be 0, 8, or 16"),
                ));
            }
        };
        if bits.is_none() && t.impact_clip_max != 0.0 {
            return Err(Failure::new(
                TURBO_E_INVALID_ARGUMENT,
                0,
                "splade_targets[].impact_clip_max requires impact_bits to be 8 or 16",
            ));
        }
        let clip_max = if t.impact_clip_max == 0.0 {
            DEFAULT_IMPACT_CLIP_MAX
        } else if !t.impact_clip_max.is_finite() || t.impact_clip_max <= 0.0 {
            return Err(Failure::new(
                TURBO_E_INVALID_ARGUMENT,
                0,
                format!("splade_targets[].impact_clip_max {}: must be finite and > 0", t.impact_clip_max),
            ));
        } else {
            t.impact_clip_max
        };
        let use_residual = t.use_residual_text && !plan.free_text.is_empty();
        let text = if use_residual { plan.free_text.as_str() } else { plan.query.as_str() };
        let run_offsets = if use_residual { OffsetMode::Utf8 } else { offsets };
        let held = loaded.acquire(priority, deadline)?;
        check_before_exec(deadline, &loaded.metrics)?;
        held.reserve_token_ranges()?;
        let opts = default_query_opts(0);
        let cut = Cut { top_k: t.top_k, floor: t.floor };
        let outcomes = embed.run_sparse_batch(
            &state.stream,
            deadline.cancel_flag(),
            held.slot(),
            &[text],
            &opts,
            run_offsets,
            cut,
        )?;
        drop(held);
        let Some(ProtoOutcome::SparseTerms(st)) = outcomes.into_iter().next() else {
            return Err(Failure::new(TURBO_E_INTERNAL, 0, "expected SparseTerms outcome from SPLADE query batch"));
        };
        let (has_span, span_start, span_end) =
            if use_residual { (Vec::new(), Vec::new(), Vec::new()) } else { (st.has_span, st.span_start, st.span_end) };
        let (impact_bits, impact, impact_scale, impact_clip_max, l1_rounding_bound, dot_slack_bound) =
            if let Some(b) = bits {
                let cfg = ImpactQuantizerConfig { bits: b, top_k_doc: 0, top_k_query: 0, min_weight: 0.0, clip_max };
                let pairs: Vec<(u32, f32)> = st.term.iter().copied().zip(st.weight.iter().copied()).collect();
                let q = cfg.quantize_query_pairs(&pairs).map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, e))?;
                let impacts: Vec<u32> = st
                    .term
                    .iter()
                    .map(|&id| {
                        let idx = q.pairs().binary_search_by_key(&id, |&(tid, _)| tid).unwrap_or(0);
                        u32::from(q.pairs().get(idx).map_or(1, |&(_, imp)| imp))
                    })
                    .collect();
                (b.bits(), impacts, q.scale, q.clip_max, q.l1_rounding_bound, q.dot_slack_bound)
            } else {
                (0, Vec::new(), 0.0, 0.0, 0.0, 0.0)
            };
        let term_keys = if t.emit_term_keys {
            st.term
                .iter()
                .map(|&id| {
                    embed
                        .tokenizer
                        .piece(id as i32)
                        .filter(|s| !s.is_empty())
                        .map_or_else(|| format_token_key(id), str::to_owned)
                })
                .collect()
        } else {
            Vec::new()
        };
        let target_id = if t.target_id.is_empty() { model_name.to_owned() } else { t.target_id.clone() };
        out.push(QuerySpladeResultProto {
            target_id,
            model: st.model,
            term_id: st.term,
            weight: st.weight,
            has_span,
            span_start,
            span_end,
            impact_bits,
            impact,
            impact_scale,
            impact_clip_max,
            l1_rounding_bound,
            dot_slack_bound,
            term_keys,
        });
    }
    Ok(out)
}
