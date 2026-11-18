//! Serving `ANALYZE_OUTPUT_MULTIRES_VECTORS` on `TASK_EMBED` bundles over
//! `AnalysisService::AnalyzeStream` (`#2344`).
//!
//! Runs one encoder pass over packed subword windows (`turbo_embed_write_ranges`)
//! to simultaneously produce full-width vectors, Matryoshka normalized prefix
//! tiers, optional `RuleSplitter` sentence vectors, precomputed `inv_norm`,
//! streaming SHA-256 digests, and optional compact `TQPACK01` `TurboQuant` codes.

use std::sync::atomic::AtomicBool;

use turbo::{
    TURBO_NORMALIZE_L2, TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN, TURBO_TRUNCATE_NONE, turbo_embed_options,
    turbo_token_range,
};
use turbo_analysis::Span;
use turbo_analysis_embed::window::{Placed, Plan, Target, Which, pieces_of};
use turbo_analysis_embed::{
    MultiResVectors, Rows, build_tier_from_unnormalized, normalize_matryoshka_dims, validate_turboquant_options,
};
use turbo_sentdetect::{RuleScratch, RuleSplitter};

use crate::api::{Failure, Result, TURBO_E_INVALID_ARGUMENT, TURBO_E_UNSUPPORTED_OPTION};
use crate::offsets::OffsetMode;
use crate::proto::result::analyze_stream_response::Outcome as ProtoOutcome;
use crate::results::multires::{MultiResRequestOptions, MultiResVectorsWire};
use crate::results::{MAX_TOKEN_VECTOR_BYTES, MAX_WINDOWS, Wire, over};
use crate::state::Slot;
use crate::stream::Shared;
use crate::tasks::EmbedModel;
use crate::tasks::embed::exec::{check_alive, run_ranges_on_slot};

impl EmbedModel {
    /// Validates `EmbedOptions` and `MultiResRequestOptions` for
    /// `ANALYZE_OUTPUT_MULTIRES_VECTORS`, returning `(full_dim, prefixes)`.
    pub(crate) fn validate_multires(
        &self,
        opts: &turbo_embed_options,
        mr: &MultiResRequestOptions,
    ) -> Result<(u32, Vec<u32>)> {
        let bad_opt = |f, msg: String| Err(Failure::new(TURBO_E_UNSUPPORTED_OPTION, f, msg));
        let bad_arg = |f, msg: String| Err(Failure::new(TURBO_E_INVALID_ARGUMENT, f, msg));
        if opts.truncate != 0 && opts.truncate != TURBO_TRUNCATE_NONE {
            return bad_opt(
                1,
                format!(
                    "truncate {}: windowed multires outputs do not truncate; use 0 or TURBO_TRUNCATE_NONE",
                    opts.truncate
                ),
            );
        }
        if self.pooling != TURBO_POOLING_MEAN || (opts.pooling != 0 && opts.pooling != TURBO_POOLING_MEAN) {
            return bad_opt(5, format!("{}: multires range pooling requires a mean-pooled bundle", self.model.id));
        }
        let full_dim = match opts.output_dim {
            0 => self.dim,
            d if d > self.dim => return bad_arg(6, format!("output_dim {d} exceeds encoder width {}", self.dim)),
            d if d == self.dim || self.output_dims.contains(&d) => d,
            d => {
                return bad_opt(
                    6,
                    format!(
                        "output_dim {d}: the model was trained to be cut to {:?} besides its dim {}",
                        self.output_dims, self.dim
                    ),
                );
            }
        };
        let prefixes = normalize_matryoshka_dims("multires_vectors", &mr.matryoshka_dims, full_dim, &self.output_dims)
            .map_err(|e| Failure::new(TURBO_E_UNSUPPORTED_OPTION, 0, e.to_string()))?;
        let tq = (mr.turboquant_bits > 0).then_some((mr.turboquant_bits, mr.turboquant_calibrate));
        validate_turboquant_options("multires_vectors", full_dim, tq)
            .map_err(|e| Failure::new(TURBO_E_UNSUPPORTED_OPTION, 0, e.to_string()))?;
        Ok((full_dim, prefixes))
    }

    /// Executes a micro-batch of `ANALYZE_OUTPUT_MULTIRES_VECTORS` documents on `slot`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_multires_batch(
        &self,
        shared: &Shared,
        dead: &AtomicBool,
        slot: &Slot,
        texts: &[&str],
        opts: &turbo_embed_options,
        offsets: OffsetMode,
        mr: &MultiResRequestOptions,
    ) -> Result<Vec<ProtoOutcome>> {
        let (full_dim, prefixes) = self.validate_multires(opts, mr)?;
        let (head, tail) = self.row_ends(opts.prompt_role)?;
        let width = self.window_width(slot, opts, head.len(), tail.len())?;
        let hidden_dim = self.dim as usize;
        let l2 = match opts.normalize {
            TURBO_NORMALIZE_NONE => false,
            TURBO_NORMALIZE_L2 => true,
            _ => self.l2,
        };

        let splitter = mr.emit_sentence_vectors.then(|| RuleSplitter::new("en"));
        let mut split_scratch = RuleScratch::default();
        let mut subwords = Vec::with_capacity(texts.len());
        let mut sent_spans: Vec<Vec<Span>> = Vec::with_capacity(texts.len());

        for &t in texts {
            check_alive(dead)?;
            let sub = self.tokenize_subwords(t)?;
            let mut doc_sents = Vec::new();
            if let Some(sp) = &splitter {
                sp.split_with(t, &mut split_scratch, &mut doc_sents);
            }
            let est_rows = 1u64 + doc_sents.len() as u64;
            let est_bytes = est_rows.saturating_mul(full_dim as u64).saturating_mul(4);
            if est_bytes > MAX_TOKEN_VECTOR_BYTES {
                return Err(over("max_token_vector_bytes", "multires vector data", est_bytes));
            }
            subwords.push(sub);
            sent_spans.push(doc_sents);
        }

        let mut plan = Plan::new(width);
        let mut doc_empty: Vec<Vec<u32>> = vec![Vec::new(); texts.len()];
        let mut sent_empty: Vec<Vec<u32>> = vec![Vec::new(); texts.len()];

        for (d, (sub, (&text, sents))) in subwords.iter().zip(texts.iter().zip(&sent_spans)).enumerate() {
            let (starts, ends) = sub.spans.columns();
            let d_u32 = d as u32;
            if mr.emit_sentence_vectors {
                for (i, span) in sents.iter().enumerate() {
                    let r = pieces_of(starts, ends, span.start, span.end);
                    if r.is_empty() {
                        sent_empty[d].push(i as u32);
                    }
                    plan.layout(Target { doc: d_u32, which: Which::Sentences, item: i as u32 }, r);
                }
            }
            let doc_r = pieces_of(starts, ends, 0, text.len() as u32);
            if doc_r.is_empty() {
                doc_empty[d].push(0);
            }
            let doc_target = Target { doc: d_u32, which: Which::Chunks, item: 0 };
            if mr.emit_sentence_vectors && !sents.is_empty() {
                plan.item(doc_target, doc_r);
            } else {
                plan.layout(doc_target, doc_r);
            }
            plan.place(d_u32, starts.len() as u32)
                .map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, e.to_string()))?;
        }
        if plan.windows.len() as u64 > MAX_WINDOWS {
            return Err(over("max_windows", "encoder windows", plan.windows.len() as u64));
        }

        let mut doc_hidden: Vec<Vec<f32>> = (0..texts.len()).map(|_| vec![0.0f32; hidden_dim]).collect();
        let mut sent_hidden: Vec<Vec<f32>> = sent_spans.iter().map(|s| vec![0.0f32; s.len() * hidden_dim]).collect();

        let (max_batch, max_ranges) = (slot.info.max_batch, slot.max_ranges() as usize);
        let mut rows = Rows::new(slot.info.max_seq, self.tokenizer.fill_id());
        let mut ranges: Vec<turbo_token_range> = Vec::new();
        let mut targets: Vec<u32> = Vec::new();
        let mut flat: Vec<f32> = Vec::new();
        let head_len = head.len() as u32;
        let mut p = 0usize;

        let flush = |rows: &mut Rows,
                     ranges: &mut Vec<turbo_token_range>,
                     targets: &mut Vec<u32>,
                     flat: &mut Vec<f32>,
                     doc_hidden: &mut [Vec<f32>],
                     sent_hidden: &mut [Vec<f32>]|
         -> Result<()> {
            if rows.is_empty() {
                return Ok(());
            }
            run_ranges_on_slot(shared, dead, slot, rows, ranges, hidden_dim as u32, flat)?;
            for (v, &t) in flat.chunks_exact(hidden_dim).zip(targets.iter()) {
                let x: Placed = plan.placed[t as usize];
                let dst = match x.target.which {
                    Which::Chunks => &mut doc_hidden[x.target.doc as usize][..hidden_dim],
                    Which::Sentences => {
                        let i = x.target.item as usize;
                        &mut sent_hidden[x.target.doc as usize][i * hidden_dim..(i + 1) * hidden_dim]
                    }
                    Which::Levels(_) => unreachable!(),
                };
                if x.whole {
                    dst.copy_from_slice(v);
                } else {
                    let n = (x.end - x.start) as f32;
                    for (d_val, &s_val) in dst.iter_mut().zip(v) {
                        *d_val += s_val * n;
                    }
                }
            }
            rows.clear();
            ranges.clear();
            targets.clear();
            Ok(())
        };

        for (w, win) in plan.windows.iter().enumerate() {
            let end = p + plan.placed[p..].iter().take_while(|x| x.window as usize == w).count();
            while p < end {
                if rows.len() == max_batch || ranges.len() == max_ranges {
                    flush(&mut rows, &mut ranges, &mut targets, &mut flat, &mut doc_hidden, &mut sent_hidden)?;
                }
                let take = (end - p).min(max_ranges - ranges.len());
                let ids = &subwords[win.doc as usize].ids[win.start as usize..win.end as usize];
                let row = rows.push(&[&head, ids, &tail]);
                for (k, x) in plan.placed[p..p + take].iter().enumerate() {
                    ranges.push(turbo_token_range { row, start: head_len + x.start, end: head_len + x.end });
                    targets.push((p + k) as u32);
                }
                p += take;
            }
        }
        flush(&mut rows, &mut ranges, &mut targets, &mut flat, &mut doc_hidden, &mut sent_hidden)?;

        for sp in &plan.split {
            let inv = 1.0 / sp.pieces as f32;
            let dst = match sp.target.which {
                Which::Chunks => &mut doc_hidden[sp.target.doc as usize][..hidden_dim],
                Which::Sentences => {
                    let i = sp.target.item as usize;
                    &mut sent_hidden[sp.target.doc as usize][i * hidden_dim..(i + 1) * hidden_dim]
                }
                Which::Levels(_) => unreachable!(),
            };
            for val in dst {
                *val *= inv;
            }
        }

        let tq = (mr.turboquant_bits > 0).then_some((mr.turboquant_bits, mr.turboquant_calibrate));
        let map_err = |e: turbo_analysis::Error| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, e.to_string());
        let mut outcomes = Vec::with_capacity(texts.len());

        for d in 0..texts.len() {
            check_alive(dead)?;
            let full = build_tier_from_unnormalized(
                "multires_vectors",
                &self.model,
                &doc_hidden[d],
                hidden_dim,
                full_dim,
                l2,
                &doc_empty[d],
                mr.emit_row_sha256,
                tq,
            )
            .map_err(map_err)?;

            let mut prefix_tiers = Vec::with_capacity(prefixes.len());
            for &p_dim in &prefixes {
                prefix_tiers.push(
                    build_tier_from_unnormalized(
                        "multires_vectors",
                        &self.model,
                        &doc_hidden[d],
                        hidden_dim,
                        p_dim,
                        l2,
                        &doc_empty[d],
                        mr.emit_row_sha256,
                        None,
                    )
                    .map_err(map_err)?,
                );
            }

            let sentences = if mr.emit_sentence_vectors {
                Some(
                    build_tier_from_unnormalized(
                        "multires_vectors",
                        &self.model,
                        &sent_hidden[d],
                        hidden_dim,
                        full_dim,
                        l2,
                        &sent_empty[d],
                        mr.emit_row_sha256,
                        None,
                    )
                    .map_err(map_err)?,
                )
            } else {
                None
            };

            let wire = MultiResVectorsWire::new(
                MultiResVectors {
                    model: self.model.clone(),
                    l2_normalized: l2,
                    full,
                    prefixes: prefix_tiers,
                    sentences,
                },
                std::mem::take(&mut sent_spans[d]),
                texts[d],
                offsets,
            );
            let msg = wire.to_msg();
            MultiResVectorsWire::from_msg(&msg)?;
            outcomes.push(ProtoOutcome::MultiresVectors(msg));
        }

        Ok(outcomes)
    }
}
