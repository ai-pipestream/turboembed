//! Serving `ANALYZE_OUTPUT_SPARSE_TERMS` and `ANALYZE_OUTPUT_TOKEN_VECTORS`
//! on `TASK_EMBED` bundles over `AnalysisService::AnalyzeStream` (`#917`).
//!
//! Both outputs run on the model's pooled `Slot` sessions via
//! `turbo_embed_write_ranges`, sharing the same micro-batching, session
//! pool, flow control, ordering, and cancellation machinery as
//! `AnalyzeVectors`.

#[path = "embed_exec.rs"]
mod exec;

use std::path::Path;
use std::sync::atomic::AtomicBool;

use turbo::bundle::Bundle;
use turbo::head::ProjectionHead;
use turbo::manifest::{PromptRole, Truncation};
use turbo::mlm::MlmHead;
use turbo::sparse::{MaxLogPool, SparseOptions as Cut};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo::{
    TURBO_NORMALIZE_L2, TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN, TURBO_PROMPT_DOCUMENT, TURBO_PROMPT_QUERY,
    TURBO_TRUNCATE_NONE, turbo_embed_options, turbo_model_info, turbo_token_range,
};
use turbo_analysis::{Ix, Layer, ModelRef, Span, Spans, View, ViewId};
use turbo_analysis_embed::window::{Placed, Plan, Target, Which};
use turbo_analysis_embed::{ContextualTokens, Rows};
use turbo_analysis_layers::{SparseTerms, Subwords, TokenVectorWindows, Vectors};

use crate::api::{
    Failure, Result, TURBO_E_CAPACITY, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_SHAPE, TURBO_E_UNSUPPORTED_OPTION,
};
use crate::offsets::OffsetMode;
use crate::proto::result::AnalyzeOutput;
use crate::proto::result::analyze_stream_response::Outcome as ProtoOutcome;
use crate::results::sparse::SparseTermsWire;
use crate::results::tokens::TokenVectors;
use crate::results::{MAX_SPARSE_ENTRIES, MAX_SPARSE_OCCURRENCES, MAX_TOKEN_VECTOR_BYTES, MAX_WINDOWS, Wire, over};
use crate::state::Slot;
use crate::stream::Shared;

use exec::{TokenRunBufs, check_alive, fill_sparse_terms, flush_sparse};

/// Companion state for a `TASK_EMBED` bundle enabling learned sparse-term
/// (`SPLADE`) and per-token contextual vector (`ColBERT` / late-interaction)
/// extraction on pooled sessions.
pub struct EmbedModel {
    pub tokenizer: Tokenizer,
    pub mlm: Option<MlmHead>,
    pub projection: Option<ProjectionHead>,
    pub model: ModelRef,
    pub vocab: usize,
    pub dim: u32,
    pub output_dims: Vec<u32>,
    pub l2: bool,
    pub pooling: u32,
}

impl EmbedModel {
    /// Loads the tokenizer and optional MLM / linear projection heads for a
    /// `TASK_EMBED` bundle.
    pub fn load(bundle_path: &str, info: &turbo_model_info) -> Result<Self> {
        let map_err = |e: turbo::status::Error| Failure::new(e.code, e.field, e.message);
        let bundle = Bundle::open(Path::new(bundle_path)).map_err(map_err)?;
        let tokenizer = Tokenizer::load(&bundle).map_err(map_err)?;
        let projection = ProjectionHead::load_optional(&bundle).map_err(map_err)?;
        if let Some(proj) = &projection {
            proj.validate().map_err(map_err)?;
            if proj.hidden != info.dim as usize {
                let msg = format!("projection input width {} does not match encoder width {}", proj.hidden, info.dim);
                return Err(Failure::new(TURBO_E_INVALID_SHAPE, 0, msg));
            }
        }
        let vocab = bundle.manifest.architecture.as_ref().map_or(0, |a| a.vocab_size as usize);
        let has_sparse = bundle.manifest.embed.as_ref().is_some_and(|e| e.sparse.is_some());
        let mlm = if has_sparse {
            let head = MlmHead::load(&bundle).map_err(map_err)?;
            if head.hidden != info.dim as usize {
                let msg =
                    format!("the MLM head reads width {} hidden states, the encoder gives {}", head.hidden, info.dim);
                return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, msg));
            }
            if head.vocab != vocab {
                let msg = format!("the MLM head decodes {} terms, the model's vocabulary has {vocab}", head.vocab);
                return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, msg));
            }
            Some(head)
        } else {
            None
        };
        let m = &bundle.manifest.model;
        let model = ModelRef { id: m.id.clone(), revision: m.revision.clone(), sha256: bundle.manifest_sha256.clone() };
        let n_dims = (info.output_dims_count as usize).min(info.output_dims.len());
        Ok(Self {
            tokenizer,
            mlm,
            projection,
            model,
            vocab,
            dim: info.dim,
            output_dims: info.output_dims[..n_dims].to_vec(),
            l2: info.normalize == TURBO_NORMALIZE_L2,
            pooling: info.pooling,
        })
    }

    /// Validates `EmbedOptions` and `sparse_top_k` for `output`, returning the
    /// resolved per-token vector dimension (`output_dim`) when applicable.
    pub(crate) fn validate_options(
        &self,
        output: AnalyzeOutput,
        opts: &turbo_embed_options,
        sparse_top_k: u32,
    ) -> Result<u32> {
        let bad_opt = |f, msg: String| Err(Failure::new(TURBO_E_UNSUPPORTED_OPTION, f, msg));
        let bad_arg = |f, msg: String| Err(Failure::new(TURBO_E_INVALID_ARGUMENT, f, msg));
        if opts.truncate != 0 && opts.truncate != TURBO_TRUNCATE_NONE {
            return bad_opt(
                1,
                format!(
                    "truncate {}: windowed structured outputs do not truncate; use 0 or TURBO_TRUNCATE_NONE",
                    opts.truncate
                ),
            );
        }
        if opts.pooling != 0 && opts.pooling != TURBO_POOLING_MEAN {
            return bad_opt(
                5,
                format!("pooling {}: single-token range extraction requires 0 or TURBO_POOLING_MEAN", opts.pooling),
            );
        }
        match output {
            AnalyzeOutput::SparseTerms => {
                if self.mlm.is_none() {
                    return bad_opt(0, "the bundle declares no MLM head for sparse terms".into());
                }
                if opts.normalize != 0 && opts.normalize != TURBO_NORMALIZE_L2 {
                    return bad_opt(4, format!("normalize {}: sparse terms are always L2-normalized", opts.normalize));
                }
                if opts.output_dim > self.dim {
                    return bad_arg(6, format!("output_dim {} exceeds encoder width {}", opts.output_dim, self.dim));
                }
                if opts.output_dim != 0 && opts.output_dim != self.dim {
                    return bad_opt(
                        6,
                        format!("output_dim {}: sparse terms decode full width {}", opts.output_dim, self.dim),
                    );
                }
                if sparse_top_k as u64 > MAX_SPARSE_ENTRIES {
                    return Err(over("max_sparse_entries", "sparse term entries", sparse_top_k as u64));
                }
                Ok(self.dim)
            }
            AnalyzeOutput::TokenVectors => {
                if self.pooling != TURBO_POOLING_MEAN {
                    return bad_opt(5, format!("{}: token vectors require a mean-pooled bundle", self.model.id));
                }
                if let Some(proj) = &self.projection {
                    let pdim = proj.dim as u32;
                    match opts.output_dim {
                        0 => Ok(pdim),
                        d if d == pdim => Ok(pdim),
                        d if d > pdim => bad_arg(6, format!("output_dim {d} is over the projection head's dim {pdim}")),
                        d => bad_opt(6, format!("output_dim {d}: projection head produces fixed dim {pdim}")),
                    }
                } else {
                    match opts.output_dim {
                        0 => Ok(self.dim),
                        d if d > self.dim => bad_arg(6, format!("output_dim {d} is over the model's dim {}", self.dim)),
                        d if d == self.dim || self.output_dims.contains(&d) => Ok(d),
                        d => bad_opt(
                            6,
                            format!(
                                "output_dim {d}: the model was trained to be cut to {:?} besides its dim {}",
                                self.output_dims, self.dim
                            ),
                        ),
                    }
                }
            }
            _ => bad_opt(0, "unsupported structured embed output".into()),
        }
    }

    fn tokenize_subwords(&self, text: &str) -> Result<Subwords> {
        let opts = Encode {
            add_special_tokens: false,
            truncation: Truncation::None,
            max_tokens: u32::MAX,
            prompt: PromptRole::None,
        };
        let (mut ids, mut raw_spans) = (Vec::new(), Vec::new());
        self.tokenizer
            .encode_spans(text, opts, &mut ids, &mut raw_spans)
            .map_err(|e| Failure::new(e.code, e.field, e.message))?;
        let mut spans = Spans::with_capacity(ViewId::SOURCE, raw_spans.len());
        for &[start, end] in &raw_spans {
            spans.push(Span::new(start, end));
        }
        Ok(Subwords { spans, ids, model: self.model.clone(), token: None })
    }

    fn row_ends(&self, prompt_role: u32) -> Result<(Vec<i32>, Vec<i32>)> {
        let prompt = match prompt_role {
            TURBO_PROMPT_QUERY => PromptRole::Query,
            TURBO_PROMPT_DOCUMENT | 0 => PromptRole::Document,
            _ => PromptRole::None,
        };
        let (mut head, tail) = self.tokenizer.template_ends();
        let bare = Encode { add_special_tokens: false, truncation: Truncation::None, max_tokens: u32::MAX, prompt };
        self.tokenizer.encode_into("", bare, &mut head).map_err(|e| Failure::new(e.code, e.field, e.message))?;
        Ok((head, tail))
    }

    fn window_width(&self, slot: &Slot, opts: &turbo_embed_options, head_len: usize, tail_len: usize) -> Result<u32> {
        if opts.max_tokens > slot.info.max_seq {
            let msg = format!("max_tokens {} is over the session's max_seq {}", opts.max_tokens, slot.info.max_seq);
            return Err(Failure::new(TURBO_E_CAPACITY, 2, msg));
        }
        let max_seq = if opts.max_tokens == 0 { slot.info.max_seq } else { opts.max_tokens };
        let overhead = (head_len + tail_len) as u32;
        if max_seq <= overhead {
            let msg = format!("max_seq {max_seq} leaves no room for text beside {overhead} tokens of overhead");
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 2, msg));
        }
        Ok(max_seq - overhead)
    }

    /// Executes a micro-batch of `ANALYZE_OUTPUT_SPARSE_TERMS` documents on `slot`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_sparse_batch(
        &self,
        shared: &Shared,
        dead: &AtomicBool,
        slot: &Slot,
        texts: &[&str],
        opts: &turbo_embed_options,
        offsets: OffsetMode,
        cut: Cut,
    ) -> Result<Vec<ProtoOutcome>> {
        self.validate_options(AnalyzeOutput::SparseTerms, opts, cut.top_k)?;
        let mlm = self.mlm.as_ref().expect("validated MLM head");
        let (head, tail) = self.row_ends(opts.prompt_role)?;
        let width = self.window_width(slot, opts, head.len(), tail.len())?;
        let mut subwords = Vec::with_capacity(texts.len());
        for &t in texts {
            check_alive(dead)?;
            let sub = self.tokenize_subwords(t)?;
            if sub.len() as u64 > MAX_SPARSE_OCCURRENCES {
                return Err(over("max_sparse_occurrences", "occurrence spans", sub.len() as u64));
            }
            let wins = sub.ids.len().div_ceil(width as usize) as u64;
            if wins > MAX_WINDOWS {
                return Err(over("max_windows", "encoder windows", wins));
            }
            subwords.push(sub);
        }

        let (max_batch, max_ranges) = (slot.info.max_batch, slot.max_ranges() as usize);
        let mut rows = Rows::new(slot.info.max_seq, self.tokenizer.fill_id());
        let (mut ranges, mut owners, mut states) = (Vec::new(), Vec::new(), Vec::new());
        let mut pools: Vec<Option<MaxLogPool>> = vec![None; texts.len()];

        for (d, sub) in subwords.iter().enumerate() {
            for w in 0..sub.ids.len().div_ceil(width as usize) {
                let (lo, hi) = (w * width as usize, ((w + 1) * width as usize).min(sub.ids.len()));
                let mut piece = lo;
                while piece < hi {
                    if rows.len() == max_batch || ranges.len() == max_ranges {
                        flush_sparse(
                            shared,
                            dead,
                            slot,
                            mlm,
                            &mut rows,
                            &mut ranges,
                            &mut owners,
                            &mut states,
                            &mut pools,
                        )?;
                    }
                    let take = (hi - piece).min(max_ranges - ranges.len());
                    let row = rows.push(&[&head, &sub.ids[lo..hi], &tail]);
                    for p in piece..piece + take {
                        let at = head.len() as u32 + (p - lo) as u32;
                        ranges.push(turbo_token_range { row, start: at, end: at + 1 });
                        owners.push((d as u32, p as u32));
                    }
                    if pools[d].is_none() {
                        pools[d] = Some(MaxLogPool::new(self.vocab));
                    }
                    piece += take;
                }
            }
        }
        flush_sparse(shared, dead, slot, mlm, &mut rows, &mut ranges, &mut owners, &mut states, &mut pools)?;

        let mut outcomes = Vec::with_capacity(texts.len());
        for ((&text, sub), pool) in texts.iter().zip(&subwords).zip(pools) {
            check_alive(dead)?;
            let pool = pool.unwrap_or_else(|| MaxLogPool::new(self.vocab));
            let selected = pool.terms(&cut).map_err(|e| Failure::new(e.code, e.field, e.message))?;
            let mut terms = SparseTerms::<View>::new(self.model.clone());
            fill_sparse_terms(&mut terms, sub, &selected);
            let msg = SparseTermsWire::new(terms, text, offsets).to_msg();
            if msg.term.len() as u64 > MAX_SPARSE_ENTRIES {
                return Err(over("max_sparse_entries", "sparse term entries", msg.term.len() as u64));
            }
            if msg.occ_start.len() as u64 > MAX_SPARSE_OCCURRENCES {
                return Err(over("max_sparse_occurrences", "occurrence spans", msg.occ_start.len() as u64));
            }
            outcomes.push(ProtoOutcome::SparseTerms(msg));
        }
        Ok(outcomes)
    }

    /// Executes a micro-batch of `ANALYZE_OUTPUT_TOKEN_VECTORS` documents on `slot`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_token_vectors_batch(
        &self,
        shared: &Shared,
        dead: &AtomicBool,
        slot: &Slot,
        texts: &[&str],
        opts: &turbo_embed_options,
        offsets: OffsetMode,
        stride: u32,
    ) -> Result<Vec<ProtoOutcome>> {
        let out_dim = self.validate_options(AnalyzeOutput::TokenVectors, opts, 0)?;
        let (head, tail) = self.row_ends(opts.prompt_role)?;
        let width = self.window_width(slot, opts, head.len(), tail.len())?;
        if stride > width {
            return Err(Failure::new(
                TURBO_E_INVALID_ARGUMENT,
                0,
                format!("token_stride {stride} exceeds window width {width}"),
            ));
        }
        let mut subwords = Vec::with_capacity(texts.len());
        for &t in texts {
            check_alive(dead)?;
            let sub = self.tokenize_subwords(t)?;
            let token_bytes =
                (sub.len() as u64).checked_mul(out_dim as u64).and_then(|v| v.checked_mul(4)).unwrap_or(u64::MAX);
            if token_bytes > MAX_TOKEN_VECTOR_BYTES {
                return Err(over("max_token_vector_bytes", "token vector data", token_bytes));
            }
            subwords.push(sub);
        }

        let (hidden_dim, out_width) = (self.dim as usize, out_dim as usize);
        let l2 = match opts.normalize {
            TURBO_NORMALIZE_NONE => false,
            TURBO_NORMALIZE_L2 => true,
            _ => self.l2,
        };

        let mut plan = Plan::new(width);
        let mut doc_windows: Vec<TokenVectorWindows> = Vec::with_capacity(subwords.len());
        for (d, sub) in subwords.iter().enumerate() {
            let win_offset = plan.windows.len() as u32;
            let placements = plan
                .place_tokens(d as u32, sub.len() as u32, stride)
                .map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, e.to_string()))?;
            let mut window_indices = Vec::with_capacity(placements.len());
            for tp in placements {
                window_indices.push(tp.window - win_offset);
                plan.placed.push(Placed {
                    window: tp.window,
                    start: tp.offset,
                    end: tp.offset + 1,
                    target: Target { doc: d as u32, which: Which::Sentences, item: tp.piece },
                    whole: true,
                });
            }
            let windows = ContextualTokens::token_vector_windows_from(
                (0..sub.len()).map(|i| (sub.spans.get(Ix::new(i as u32)), window_indices[i])),
                ViewId::SOURCE,
            );
            if windows.spans.len() as u64 > MAX_WINDOWS {
                return Err(over("max_windows", "encoder windows", windows.spans.len() as u64));
            }
            doc_windows.push(windows);
        }
        plan.placed.sort_by_key(|x| x.window);

        let mut out_states: Vec<Vec<f32>> = subwords.iter().map(|sub| vec![0.0f32; sub.len() * out_width]).collect();
        let (max_batch, max_ranges) = (slot.info.max_batch, slot.max_ranges() as usize);
        let mut bufs = TokenRunBufs::new(slot.info.max_seq, self.tokenizer.fill_id());
        let (head_len, mut p) = (head.len() as u32, 0usize);

        for (w, win) in plan.windows.iter().enumerate() {
            let end = p + plan.placed[p..].iter().take_while(|x| x.window as usize == w).count();
            if p == end {
                continue;
            }
            let ids = &subwords[win.doc as usize].ids[win.start as usize..win.end as usize];
            while p < end {
                if bufs.rows.len() == max_batch || bufs.ranges.len() == max_ranges {
                    bufs.flush(
                        shared,
                        dead,
                        slot,
                        hidden_dim,
                        out_width,
                        self.projection.as_ref(),
                        l2,
                        &plan.placed,
                        &mut out_states,
                    )?;
                }
                let row = bufs.rows.push(&[&head, ids, &tail]);
                let take = (end - p).min(max_ranges - bufs.ranges.len());
                for (k, x) in plan.placed[p..p + take].iter().enumerate() {
                    bufs.ranges.push(turbo_token_range { row, start: head_len + x.start, end: head_len + x.end });
                    bufs.targets.push((p + k) as u32);
                }
                p += take;
            }
        }
        if !bufs.rows.is_empty() {
            bufs.flush(
                shared,
                dead,
                slot,
                hidden_dim,
                out_width,
                self.projection.as_ref(),
                l2,
                &plan.placed,
                &mut out_states,
            )?;
        }

        let mut outcomes = Vec::with_capacity(texts.len());
        for ((&text, data), windows) in texts.iter().zip(out_states).zip(doc_windows) {
            check_alive(dead)?;
            let vectors = Vectors::<Subwords>::new(out_dim, data, Vec::new(), self.model.clone());
            let msg = TokenVectors::new(&vectors, &windows).with_offsets(text, offsets).to_msg();
            if msg.data.len() as u64 > MAX_TOKEN_VECTOR_BYTES {
                return Err(over("max_token_vector_bytes", "token vector data", msg.data.len() as u64));
            }
            if msg.window_start.len() as u64 > MAX_WINDOWS {
                return Err(over("max_windows", "encoder windows", msg.window_start.len() as u64));
            }
            outcomes.push(ProtoOutcome::TokenVectors(msg));
        }
        Ok(outcomes)
    }
}
