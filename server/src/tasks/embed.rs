//! Serving `ANALYZE_OUTPUT_SPARSE_TERMS` and `ANALYZE_OUTPUT_TOKEN_VECTORS`
//! on `TASK_EMBED` bundles over `AnalysisService::AnalyzeStream` (`#917`).
//!
//! Both outputs run on the model's pooled `Slot` sessions via
//! `turbo_embed_write_ranges`, sharing the same micro-batching, session
//! pool, flow control, ordering, and cancellation machinery as
//! `AnalyzeVectors`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::Ordering;

use turbo::bundle::Bundle;
use turbo::head::ProjectionHead;
use turbo::manifest::{PromptRole, Truncation};
use turbo::mlm::MlmHead;
use turbo::sparse::{MaxLogPool, SparseOptions as Cut, SparseTerm};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo::{
    TURBO_DTYPE_F32, TURBO_NORMALIZE_L2, TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN, TURBO_PROMPT_DOCUMENT,
    TURBO_PROMPT_QUERY, turbo_embed_options, turbo_model_info, turbo_token_range,
};
use turbo_analysis::{Ix, Layer, ModelRef, Span, Spans, View, ViewId};
use turbo_analysis_embed::window::{Placed, Plan, Target, Which};
use turbo_analysis_embed::{Rows, normalize};
use turbo_analysis_layers::{SparseTerms, Subwords, TokenVectorWindows, Vectors};

use crate::api::{
    Failure, Result, TURBO_E_CAPACITY, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_SHAPE,
    TURBO_E_UNSUPPORTED, TURBO_E_UNSUPPORTED_OPTION,
};
use crate::offsets::OffsetMode;
use crate::proto::result::analyze_stream_response::Outcome as ProtoOutcome;
use crate::results::sparse::SparseTermsWire;
use crate::results::tokens::TokenVectors;
use crate::results::{MAX_SPARSE_ENTRIES, MAX_SPARSE_OCCURRENCES, MAX_TOKEN_VECTOR_BYTES, MAX_WINDOWS, Wire, over};
use crate::state::Slot;
use crate::stream::Shared;

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
        Ok(Self {
            tokenizer,
            mlm,
            projection,
            model,
            vocab,
            dim: info.dim,
            l2: info.normalize == TURBO_NORMALIZE_L2,
            pooling: info.pooling,
        })
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
            return Err(Failure::new(
                TURBO_E_CAPACITY,
                2,
                format!("max_tokens {} is over the session's max_seq {}", opts.max_tokens, slot.info.max_seq),
            ));
        }
        let max_seq = if opts.max_tokens == 0 { slot.info.max_seq } else { opts.max_tokens };
        let overhead = (head_len + tail_len) as u32;
        if max_seq <= overhead {
            return Err(Failure::new(
                TURBO_E_INVALID_ARGUMENT,
                2,
                format!("max_seq {max_seq} leaves no room for text beside {overhead} tokens of overhead"),
            ));
        }
        Ok(max_seq - overhead)
    }

    /// Executes a micro-batch of `ANALYZE_OUTPUT_SPARSE_TERMS` documents on `slot`.
    pub(crate) fn run_sparse_batch(
        &self,
        shared: &Shared,
        slot: &Slot,
        texts: &[&str],
        opts: &turbo_embed_options,
        offsets: OffsetMode,
        cut: Cut,
    ) -> Result<Vec<ProtoOutcome>> {
        let mlm = self.mlm.as_ref().ok_or_else(|| {
            Failure::new(TURBO_E_UNSUPPORTED_OPTION, 0, "the bundle declares no MLM head for sparse terms")
        })?;
        let (head, tail) = self.row_ends(opts.prompt_role)?;
        let width = self.window_width(slot, opts, head.len(), tail.len())?;
        let subwords: Vec<Subwords> = texts.iter().map(|t| self.tokenize_subwords(t)).collect::<Result<Vec<_>>>()?;

        let (max_batch, max_ranges) = (slot.info.max_batch, slot.max_ranges as usize);
        let mut rows = Rows::new(slot.info.max_seq, self.tokenizer.fill_id());
        let (mut ranges, mut owners, mut states) = (Vec::new(), Vec::new(), Vec::new());
        let mut pools: Vec<Option<MaxLogPool>> = vec![None; texts.len()];

        for (d, sub) in subwords.iter().enumerate() {
            for w in 0..sub.ids.len().div_ceil(width as usize) {
                let (lo, hi) = (w * width as usize, ((w + 1) * width as usize).min(sub.ids.len()));
                let mut piece = lo;
                while piece < hi {
                    if rows.len() == max_batch || ranges.len() == max_ranges {
                        flush_sparse(shared, slot, mlm, &mut rows, &mut ranges, &mut owners, &mut states, &mut pools)?;
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
        flush_sparse(shared, slot, mlm, &mut rows, &mut ranges, &mut owners, &mut states, &mut pools)?;

        let mut outcomes = Vec::with_capacity(texts.len());
        for ((&text, sub), pool) in texts.iter().zip(&subwords).zip(pools) {
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
    pub(crate) fn run_token_vectors_batch(
        &self,
        shared: &Shared,
        slot: &Slot,
        texts: &[&str],
        opts: &turbo_embed_options,
        offsets: OffsetMode,
        stride: u32,
    ) -> Result<Vec<ProtoOutcome>> {
        if self.pooling != TURBO_POOLING_MEAN {
            return Err(Failure::new(
                TURBO_E_UNSUPPORTED_OPTION,
                5,
                format!("{}: token vectors require a mean-pooled bundle", self.model.id),
            ));
        }
        let (head, tail) = self.row_ends(opts.prompt_role)?;
        let width = self.window_width(slot, opts, head.len(), tail.len())?;
        if stride > width {
            return Err(Failure::new(
                TURBO_E_INVALID_ARGUMENT,
                0,
                format!("token_stride {stride} exceeds window width {width}"),
            ));
        }
        let subwords: Vec<Subwords> = texts.iter().map(|t| self.tokenize_subwords(t)).collect::<Result<Vec<_>>>()?;
        let (hidden_dim, out_dim) = (self.dim as usize, self.projection.as_ref().map_or(self.dim, |p| p.dim as u32));
        let l2 = match opts.normalize {
            TURBO_NORMALIZE_NONE => false,
            TURBO_NORMALIZE_L2 => true,
            _ => self.l2,
        };

        let mut plan = Plan::new(width);
        let mut doc_windows: Vec<Vec<u32>> = Vec::with_capacity(subwords.len());
        let mut raw_states: Vec<Vec<f32>> = subwords.iter().map(|sub| vec![0.0f32; sub.len() * hidden_dim]).collect();

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
            doc_windows.push(window_indices);
        }
        plan.placed.sort_by_key(|x| x.window);

        let (max_batch, max_ranges) = (slot.info.max_batch, slot.max_ranges as usize);
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
                    bufs.flush(shared, slot, hidden_dim, &plan.placed, &mut raw_states)?;
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
            bufs.flush(shared, slot, hidden_dim, &plan.placed, &mut raw_states)?;
        }

        let mut outcomes = Vec::with_capacity(texts.len());
        for (((&text, sub), mut states), win_indices) in texts.iter().zip(&subwords).zip(raw_states).zip(doc_windows) {
            let data = if let Some(proj) = &self.projection {
                proj.project(&states, l2).map_err(|e| Failure::new(e.code, e.field, e.message))?
            } else {
                if l2 {
                    for row in states.chunks_exact_mut(hidden_dim) {
                        normalize(row);
                    }
                }
                states
            };
            let vectors = Vectors::<Subwords>::new(out_dim, data, Vec::new(), self.model.clone());
            let windows = build_token_vector_windows(sub, &win_indices);
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

fn build_token_vector_windows(sub: &Subwords, window_indices: &[u32]) -> TokenVectorWindows {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut i = 0;
    while i < sub.len() {
        let w = window_indices[i];
        let first = sub.spans.get(Ix::new(i as u32));
        let (start, mut end) = (first.start, first.end);
        i += 1;
        while i < sub.len() && window_indices[i] == w {
            end = sub.spans.get(Ix::new(i as u32)).end;
            i += 1;
        }
        if start < end {
            spans.push(Span::new(start, end));
        }
    }
    TokenVectorWindows { spans }
}

fn fill_sparse_terms(out: &mut SparseTerms<View>, sub: &Subwords, selected: &[SparseTerm]) {
    let mut spans: HashMap<i32, (Span, Vec<Span>)> = HashMap::new();
    for k in 0..sub.ids.len() {
        let s = sub.spans.get(Ix::new(k as u32));
        spans
            .entry(sub.ids[k])
            .and_modify(|(hull, occ)| {
                hull.start = hull.start.min(s.start);
                hull.end = hull.end.max(s.end);
                occ.push(s);
            })
            .or_insert((s, vec![s]));
    }
    for t in selected {
        match spans.get(&(t.term as i32)) {
            Some((hull, occ)) => {
                out.push(t.term, t.weight, Some(*hull));
                out.push_occurrences(occ);
            }
            None => {
                out.push(t.term, t.weight, None);
                out.push_occurrences(&[]);
            }
        }
    }
    out.unit();
}

#[allow(clippy::too_many_arguments)]
fn flush_sparse(
    shared: &Shared,
    slot: &Slot,
    mlm: &MlmHead,
    rows: &mut Rows,
    ranges: &mut Vec<turbo_token_range>,
    owners: &mut Vec<(u32, u32)>,
    states: &mut Vec<f32>,
    pools: &mut [Option<MaxLogPool>],
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    run_ranges_on_slot(shared, slot, rows, ranges, mlm.hidden as u32, states)?;
    let mut at = 0usize;
    while at < owners.len() {
        let d = owners[at].0;
        let end = owners[at..].iter().position(|&(x, _)| x != d).map_or(owners.len(), |k| at + k);
        let pool = pools[d as usize].as_mut().expect("a document with rows has a pool");
        mlm.relu_peak_into(&states[at * mlm.hidden..end * mlm.hidden], pool)
            .map_err(|e| Failure::new(e.code, e.field, e.message))?;
        at = end;
    }
    rows.clear();
    ranges.clear();
    owners.clear();
    Ok(())
}

struct TokenRunBufs {
    rows: Rows,
    ranges: Vec<turbo_token_range>,
    targets: Vec<u32>,
    flat: Vec<f32>,
}

impl TokenRunBufs {
    fn new(max_seq: u32, fill: i32) -> Self {
        Self { rows: Rows::new(max_seq, fill), ranges: Vec::new(), targets: Vec::new(), flat: Vec::new() }
    }

    fn flush(
        &mut self,
        shared: &Shared,
        slot: &Slot,
        dim: usize,
        placed: &[Placed],
        out: &mut [Vec<f32>],
    ) -> Result<()> {
        run_ranges_on_slot(shared, slot, &self.rows, &self.ranges, dim as u32, &mut self.flat)?;
        for (v, &t) in self.flat.chunks_exact(dim).zip(self.targets.iter()) {
            let x = placed[t as usize];
            let states = &mut out[x.target.doc as usize];
            let idx = x.target.item as usize;
            states[idx * dim..(idx + 1) * dim].copy_from_slice(v);
        }
        self.rows.clear();
        self.ranges.clear();
        self.targets.clear();
        Ok(())
    }
}

fn run_ranges_on_slot(
    shared: &Shared,
    slot: &Slot,
    rows: &Rows,
    ranges: &[turbo_token_range],
    dim: u32,
    out: &mut Vec<f32>,
) -> Result<()> {
    let opts = turbo_embed_options {
        struct_size: size_of::<turbo_embed_options>() as u32,
        truncate: 0,
        max_tokens: 0,
        prompt_role: 0,
        normalize: TURBO_NORMALIZE_NONE,
        pooling: TURBO_POOLING_MEAN,
        output_dim: 0,
    };
    slot.session.write_ranges(&rows.batch(), ranges, &opts).map_err(|f| {
        if f.code == TURBO_E_UNSUPPORTED && f.message.contains("does not offer embed_write_ranges") {
            Failure::new(
                TURBO_E_UNSUPPORTED,
                f.field,
                format!("the backend does not offer range pooling (embed_write_ranges): {}", f.message),
            )
        } else {
            f
        }
    })?;
    shared.runs.fetch_add(1, Ordering::Relaxed);
    let result = slot.session.run()?;
    let info = result.info()?;
    if info.dtype != TURBO_DTYPE_F32 {
        return Err(Failure::new(TURBO_E_UNSUPPORTED_OPTION, 0, "range result is not f32"));
    }
    if info.batch != ranges.len() as u32 || info.dim != dim {
        return Err(Failure::new(
            TURBO_E_INTERNAL,
            0,
            format!("range result has [{}, {}], expected [{}, {dim}]", info.batch, info.dim, ranges.len()),
        ));
    }
    let bytes = result.read(info.bytes)?;
    out.clear();
    out.reserve(ranges.len() * dim as usize);
    for chunk in bytes.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*chunk));
    }
    Ok(())
}
