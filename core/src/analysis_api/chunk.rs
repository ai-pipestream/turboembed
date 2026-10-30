//! Dynamic-programming sentence and token chunking for `turbo_analyzer`.

use turbo_analysis::{Document, Ix, IxRange, Key, Layer, OffsetTable, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, Sentences, Tokens};

use super::engine::AnalyzerEngine;
use super::{
    ANALYSIS_RESULT_MAGIC, AnalysisResultInner, SpanDraft, SpanPoolBuilder, TURBO_ANALYSIS_LAYER_CHUNKS,
    analyzer_handle, coordinate_range, turbo_analysis_result, turbo_analyzer,
};
use crate::ffi::{call, out_ptr, text};
use crate::status::{BUNDLE_INVALID, CAPACITY, Error, INVALID_ARGUMENT, Result};
use crate::{turbo_error, turbo_text};

const DEFAULT_TARGET_TOKENS: u32 = 64;
const DEFAULT_MAX_TOKENS: u32 = 128;
const SOURCE_VIEW: u32 = ViewId::SOURCE.0 as u32;

pub(crate) fn cut_label(cut: Cut) -> &'static str {
    match cut {
        Cut::DocumentStart => "document_start",
        Cut::Sentence => "sentence",
        Cut::Section => "section",
        Cut::Heading => "heading",
        Cut::Paragraph => "paragraph",
        Cut::TopicShift => "topic_shift",
        Cut::Clause => "clause",
        Cut::Item => "item",
        Cut::Inside(_) => "inside",
        Cut::Overlap(_) => "overlap",
    }
}

fn penalty(tokens: u32, target: u32) -> i128 {
    let t = i128::from(target.max(1));
    let d = i128::from(tokens) - t;
    i128::from(Boundaries::UNIT) * d * d / (t * t)
}

struct ChunkPiece {
    span: Span,
    sentences: IxRange<Sentences>,
    tokens: (u32, u32),
    cut: Cut,
    cost: i64,
    over_budget: bool,
}

fn solve_chunks(
    sent_spans: &Spans<Sentences>,
    tok_spans: &Spans<Tokens>,
    target_tokens: u32,
    max_tokens: u32,
) -> Vec<ChunkPiece> {
    let n = sent_spans.len();
    if n == 0 {
        return Vec::new();
    }
    let mut sent_tok: Vec<(u32, u32)> = Vec::with_capacity(n);
    for idx in 0..n {
        let range = tok_spans.within(sent_spans.get(Ix::new(idx as u32)));
        sent_tok.push((range.start.get() as u32, range.end.get() as u32));
    }

    // Dynamic programming backwards from n to 0, minimizing total boundary + quadratic size penalty.
    let mut best_cost = vec![i128::MAX; n + 1];
    let mut next_cut = vec![n; n + 1];
    best_cost[n] = 0;

    for i in (0..n).rev() {
        for j in (i + 1)..=n {
            let tok_start = sent_tok[i].0;
            let tok_end = sent_tok[j - 1].1;
            let tok_count = tok_end.saturating_sub(tok_start);
            if j > i + 1 && tok_count > max_tokens {
                break;
            }
            let close_cost = if j < n { i128::from(Boundaries::UNIT) } else { 0 };
            let step_cost = penalty(tok_count, target_tokens) + close_cost;
            let candidate = step_cost.saturating_add(best_cost[j]);
            if candidate < best_cost[i] {
                best_cost[i] = candidate;
                next_cut[i] = j;
            }
        }
    }

    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let j = next_cut[i].max(i + 1);
        let tok_start = sent_tok[i].0;
        let tok_end = sent_tok[j - 1].1;
        let tok_count = tok_end.saturating_sub(tok_start);
        let base_cut = if i == 0 { Cut::DocumentStart } else { Cut::Sentence };

        if j == i + 1 && tok_count > max_tokens && max_tokens > 0 {
            // Split oversized single sentence on token windows of max_tokens.
            let mut cur = tok_start;
            let mut first = true;
            while cur < tok_end {
                let win_end = (cur + max_tokens).min(tok_end);
                let first_span = tok_spans.get(Ix::new(cur));
                let last_span = tok_spans.get(Ix::new(win_end - 1));
                let win_tokens = win_end - cur;
                let c_cost =
                    penalty(win_tokens, target_tokens).clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
                out.push(ChunkPiece {
                    span: Span::new(first_span.start, last_span.end),
                    sentences: IxRange::new(i as u32, j as u32),
                    tokens: (cur, win_end),
                    cut: if first { base_cut } else { Cut::Clause },
                    cost: c_cost,
                    over_budget: win_tokens > max_tokens,
                });
                cur = win_end;
                first = false;
            }
        } else {
            let first_sent = sent_spans.get(Ix::new(i as u32));
            let last_sent = sent_spans.get(Ix::new((j - 1) as u32));
            let close_cost = if j < n { i128::from(Boundaries::UNIT) } else { 0 };
            let step_cost = (penalty(tok_count, target_tokens) + close_cost)
                .clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
            out.push(ChunkPiece {
                span: Span::new(first_sent.start, last_sent.end),
                sentences: IxRange::new(i as u32, j as u32),
                tokens: (tok_start, tok_end),
                cut: base_cut,
                cost: step_cost,
                over_budget: tok_count > max_tokens,
            });
        }
        i = j;
    }
    out
}

impl AnalyzerEngine {
    pub(crate) fn attach_default_chunks(&self, doc: &mut Document) -> Result<()> {
        let (Some(sentences), Some(tokens)) = (doc.get::<Sentences>(), doc.get::<Tokens>()) else {
            return Ok(());
        };
        let pieces = solve_chunks(&sentences.spans, &tokens.spans, DEFAULT_TARGET_TOKENS, DEFAULT_MAX_TOKENS);
        let mut chunks = Chunks {
            spans: Spans::new(ViewId::SOURCE),
            level: Vec::with_capacity(pieces.len()),
            parent: Vec::with_capacity(pieces.len()),
            sentences: Vec::with_capacity(pieces.len()),
            cut: Vec::with_capacity(pieces.len()),
            cost: Vec::with_capacity(pieces.len()),
            model_tokens: Vec::with_capacity(pieces.len()),
            over_budget: Vec::with_capacity(pieces.len()),
            prompt_overhead: 0,
        };
        for p in pieces {
            chunks.spans.push(p.span);
            chunks.level.push(0);
            chunks.parent.push(None);
            chunks.sentences.push(p.sentences);
            chunks.cut.push(p.cut);
            chunks.cost.push(p.cost);
            chunks.model_tokens.push(p.tokens.1.saturating_sub(p.tokens.0));
            chunks.over_budget.push(p.over_budget);
        }
        doc.add(&Key::<Chunks>::DEFAULT, chunks, Provenance::by("turbo:chunker"))
            .map_err(|e| Error::new(BUNDLE_INVALID, format!("add chunks: {e}")))?;
        Ok(())
    }

    pub fn chunk_text(&self, text: &str, target_tokens: u32, max_tokens: u32) -> Result<AnalysisResultInner> {
        if target_tokens == 0 || max_tokens == 0 || target_tokens > max_tokens {
            return Err(Error::new(
                INVALID_ARGUMENT,
                "target_tokens and max_tokens must be > 0 with target_tokens <= max_tokens",
            ));
        }
        let offsets = OffsetTable::build(text);
        let mut sent_spans = Spans::<Sentences>::new(ViewId::SOURCE);
        for (span, _) in self.sentence_spans(text) {
            sent_spans.push(span);
        }
        let mut tok_spans = Spans::<Tokens>::new(ViewId::SOURCE);
        for (span, _) in self.token_spans(text) {
            tok_spans.push(span);
        }
        let pieces = solve_chunks(&sent_spans, &tok_spans, target_tokens, max_tokens);
        let mut builder = SpanPoolBuilder::with_capacity(pieces.len(), pieces.len().saturating_mul(12));
        let layer = builder.layer_ref(TURBO_ANALYSIS_LAYER_CHUNKS, Chunks::ID, "", SOURCE_VIEW)?;
        for (idx, piece) in pieces.into_iter().enumerate() {
            let item = u32::try_from(idx).map_err(|_| Error::new(CAPACITY, "chunk count exceeds u32"))?;
            let coords = coordinate_range(text, &offsets, (piece.span.start, piece.span.end))?;
            let label = builder.string_ref(cut_label(piece.cut))?;
            builder.push(
                layer,
                SpanDraft {
                    item_index: item,
                    classification: piece.tokens.1.saturating_sub(piece.tokens.0),
                    view: Some(coords),
                    source: Some(coords),
                    tokens: Some(piece.tokens),
                    label: Some(label),
                    ..SpanDraft::default()
                },
            )?;
        }
        Ok(builder.finish(Vec::new()))
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as `turbo.h` says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_analyzer_chunk_text(
    a: *mut turbo_analyzer,
    input: turbo_text,
    target_tokens: u32,
    max_tokens: u32,
    out: *mut *mut turbo_analysis_result,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let out = out_ptr(out, "out")?;
            *out = std::ptr::null_mut();
            let a = analyzer_handle(a)?;
            let s = text(input, "text")?;
            let inner = a.inner.chunk_text(s, target_tokens, max_tokens)?;
            *out = std::sync::Arc::into_raw(std::sync::Arc::new(turbo_analysis_result {
                magic: ANALYSIS_RESULT_MAGIC,
                inner,
            }))
            .cast_mut();
            Ok(())
        })
    }
}
