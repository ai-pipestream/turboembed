//! Ranked [`TermVectorStackProto`] assembly for chunks and documents from
//! [`SparseTerms`], [`TermStats`], or [`Tokens`] (`ProjectChunks`,
//! issue #1220).

use std::any::TypeId;
use std::collections::BTreeMap;

use turbo_analysis::{Document, Ix, Key, Layer, Span, View, ViewId};
use turbo_analysis_layers::{Chunks, SparseTerms, Tokens};
use turbo_chunker::terms::{IndexedTerms, TermStats, Vocabulary};

use super::bad;
use crate::api::Failure;
use crate::offsets::Offsetter;
use crate::proto::result::{TermVectorStackProto, TermWeightSegmentProto};

/// Token indices sorted by span start, so the tokens inside a unit are one
/// binary search away rather than a scan of every token per unit. Tokens
/// may overlap, so the order is not their stored one.
struct TokenIndex<'a> {
    tokens: &'a Tokens,
    by_start: Vec<u32>,
}

impl<'a> TokenIndex<'a> {
    fn new(tokens: &'a Tokens) -> Self {
        let mut by_start: Vec<u32> = (0..tokens.spans.len() as u32).collect();
        by_start.sort_by_key(|&i| (tokens.spans.get(Ix::new(i)).start, i));
        TokenIndex { tokens, by_start }
    }

    /// The tokens wholly inside `span`, in stored order.
    fn inside(&self, span: Span) -> Vec<(usize, Span)> {
        let first = self.by_start.partition_point(|&i| self.tokens.spans.get(Ix::new(i)).start < span.start);
        let mut out: Vec<(usize, Span)> = self.by_start[first..]
            .iter()
            .map(|&i| (i as usize, self.tokens.spans.get(Ix::new(i))))
            .take_while(|(_, t)| t.start <= span.end)
            .filter(|(_, t)| t.end <= span.end)
            .collect();
        out.sort_unstable_by_key(|&(i, _)| i);
        out
    }
}

/// The view a `SparseTerms<L>` instance's spans are in: that of the `L`
/// instance it was made on, else of the default `L`.
fn sparse_view<L: Layer>(doc: &Document) -> Option<ViewId> {
    let key = Key::<SparseTerms<L>>::DEFAULT.id();
    let anchor = doc.made_on(key).iter().find(|k| k.ty == TypeId::of::<L>()).map_or("", |k| k.name);
    if TypeId::of::<L>() == TypeId::of::<Chunks>() {
        doc.get_key(&Key::<Chunks>::named(anchor)).map(|c| c.spans.view())
    } else if TypeId::of::<L>() == TypeId::of::<View>() {
        doc.get_key(&Key::<View>::named(anchor)).map(View::id)
    } else {
        None
    }
}

/// Refuses sparse terms whose spans are in another view, or that do not lie
/// on `text`: converting them with `text` would misplace them or panic.
fn check_sparse<L: Layer>(doc: &Document, sparse: &SparseTerms<L>, view: ViewId, text: &str) -> Result<(), Failure> {
    let id = SparseTerms::<L>::NAME;
    if let Some(at) = sparse_view::<L>(doc)
        && at != view
    {
        return Err(bad(format!("{id} spans are in view {}, but the projection view is {}", at.0, view.0)));
    }
    let fits = |s: &Span| {
        s.start <= s.end
            && (s.end as usize) <= text.len()
            && text.is_char_boundary(s.start as usize)
            && text.is_char_boundary(s.end as usize)
    };
    if let Some(bad_span) = sparse.span.iter().flatten().chain(&sparse.occ).find(|s| !fits(s)) {
        return Err(bad(format!(
            "{id} span {}..{} does not lie on the {}-byte projection view text",
            bad_span.start,
            bad_span.end,
            text.len()
        )));
    }
    Ok(())
}

#[derive(Clone)]
struct RawStackItem {
    term: String,
    term_id: u32,
    frequency: u32,
    weight: f32,
    is_expanded: bool,
    occurrences: Vec<Span>,
}

fn assemble_stack(
    unit: u32,
    mut items: Vec<RawStackItem>,
    max_segments: usize,
    text: &str,
    offsetter: &Offsetter,
) -> TermVectorStackProto {
    let total_terms = items.len() as u32;
    let total_frequency: u64 = items.iter().map(|r| u64::from(r.frequency)).sum();
    let total_weight_f64: f64 = items.iter().map(|r| f64::from(r.weight)).sum();
    items.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| b.frequency.cmp(&a.frequency))
            .then_with(|| a.term_id.cmp(&b.term_id))
    });
    let kept = max_segments.min(items.len());
    let (kept_slice, other_slice) = items.split_at(kept);
    let kept_weight_f64: f64 = kept_slice.iter().map(|r| f64::from(r.weight)).sum();
    let segments = kept_slice
        .iter()
        .map(|r| {
            let (occ_start, occ_end) = offsetter.spans_to_wire(text, &r.occurrences);
            TermWeightSegmentProto {
                term: r.term.clone(),
                term_id: r.term_id,
                frequency: r.frequency,
                weight: r.weight,
                share: if total_weight_f64 > 0.0 { (f64::from(r.weight) / total_weight_f64) as f32 } else { 0.0 },
                is_expanded: r.is_expanded,
                occ_start,
                occ_end,
            }
        })
        .collect();
    TermVectorStackProto {
        unit,
        segments,
        total_terms,
        total_frequency,
        total_weight: total_weight_f64 as f32,
        other_term_count: other_slice.len() as u32,
        other_frequency: other_slice.iter().map(|r| u64::from(r.frequency)).sum(),
        other_weight: (total_weight_f64 - kept_weight_f64).max(0.0) as f32,
    }
}

/// Builds per-chunk and whole-document [`TermVectorStackProto`]s from `doc`.
pub(crate) fn build_term_stacks(
    doc: &Document,
    view: ViewId,
    text: &str,
    offsetter: &Offsetter,
    max_segments: usize,
) -> Result<(Vec<TermVectorStackProto>, Option<TermVectorStackProto>), Failure> {
    let vocab = doc.get::<Vocabulary>();
    let tokens = doc.get::<Tokens>().filter(|t| t.spans.view() == view).map(TokenIndex::new);
    let tokens = tokens.as_ref();
    let indexed = doc.get::<IndexedTerms>();

    let mut chunk_stacks = Vec::new();
    if let Some(sparse_chunks) = doc.get::<SparseTerms<Chunks>>() {
        check_sparse(doc, sparse_chunks, view, text)?;
        let units = sparse_chunks.row.len().saturating_sub(1);
        for u in 0..units {
            chunk_stacks.push(stack_from_sparse(sparse_chunks, u, vocab, max_segments, text, offsetter));
        }
    } else if let (Some(v), Some(stats)) = (vocab, doc.get::<TermStats<Chunks>>()) {
        let chunks = doc.get::<Chunks>();
        for u in 0..stats.length.len() {
            let c_span = chunks.and_then(|c| (u < c.spans.len()).then(|| c.spans.get(Ix::new(u as u32))));
            chunk_stacks.push(stack_from_stats(v, stats, u, c_span, tokens, indexed, max_segments, text, offsetter));
        }
    } else if let (Some(chunks), Some(toks)) = (doc.get::<Chunks>(), tokens)
        && chunks.spans.view() == view
    {
        for u in 0..chunks.spans.len() {
            let c_span = chunks.spans.get(Ix::new(u as u32));
            chunk_stacks.push(stack_from_tokens_in_span(u as u32, toks, c_span, max_segments, text, offsetter));
        }
    }

    let doc_stack = if let Some(sparse_doc) = doc.get::<SparseTerms<View>>()
        && sparse_doc.row.len() >= 2
    {
        check_sparse(doc, sparse_doc, view, text)?;
        Some(stack_from_sparse(sparse_doc, 0, vocab, max_segments, text, offsetter))
    } else if let (Some(v), Some(stats_doc)) = (vocab, doc.get::<TermStats<View>>())
        && !stats_doc.length.is_empty()
    {
        let full = Span::new(0, text.len() as u32);
        Some(stack_from_stats(v, stats_doc, 0, Some(full), tokens, indexed, max_segments, text, offsetter))
    } else if let Some(toks) = tokens {
        let full = Span::new(0, text.len() as u32);
        Some(stack_from_tokens_in_span(0, toks, full, max_segments, text, offsetter))
    } else {
        None
    };

    Ok((chunk_stacks, doc_stack))
}

fn stack_from_sparse<L: Layer>(
    sparse: &SparseTerms<L>,
    unit: usize,
    vocab: Option<&Vocabulary>,
    max_segments: usize,
    text: &str,
    offsetter: &Offsetter,
) -> TermVectorStackProto {
    let r = sparse.row[unit] as usize..sparse.row[unit + 1] as usize;
    let mut raw = Vec::with_capacity(r.len());
    for e in r {
        let tid = sparse.term[e];
        let term = vocab
            .and_then(|v| ((tid as usize) < v.len()).then(|| v.get(Ix::new(tid))))
            .map_or_else(|| format!("term:{tid}"), str::to_owned);
        let occs: Vec<Span> = if sparse.has_occurrences() {
            sparse.occurrences(e).to_vec()
        } else if let Some(Some(sp)) = sparse.span.get(e).copied() {
            vec![sp]
        } else {
            Vec::new()
        };
        let is_expanded = occs.is_empty() && sparse.span.get(e).copied().flatten().is_none();
        let frequency = if is_expanded { 0 } else { (occs.len() as u32).max(1) };
        raw.push(RawStackItem {
            term,
            term_id: tid,
            frequency,
            weight: sparse.weight[e],
            is_expanded,
            occurrences: occs,
        });
    }
    assemble_stack(unit as u32, raw, max_segments, text, offsetter)
}

#[allow(clippy::too_many_arguments)]
fn stack_from_stats<L: Layer>(
    vocab: &Vocabulary,
    stats: &TermStats<L>,
    unit: usize,
    unit_span: Option<Span>,
    tokens: Option<&TokenIndex>,
    indexed: Option<&IndexedTerms>,
    max_segments: usize,
    text: &str,
    offsetter: &Offsetter,
) -> TermVectorStackProto {
    // The unit's tokens once, grouped by term, instead of a scan of every
    // token per entry.
    let mut by_term: BTreeMap<usize, Vec<Span>> = BTreeMap::new();
    if let (Some(sp), Some(toks), Some(idx)) = (unit_span, tokens, indexed) {
        for (ti, tsp) in toks.inside(sp) {
            if let Some(Some(t)) = idx.term.get(ti) {
                by_term.entry(t.get()).or_default().push(tsp);
            }
        }
    }
    let r = stats.row[unit] as usize..stats.row[unit + 1] as usize;
    let mut raw = Vec::with_capacity(r.len());
    for e in r {
        let tid = stats.term[e];
        let tf = stats.tf[e];
        let term_str = if tid.get() < vocab.len() { vocab.get(tid).to_owned() } else { format!("term:{}", tid.get()) };
        let occs = by_term.get(&tid.get()).cloned().unwrap_or_default();
        raw.push(RawStackItem {
            term: term_str,
            term_id: tid.get() as u32,
            frequency: tf,
            weight: tf as f32,
            is_expanded: false,
            occurrences: occs,
        });
    }
    assemble_stack(unit as u32, raw, max_segments, text, offsetter)
}

fn stack_from_tokens_in_span(
    unit: u32,
    tokens: &TokenIndex,
    span: Span,
    max_segments: usize,
    text: &str,
    offsetter: &Offsetter,
) -> TermVectorStackProto {
    let mut map: BTreeMap<String, Vec<Span>> = BTreeMap::new();
    for (_, tsp) in tokens.inside(span) {
        let s = &text[tsp.range()];
        if s.chars().any(char::is_alphanumeric) {
            map.entry(s.to_lowercase()).or_default().push(tsp);
        }
    }
    let raw: Vec<RawStackItem> = map
        .into_iter()
        .enumerate()
        .map(|(tid, (term, occs))| {
            let freq = occs.len() as u32;
            RawStackItem {
                term,
                term_id: tid as u32,
                frequency: freq,
                weight: freq as f32,
                is_expanded: false,
                occurrences: occs,
            }
        })
        .collect();
    assemble_stack(unit, raw, max_segments, text, offsetter)
}
