//! Ranked term stacks: the top terms of one unit (or a whole document)
//! as proportional segments, plus the aggregated tail.
//!
//! A stack answers "what is this chunk made of, lexically or as learned
//! sparse weights" in one deterministic shape whichever source layer it
//! came from: [`TermStats`] term frequencies, document-wide
//! [`IndexedTerms`] counts, or [`SparseTerms`] SPLADE-style weights.
//! Segments sort by weight descending, then frequency descending, then
//! term id ascending, so equal inputs rank identically everywhere.
//! Sums accumulate in `f64` in stored entry order; `share` is each kept
//! segment's fraction of the total weight, and the `other_*` fields
//! aggregate exactly the entries past the segment cap, so kept shares
//! plus `other_weight / total_weight` account for the whole unit.

use turbo_analysis::{Ix, Layer, Span};
use turbo_analysis_layers::{SparseTerms, Tokens};

use super::layer::{IndexedTerms, TermStats, Vocabulary};

/// One ranked term of a stack. `weight` is the ranking quantity (term
/// frequency for counted sources, the learned weight for sparse ones);
/// `frequency` is how often the term occurs (its term frequency for
/// counted sources, its recorded occurrence count for sparse ones);
/// `share` is `weight` over the stack's total weight. A sparse entry
/// with no recorded occurrences and no span is `is_expanded`: the model
/// proposed it, the text does not contain it.
#[derive(Clone, Debug, PartialEq)]
pub struct TermStackSegment {
    pub term: String,
    pub term_id: u32,
    pub frequency: u32,
    pub weight: f32,
    pub share: f32,
    pub is_expanded: bool,
    pub occurrences: Vec<Span>,
}

/// The ranked top terms of one unit plus the aggregated remainder.
/// `total_terms` counts every distinct entry of the source unit, kept
/// and other alike; an empty unit (or an empty document) gives all-zero
/// totals and no segments.
#[derive(Clone, Debug, PartialEq)]
pub struct TermVectorStack {
    pub segments: Vec<TermStackSegment>,
    pub total_terms: u32,
    pub total_frequency: u64,
    pub total_weight: f32,
    pub other_term_count: u32,
    pub other_frequency: u64,
    pub other_weight: f32,
}

struct Raw {
    term: String,
    term_id: u32,
    frequency: u32,
    weight: f32,
    is_expanded: bool,
    occurrences: Vec<Span>,
}

fn build(raw: Vec<Raw>, max_segments: usize) -> TermVectorStack {
    let total_frequency: u64 = raw.iter().map(|r| r.frequency as u64).sum();
    let total: f64 = raw.iter().map(|r| r.weight as f64).sum();
    let mut raw = raw;
    raw.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| b.frequency.cmp(&a.frequency))
            .then_with(|| a.term_id.cmp(&b.term_id))
    });
    let total_terms = raw.len() as u32;
    let kept = max_segments.min(raw.len());
    let (kept_raw, other_raw) = raw.split_at(kept);
    let kept_weight: f64 = kept_raw.iter().map(|r| r.weight as f64).sum();
    let segments = kept_raw
        .iter()
        .map(|r| TermStackSegment {
            term: r.term.clone(),
            term_id: r.term_id,
            frequency: r.frequency,
            weight: r.weight,
            share: if total > 0.0 { (r.weight as f64 / total) as f32 } else { 0.0 },
            is_expanded: r.is_expanded,
            occurrences: r.occurrences.clone(),
        })
        .collect();
    TermVectorStack {
        segments,
        total_terms,
        total_frequency,
        total_weight: total as f32,
        other_term_count: other_raw.len() as u32,
        other_frequency: other_raw.iter().map(|r| r.frequency as u64).sum(),
        other_weight: (total - kept_weight) as f32,
    }
}

fn unit_range(row: &[u32], unit: usize, what: &str) -> Result<std::ops::Range<usize>, String> {
    if row.is_empty() || unit + 1 >= row.len() {
        return Err(format!("{what} has {} units, unit {unit} does not exist", row.len().saturating_sub(1)));
    }
    Ok(row[unit] as usize..row[unit + 1] as usize)
}

impl TermVectorStack {
    /// The stack of one unit of `stats`: weight and frequency are the
    /// entry's term frequency. Counted layers carry no occurrence spans,
    /// so segments have none and are never expanded. Refuses an invalid
    /// `stats`, a missing unit, and a term id outside `vocab`.
    pub fn from_term_stats<L: Layer>(
        vocab: &Vocabulary,
        stats: &TermStats<L>,
        unit: Ix<L>,
        max_segments: usize,
    ) -> Result<TermVectorStack, String> {
        stats.check()?;
        let range = unit_range(&stats.row, unit.get(), "TermStats")?;
        let mut raw = Vec::with_capacity(range.len());
        for e in range {
            let term = stats.term[e];
            if term.get() >= vocab.len() {
                return Err(format!(
                    "TermStats entry {e} names term {}, the vocabulary has {} terms",
                    term.get(),
                    vocab.len()
                ));
            }
            raw.push(Raw {
                term: vocab.get(term).to_string(),
                term_id: term.get() as u32,
                frequency: stats.tf[e],
                weight: stats.tf[e] as f32,
                is_expanded: false,
                occurrences: Vec::new(),
            });
        }
        Ok(build(raw, max_segments))
    }

    /// The stack of a whole document's `indexed` terms: every token's
    /// term counted once, so weight and frequency are document term
    /// frequencies. When `tokens` is given (and aligned to `indexed`),
    /// each segment's occurrences are the envelope spans of the tokens
    /// carrying the term, in token order. Refuses a term id outside
    /// `vocab` and a `tokens` length that differs from `indexed`.
    pub fn from_indexed_terms(
        vocab: &Vocabulary,
        indexed: &IndexedTerms,
        tokens: Option<&Tokens>,
        max_segments: usize,
    ) -> Result<TermVectorStack, String> {
        if let Some(tokens) = tokens
            && tokens.spans.len() != indexed.term.len()
        {
            return Err(format!("IndexedTerms has {} entries, Tokens has {}", indexed.term.len(), tokens.spans.len()));
        }
        let mut counts: Vec<u32> = vec![0; vocab.len()];
        let mut occurrences: Vec<Vec<Span>> = if tokens.is_some() { vec![Vec::new(); vocab.len()] } else { Vec::new() };
        for (i, term) in indexed.term.iter().enumerate() {
            let Some(term) = term else { continue };
            if term.get() >= vocab.len() {
                return Err(format!(
                    "IndexedTerms entry {i} names term {}, the vocabulary has {} terms",
                    term.get(),
                    vocab.len()
                ));
            }
            counts[term.get()] += 1;
            if let Some(tokens) = tokens {
                occurrences[term.get()].push(tokens.spans.get(Ix::<Tokens>::new(i as u32)));
            }
        }
        let mut occ = occurrences.into_iter();
        let raw = counts
            .iter()
            .enumerate()
            .filter_map(|(id, &tf)| {
                let spans = occ.next().unwrap_or_default();
                (tf > 0).then(|| Raw {
                    term: vocab.get(Ix::<Vocabulary>::new(id as u32)).to_string(),
                    term_id: id as u32,
                    frequency: tf,
                    weight: tf as f32,
                    is_expanded: false,
                    occurrences: spans,
                })
            })
            .collect();
        Ok(build(raw, max_segments))
    }

    /// The stack of one unit of `sparse`: weight is the learned entry
    /// weight and frequency, its recorded occurrence count. Segments
    /// carry their occurrence spans (the entry's `span` when the
    /// instance records no occurrence table); an entry with no
    /// occurrences at all is expanded. `vocab_lookup` resolves model
    /// term ids to strings; an id it cannot resolve is refused rather
    /// than stacked under a blank term. Refuses an invalid `sparse`
    /// and a missing unit.
    pub fn from_sparse_terms<'a, L: Layer>(
        sparse: &SparseTerms<L>,
        unit: Ix<L>,
        vocab_lookup: impl Fn(u32) -> Option<&'a str>,
        max_segments: usize,
    ) -> Result<TermVectorStack, String> {
        sparse.check()?;
        let range = unit_range(&sparse.row, unit.get(), "SparseTerms")?;
        let mut raw = Vec::with_capacity(range.len());
        for e in range {
            let term_id = sparse.term[e];
            let term = vocab_lookup(term_id).ok_or_else(|| {
                format!("SparseTerms entry {e} names term id {term_id}, which the vocabulary lookup does not resolve")
            })?;
            let occurrences: Vec<Span> = if sparse.has_occurrences() {
                sparse.occurrences(e).to_vec()
            } else {
                sparse.span.get(e).copied().flatten().into_iter().collect()
            };
            raw.push(Raw {
                term: term.to_string(),
                term_id,
                frequency: occurrences.len() as u32,
                weight: sparse.weight[e],
                is_expanded: occurrences.is_empty(),
                occurrences,
            });
        }
        Ok(build(raw, max_segments))
    }
}
