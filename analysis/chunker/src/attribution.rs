//! Query-to-chunk attribution and document heat: which query signals
//! matched which exact spans inside a chunk, and the whole view
//! partitioned into scored chunk runs and unscored gaps.
//!
//! A [`ChunkAttribution`] starts empty over one level-0 chunk and grows
//! through one builder per signal family, each intersecting its
//! candidate spans with the chunk's own fragments (so a discontinuous
//! chunk is never credited for its gaps) and accumulating its own score
//! field: lexical occurrences from `IndexedTerms`, sparse occurrences
//! from `SparseTerms<Chunks>`, late-interaction winners handed in as
//! [`LateMatch`] projections (this crate does not depend on the vector
//! index that produces them), and structured-filter overlaps from spans
//! or a cascade [`UnitFilter`]. The dense score has no spans and is set
//! directly. [`ChunkAttribution::recompute_total`] sums the present
//! signal scores times the filter boost when one applies; callers who
//! fuse differently may set `total_score` themselves, since
//! [`DocumentHeat`] ranks overlapping chunks by it. Every weight, score,
//! and boost must be finite (boosts also positive), and a sparse term id
//! the lookup cannot resolve is refused, never blank-attributed.

use turbo_analysis::{Ix, Layer, Span};
use turbo_analysis_layers::{Chunks, Levels, SparseTerms, Tokens};

use crate::cascade::{FilterMode, UnitFilter};
use crate::terms::{IndexedTerms, Vocabulary};

mod heat;

pub use heat::{DocumentHeat, HeatSegment};

/// Which query signal produced a matched span.
#[derive(Clone, Debug, PartialEq)]
pub enum MatchSignal {
    ExactTerm,
    FoldedTerm,
    StemmedTerm,
    PhraseSlop { slop: u32 },
    SparseTerm { term_id: u32, expanded: bool },
    LateToken { query_token: u32, score: f32 },
    StructuredFilter { kind: String },
}

/// One exact matched run of view bytes, with the query span that
/// produced it when the caller tracks one.
#[derive(Clone, Debug, PartialEq)]
pub struct MatchedSpan {
    pub span: Span,
    pub query_span: Option<Span>,
    pub term: String,
    pub weight: f32,
    pub signal: MatchSignal,
}

/// One lexical query term for [`ChunkAttribution::add_lexical`].
#[derive(Clone, Debug, PartialEq)]
pub struct LexicalTerm {
    pub term: Ix<Vocabulary>,
    pub weight: f32,
    pub signal: MatchSignal,
    pub query_span: Option<Span>,
}

/// One late-interaction winner: the winning document span of one query
/// token and its MaxSim score, as a token index reports them.
#[derive(Clone, Debug, PartialEq)]
pub struct LateMatch {
    pub query_token: u32,
    pub span: Span,
    pub score: f32,
    pub term: String,
    pub query_span: Option<Span>,
}

/// Everything that matched inside one chunk, by signal family.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkAttribution {
    pub chunk: Ix<Chunks>,
    pub span: Span,
    pub total_score: f32,
    pub lexical_score: Option<f32>,
    pub dense_score: Option<f32>,
    pub sparse_score: Option<f32>,
    pub late_score: Option<f32>,
    pub filter_boost: Option<f32>,
    pub matched_spans: Vec<MatchedSpan>,
}

fn finite(what: &str, v: f32) -> Result<f32, String> {
    if v.is_finite() { Ok(v) } else { Err(format!("{what} is not finite: {v}")) }
}

/// The positive-length intersections of `span` with a chunk's fragments,
/// plus `span` itself when zero-width and on a fragment (edges included).
fn clip(chunks: &Chunks, chunk: Ix<Chunks>, span: Span) -> Vec<Span> {
    let mut out = Vec::new();
    for f in chunks.spans.fragments(chunk) {
        if span.start == span.end {
            if f.start <= span.start && span.start <= f.end {
                out.push(span);
            }
        } else {
            let (s, e) = (span.start.max(f.start), span.end.min(f.end));
            if s < e {
                out.push(Span { start: s, end: e });
            }
        }
    }
    out
}

impl ChunkAttribution {
    /// An empty attribution over chunk `chunk` (refused when out of range).
    pub fn new(chunks: &Chunks, chunk: Ix<Chunks>) -> Result<ChunkAttribution, String> {
        if chunk.get() >= chunks.spans.len() {
            return Err(format!("chunk {} of {}", chunk.get(), chunks.spans.len()));
        }
        Ok(ChunkAttribution {
            chunk,
            span: chunks.spans.get(chunk),
            total_score: 0.0,
            lexical_score: None,
            dense_score: None,
            sparse_score: None,
            late_score: None,
            filter_boost: None,
            matched_spans: Vec::new(),
        })
    }

    /// Refuses a `chunks` other than the one this attribution was opened
    /// over: its chunk must exist there with the same envelope.
    fn owns(&self, chunks: &Chunks) -> Result<(), String> {
        let at = self.chunk.get();
        if at >= chunks.spans.len() || chunks.spans.get(self.chunk) != self.span {
            return Err(format!("chunk {at} {}..{} is not in these chunks", self.span.start, self.span.end));
        }
        Ok(())
    }

    /// Sets `total_score` to the present signal scores' sum, times any boost.
    pub fn recompute_total(&mut self) {
        let sum = self.lexical_score.unwrap_or(0.0)
            + self.dense_score.unwrap_or(0.0)
            + self.sparse_score.unwrap_or(0.0)
            + self.late_score.unwrap_or(0.0);
        self.total_score = sum * self.filter_boost.unwrap_or(1.0);
    }

    /// Adds one matched span per occurrence of each query term among the
    /// tokens overlapping this chunk (token/chunk fragment intersections),
    /// summing weights into `lexical_score`; returns the span count.
    /// Refuses other families' signals, terms outside `vocab`, an
    /// `indexed`/`tokens` length mismatch, and tokens in another view.
    pub fn add_lexical(
        &mut self,
        chunks: &Chunks,
        tokens: &Tokens,
        indexed: &IndexedTerms,
        vocab: &Vocabulary,
        query: &[LexicalTerm],
    ) -> Result<usize, String> {
        self.owns(chunks)?;
        if tokens.spans.view() != chunks.spans.view() {
            return Err(format!(
                "tokens are in view {}, chunks in view {}",
                tokens.spans.view().0,
                chunks.spans.view().0
            ));
        }
        if indexed.term.len() != tokens.spans.len() {
            return Err(format!("IndexedTerms has {} entries, Tokens has {}", indexed.term.len(), tokens.spans.len()));
        }
        let range = tokens.spans.overlapping_range(self.span);
        let mut added = 0;
        let mut score = self.lexical_score.unwrap_or(0.0);
        for q in query {
            finite("lexical weight", q.weight)?;
            if !matches!(
                q.signal,
                MatchSignal::ExactTerm
                    | MatchSignal::FoldedTerm
                    | MatchSignal::StemmedTerm
                    | MatchSignal::PhraseSlop { .. }
            ) {
                return Err("a lexical term carries a non-lexical signal".into());
            }
            if q.term.get() >= vocab.len() {
                return Err(format!("term {} outside the {}-term vocabulary", q.term.get(), vocab.len()));
            }
            for t in range.start.get()..range.end.get() {
                if indexed.term[t] != Some(q.term) {
                    continue;
                }
                for frag in tokens.spans.fragments(Ix::new(t as u32)) {
                    for span in clip(chunks, self.chunk, frag) {
                        self.matched_spans.push(MatchedSpan {
                            span,
                            query_span: q.query_span,
                            term: vocab.get(q.term).to_string(),
                            weight: q.weight,
                            signal: q.signal.clone(),
                        });
                        score += q.weight;
                        added += 1;
                    }
                }
            }
        }
        if added > 0 || self.lexical_score.is_some() {
            self.lexical_score = Some(score);
        }
        Ok(added)
    }

    /// Adds one matched span per recorded occurrence of each
    /// `SparseTerms` entry of this chunk (only `wanted` ids when given),
    /// adding each matched entry's weight once to `sparse_score`.
    /// Expanded entries have no occurrences and add nothing.
    pub fn add_sparse<'a>(
        &mut self,
        chunks: &Chunks,
        sparse: &SparseTerms<Chunks>,
        vocab_lookup: impl Fn(u32) -> Option<&'a str>,
        wanted: Option<&std::collections::BTreeSet<u32>>,
    ) -> Result<usize, String> {
        self.owns(chunks)?;
        sparse.check()?;
        if self.chunk.get() + 1 >= sparse.row.len() {
            return Err(format!(
                "SparseTerms has {} units, chunk {} does not exist",
                sparse.row.len().saturating_sub(1),
                self.chunk.get()
            ));
        }
        let range = sparse.row[self.chunk.get()] as usize..sparse.row[self.chunk.get() + 1] as usize;
        let mut added = 0;
        let mut score = self.sparse_score.unwrap_or(0.0);
        for e in range {
            let term_id = sparse.term[e];
            if wanted.is_some_and(|w| !w.contains(&term_id)) {
                continue;
            }
            let term = vocab_lookup(term_id)
                .ok_or_else(|| format!("SparseTerms entry {e} names unresolvable term id {term_id}"))?;
            finite("sparse weight", sparse.weight[e])?;
            let occurrences: Vec<Span> = if sparse.has_occurrences() {
                sparse.occurrences(e).to_vec()
            } else {
                sparse.span.get(e).copied().flatten().into_iter().collect()
            };
            let mut matched = false;
            for occ in occurrences {
                for span in clip(chunks, self.chunk, occ) {
                    self.matched_spans.push(MatchedSpan {
                        span,
                        query_span: None,
                        term: term.to_string(),
                        weight: sparse.weight[e],
                        signal: MatchSignal::SparseTerm { term_id, expanded: false },
                    });
                    matched = true;
                    added += 1;
                }
            }
            if matched {
                score += sparse.weight[e];
            }
        }
        if added > 0 || self.sparse_score.is_some() {
            self.sparse_score = Some(score);
        }
        Ok(added)
    }

    /// Adds one matched span per intersection of each late match with
    /// the chunk's fragments, and each intersecting match's score once
    /// to `late_score`.
    pub fn add_late(&mut self, chunks: &Chunks, matches: &[LateMatch]) -> Result<usize, String> {
        self.owns(chunks)?;
        let mut added = 0;
        let mut score = self.late_score.unwrap_or(0.0);
        for m in matches {
            finite("late score", m.score)?;
            let spans = clip(chunks, self.chunk, m.span);
            if !spans.is_empty() {
                score += m.score;
            }
            for span in spans {
                self.matched_spans.push(MatchedSpan {
                    span,
                    query_span: m.query_span,
                    term: m.term.clone(),
                    weight: m.score,
                    signal: MatchSignal::LateToken { query_token: m.query_token, score: m.score },
                });
                added += 1;
            }
        }
        if added > 0 || self.late_score.is_some() {
            self.late_score = Some(score);
        }
        Ok(added)
    }

    /// Adds matched spans for the intersections of `spans` with the
    /// chunk's fragments under a structured-filter signal named `kind`.
    /// A given `boost` must be finite and positive and becomes
    /// `filter_boost` (matched spans carry it as weight, else 0).
    pub fn add_filter_spans(
        &mut self,
        chunks: &Chunks,
        kind: impl Into<String>,
        spans: &[Span],
        boost: Option<f32>,
    ) -> Result<usize, String> {
        self.owns(chunks)?;
        if let Some(b) = boost
            && !(b.is_finite() && b > 0.0)
        {
            return Err(format!("filter boost is not finite and positive: {b}"));
        }
        let kind = kind.into();
        let mut added = 0;
        for &s in spans {
            for span in clip(chunks, self.chunk, s) {
                self.matched_spans.push(MatchedSpan {
                    span,
                    query_span: None,
                    term: String::new(),
                    weight: boost.unwrap_or(0.0),
                    signal: MatchSignal::StructuredFilter { kind: kind.clone() },
                });
                added += 1;
            }
        }
        if boost.is_some() {
            self.filter_boost = boost;
        }
        Ok(added)
    }

    /// Applies a cascade [`UnitFilter`] (named units' fragments are the
    /// filter spans, a `Boost` mode the boost). A chunk no named unit
    /// overlaps gains nothing, boost included. Refuses levels in another
    /// view and units past the end of `levels`.
    pub fn apply_unit_filter(
        &mut self,
        chunks: &Chunks,
        levels: &Levels,
        filter: &UnitFilter,
    ) -> Result<usize, String> {
        self.owns(chunks)?;
        if levels.spans.view() != chunks.spans.view() {
            return Err(format!(
                "levels are in view {}, chunks in view {}",
                levels.spans.view().0,
                chunks.spans.view().0
            ));
        }
        if let Some(u) = filter.units.iter().find(|u| u.get() >= levels.spans.len()) {
            return Err(format!("filter {} names unit {} of {}", filter.name, u.get(), levels.spans.len()));
        }
        let spans: Vec<Span> =
            filter.units.iter().flat_map(|&u| levels.spans.fragments(u).collect::<Vec<_>>()).collect();
        let boost = match filter.mode {
            FilterMode::Require => None,
            FilterMode::Boost(f) => {
                if !(f.is_finite() && f > 0.0) {
                    return Err(format!("filter {} boost is not finite and positive: {f}", filter.name));
                }
                Some(f as f32)
            }
        };
        let added = self.add_filter_spans(chunks, filter.name.clone(), &spans, None)?;
        if added > 0 {
            self.filter_boost = boost;
            if let Some(b) = boost {
                for m in self.matched_spans.iter_mut().rev().take(added) {
                    m.weight = b;
                }
            }
        }
        Ok(added)
    }
}
