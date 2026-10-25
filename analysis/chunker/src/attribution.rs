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

use std::collections::BTreeMap;

use turbo_analysis::{Document, Error, Ix, Key, Layer, Span, ViewId};
use turbo_analysis_layers::{Chunks, Levels, SparseTerms, Tokens};

use crate::cascade::{FilterMode, UnitFilter};
use crate::terms::{IndexedTerms, Vocabulary};

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
    /// Refuses other families' signals, terms outside `vocab`, and an
    /// `indexed`/`tokens` length mismatch.
    pub fn add_lexical(
        &mut self,
        chunks: &Chunks,
        tokens: &Tokens,
        indexed: &IndexedTerms,
        vocab: &Vocabulary,
        query: &[LexicalTerm],
    ) -> Result<usize, String> {
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
    /// overlaps gains nothing, boost included.
    pub fn apply_unit_filter(
        &mut self,
        chunks: &Chunks,
        levels: &Levels,
        filter: &UnitFilter,
    ) -> Result<usize, String> {
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

/// One run of the document heat partition: an unscored gap (`score`
/// and `chunk` both `None`) or a stretch of one winning chunk.
#[derive(Clone, Debug, PartialEq)]
pub struct HeatSegment {
    pub span: Span,
    pub score: Option<f32>,
    pub chunk: Option<Ix<Chunks>>,
    pub matched_spans: Vec<MatchedSpan>,
}

/// The whole view partitioned into contiguous scored and unscored
/// heat segments.
#[derive(Clone, Debug, PartialEq)]
pub struct DocumentHeat {
    pub view: ViewId,
    pub segments: Vec<HeatSegment>,
}

fn invalid(detail: String) -> Error {
    Error::Invalid { layer: Key::<Chunks>::DEFAULT.id(), detail }
}

impl DocumentHeat {
    /// Partitions `view` of `doc` by the attributed level-0 chunks:
    /// where they overlap the highest `total_score` wins (ties to the
    /// lowest chunk index); elsewhere the segment is an unscored gap. A
    /// matched span rides with the segment its start falls in.
    ///
    /// Refuses an unknown view, invalid or other-view chunks, duplicate
    /// or non-level-0 attributions, a span that is not its chunk's
    /// envelope, a non-finite total, and a matched span reversed, past
    /// the view end, off a character boundary, or outside its envelope.
    pub fn build(
        doc: &Document,
        view: ViewId,
        chunks: &Chunks,
        attributions: &[ChunkAttribution],
    ) -> Result<DocumentHeat, Error> {
        if view.0 as usize >= doc.view_count() {
            return Err(Error::Bounds {
                layer: Key::<Chunks>::DEFAULT.id(),
                view,
                detail: format!("the document has {} views, view {} does not exist", doc.view_count(), view.0),
            });
        }
        chunks.check().map_err(invalid)?;
        if chunks.spans.view() != view {
            return Err(Error::Bounds {
                layer: Key::<Chunks>::DEFAULT.id(),
                view,
                detail: format!("chunks are in view {}, the heat is over view {}", chunks.spans.view().0, view.0),
            });
        }
        let text = doc.text(view);
        let len = text.len() as u32;
        let mut by_chunk: BTreeMap<u32, &ChunkAttribution> = BTreeMap::new();
        for a in attributions {
            let at = a.chunk.get();
            if at >= chunks.spans.len() {
                return Err(invalid(format!("attribution names chunk {at} of {}", chunks.spans.len())));
            }
            if chunks.level[at] != 0 {
                return Err(invalid(format!("attribution names chunk {at} at level {}, not 0", chunks.level[at])));
            }
            if by_chunk.insert(at as u32, a).is_some() {
                return Err(invalid(format!("chunk {at} is attributed twice")));
            }
            if a.span != chunks.spans.get(a.chunk) {
                return Err(invalid(format!(
                    "attribution span {}..{} is not chunk {at}'s envelope",
                    a.span.start, a.span.end
                )));
            }
            if !a.total_score.is_finite() {
                return Err(invalid(format!("chunk {at} has a non-finite total score {}", a.total_score)));
            }
            for m in &a.matched_spans {
                if m.span.start > m.span.end
                    || m.span.end > len
                    || m.span.start < a.span.start
                    || m.span.end > a.span.end
                {
                    return Err(invalid(format!(
                        "chunk {at} matched span {}..{} is outside its envelope {}..{}",
                        m.span.start, m.span.end, a.span.start, a.span.end
                    )));
                }
                if !text.is_char_boundary(m.span.start as usize) || !text.is_char_boundary(m.span.end as usize) {
                    return Err(invalid(format!(
                        "chunk {at} matched span {}..{} splits a character",
                        m.span.start, m.span.end
                    )));
                }
            }
        }
        if len == 0 {
            return Ok(DocumentHeat { view, segments: Vec::new() });
        }
        let mut cuts = vec![0u32, len];
        for a in by_chunk.values() {
            for f in chunks.spans.fragments(a.chunk) {
                cuts.push(f.start.min(len));
                cuts.push(f.end.min(len));
            }
        }
        cuts.sort_unstable();
        cuts.dedup();
        let mut segments: Vec<HeatSegment> = Vec::new();
        for w in cuts.windows(2) {
            let (s, e) = (w[0], w[1]);
            if s == e {
                continue;
            }
            let mut winner: Option<&ChunkAttribution> = None;
            for a in by_chunk.values() {
                let covers = chunks.spans.fragments(a.chunk).any(|f| f.start <= s && e <= f.end);
                if covers
                    && winner.is_none_or(|w| {
                        a.total_score.total_cmp(&w.total_score) == std::cmp::Ordering::Greater
                            || (a.total_score == w.total_score && a.chunk < w.chunk)
                    })
                {
                    winner = Some(a);
                }
            }
            let (score, chunk) = match winner {
                Some(a) => (Some(a.total_score), Some(a.chunk)),
                None => (None, None),
            };
            let matched: Vec<MatchedSpan> = match winner {
                Some(a) => a.matched_spans.iter().filter(|m| s <= m.span.start && m.span.start < e).cloned().collect(),
                None => Vec::new(),
            };
            if let Some(last) = segments.last_mut()
                && last.chunk == chunk
                && last.score == score
                && last.span.end == s
            {
                last.span.end = e;
                last.matched_spans.extend(matched);
            } else {
                segments.push(HeatSegment { span: Span { start: s, end: e }, score, chunk, matched_spans: matched });
            }
        }
        Ok(DocumentHeat { view, segments })
    }
}
