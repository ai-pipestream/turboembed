//! Syntax-aware sub-sentence clause boundaries (`Cut::Clause`) for splitting
//! over-budget sentences before falling back to raw token windows.
//!
//! When a single sentence exceeds `Budget::max`, `Chunker` marks it
//! `over_budget`. When `ClauseCuts` (written by `ClauseBoundaries`) is present
//! on the document, `Chunker` splits that over-budget sentence in two stages:
//! 1. **Syntactic clause DP**: runs `segment` over the sentence's candidate
//!    clauses derived from `Dependencies<Tokens>` (`parataxis`, `conj`,
//!    `advcl`, `ccomp`, `acl:relcl`, `acl`, `csubj`, `xcomp`), constituency
//!    `Phrases` (`S`, `SBAR`, `SBARQ`, `SINV`, `SQ`), shallow `PhraseChunks`,
//!    and clause-separating punctuation/coordinators.
//! 2. **Token-window fallback DP**: if an individual clause (or a sentence with
//!    no syntactic clause boundaries) still exceeds `Budget::max`, runs
//!    `segment` over the tokens of that clause, preferring phrase boundaries
//!    over mid-phrase token cuts.

mod detect;

use std::ops::Range;

use turbo_analysis::{Anchor, Annotator, Bound, Error, Ix, Layer, Refs, Span, View};
use turbo_analysis_layers::{Chunks, Cut, Dependencies, PhraseChunks, Phrases, PosTags, Sentences, Tokens};

use crate::{Budget, Lengths, Row, RunCx, no_term, saturate, segment};
use detect::build_sentence_cuts;

/// Boundary costs in `Boundaries::UNIT`s for sub-sentence cuts inside an
/// over-budget sentence.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ClauseCosts {
    /// Coordinate, parataxis, or top-level `S` clause boundary.
    pub clause: i32,
    /// Subordinate, complement, or relative clause boundary (`advcl`, `ccomp`,
    /// `acl:relcl`, `SBAR`).
    pub subordinate: i32,
    /// Phrase chunk, constituent, or comma boundary used when falling back
    /// inside an over-long clause.
    pub phrase: i32,
    /// Ordinary token boundary used when falling back inside an over-long
    /// clause or phrase.
    pub token: i32,
}

impl Default for ClauseCosts {
    fn default() -> Self {
        let u = turbo_analysis_layers::Boundaries::UNIT;
        Self { clause: u * 5 / 10, subordinate: u * 6 / 10, phrase: u * 8 / 10, token: u * 12 / 10 }
    }
}

/// What kind of sub-sentence break lies after token `k` of a sentence.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ClauseBreak {
    /// A syntactic clause boundary with its cut cost.
    Clause(i32),
    /// A phrase or punctuation boundary with its fallback cut cost.
    Phrase(i32),
    /// An ordinary token boundary with its fallback cut cost.
    Token(i32),
    /// Adjacent tokens overlap in bytes (`tok[k].end > tok[k + 1].start`), so
    /// no cut may fall between them.
    Held,
}

impl ClauseBreak {
    /// Returns the cost of cutting at this token gap when used as a fallback
    /// token boundary, or `None` if `Held`.
    pub fn fallback_cost(self) -> Option<i32> {
        match self {
            ClauseBreak::Clause(c) | ClauseBreak::Phrase(c) | ClauseBreak::Token(c) => Some(c),
            ClauseBreak::Held => None,
        }
    }

    pub(super) fn rank(self) -> u8 {
        match self {
            ClauseBreak::Held => 4,
            ClauseBreak::Clause(_) => 3,
            ClauseBreak::Phrase(_) => 2,
            ClauseBreak::Token(_) => 1,
        }
    }

    pub(super) fn upgrade(&mut self, next: ClauseBreak) {
        if *self == ClauseBreak::Held {
            return;
        }
        match (*self, next) {
            (ClauseBreak::Clause(a), ClauseBreak::Clause(b)) => *self = ClauseBreak::Clause(a.min(b)),
            (ClauseBreak::Phrase(a), ClauseBreak::Phrase(b)) => *self = ClauseBreak::Phrase(a.min(b)),
            (ClauseBreak::Token(a), ClauseBreak::Token(b)) => *self = ClauseBreak::Token(a.min(b)),
            _ if next.rank() > self.rank() => *self = next,
            _ => {}
        }
    }
}

/// Candidate clause and token cut points for one sentence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SentenceClauseCuts {
    /// Token spans in the sentence's view, clamped to the sentence span.
    pub tokens: Vec<Span>,
    /// Break after each token (`breaks.len() == tokens.len().saturating_sub(1)`).
    pub breaks: Vec<ClauseBreak>,
}

/// Sub-sentence clause cut candidates aligned to [`Sentences`] (`turbo:clause-cuts`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClauseCuts {
    pub sentences: Vec<SentenceClauseCuts>,
}

impl turbo_analysis::Layer for ClauseCuts {
    const ID: &'static str = "turbo:clause-cuts";

    fn anchor() -> Anchor {
        Anchor::aligned::<Sentences>()
    }

    fn len(&self) -> usize {
        self.sentences.len()
    }

    fn check(&self) -> Result<(), String> {
        for (i, s) in self.sentences.iter().enumerate() {
            if s.breaks.len() != s.tokens.len().saturating_sub(1) {
                return Err(format!("sentence {i} has {} breaks for {} tokens", s.breaks.len(), s.tokens.len()));
            }
            if !s.tokens.windows(2).all(|w| w[0].start <= w[1].start && w[0].end <= w[1].end) {
                return Err(format!("sentence {i} tokens are not in text order"));
            }
        }
        Ok(())
    }
}

/// Derives sub-sentence clause and fallback token cut candidates for every
/// sentence in `sentences`.
#[allow(clippy::too_many_arguments)]
pub fn clause_cuts(
    sentences: &Sentences,
    tokens: &Tokens,
    dependencies: Option<&Dependencies<Tokens>>,
    phrases: Option<&Phrases>,
    phrase_chunks: Option<&PhraseChunks>,
    pos: Option<&PosTags>,
    view: Option<&View>,
    costs: ClauseCosts,
) -> ClauseCuts {
    let n = sentences.spans.len();
    let (tok_starts, tok_ends) = tokens.spans.columns();
    let mut out = Vec::with_capacity(n);
    for (s_ix, sent_span) in sentences.spans.iter() {
        let tok_range = if tokens.sentence.is_some() {
            let r = tokens.of(s_ix);
            r.start.get()..r.end.get()
        } else {
            let a = tok_ends.partition_point(|&e| e <= sent_span.start);
            let b = tok_starts.partition_point(|&x| x < sent_span.end);
            a..b.max(a)
        };
        out.push(build_sentence_cuts(
            sent_span,
            tok_range,
            tokens,
            dependencies,
            phrases,
            phrase_chunks,
            pos,
            view,
            costs,
        ));
    }
    ClauseCuts { sentences: out }
}

/// Annotator that writes [`ClauseCuts`] (`turbo:clause-cuts`) from syntax and
/// token layers so [`Chunker`](crate::Chunker) and
/// [`CohesiveChunker`](crate::CohesiveChunker) split over-budget sentences at
/// syntactic [`Cut::Clause`] boundaries before falling back to token windows.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ClauseBoundaries {
    pub costs: ClauseCosts,
}

impl ClauseBoundaries {
    pub const fn new(costs: ClauseCosts) -> Self {
        Self { costs }
    }

    pub fn bound(self) -> Bound<Self> {
        Bound::new(self)
    }
}

impl Annotator for ClauseBoundaries {
    const ID: &'static str = "turbo:clause-boundaries";
    type Reads = (
        Sentences,
        Tokens,
        Option<Dependencies<Tokens>>,
        Option<Phrases>,
        Option<PhraseChunks>,
        Option<PosTags>,
        Option<View>,
    );
    type Writes = (ClauseCuts,);

    fn annotate(
        &self,
        (sentences, tokens, deps, phrases, phrase_chunks, pos, view): Refs<'_, Self>,
    ) -> Result<(ClauseCuts,), Error> {
        let v = sentences.spans.view();
        if tokens.spans.view() != v
            || phrases.is_some_and(|p| p.spans.view() != v)
            || phrase_chunks.is_some_and(|p| p.spans.view() != v)
            || view.is_some_and(|vw| vw.id() != v)
        {
            return Err(Error::Annotator { annotator: Self::ID, detail: "its inputs are in different views".into() });
        }
        let n_tok = tokens.spans.len();
        if let Some(d) = deps
            && d.len() != n_tok
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("{} dependencies for {n_tok} tokens", d.len()),
            });
        }
        if let Some(p) = pos
            && p.len() != n_tok
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("{} POS tags for {n_tok} tokens", p.len()),
            });
        }
        Ok((clause_cuts(sentences, tokens, deps, phrases, phrase_chunks, pos, view, self.costs),))
    }

    fn config_hash(&self) -> u64 {
        let c = self.costs;
        [c.clause, c.subordinate, c.phrase, c.token]
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x as u32 as u64).wrapping_mul(0x0100_0000_01b3))
    }
}

/// Splits an over-budget sentence `sent_ix` into sub-sentence rows using
/// syntactic `ClauseBreak::Clause` boundaries first, falling back to token
/// windows for any single clause that still exceeds `budget.max`.
///
/// Returns `Ok(true)` if the sentence was split into 2 or more rows and pushed
/// to `chunks`, or `Ok(false)` if the sentence has at most one atomic token
/// unit and cannot be split.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_clause_rows(
    id: &'static str,
    cx: &RunCx<'_>,
    sent_ix: usize,
    opening: Cut,
    parent: Option<Ix<Chunks>>,
    budget: Budget,
    clauses: &ClauseCuts,
    chunks: &mut Chunks,
) -> Result<bool, Error> {
    let Some(sc) = clauses.sentences.get(sent_ix) else {
        return Ok(false);
    };
    if sc.tokens.len() <= 1 {
        return Ok(false);
    }
    let sent_span = cx.sentences.spans.get(Ix::new(sent_ix as u32));

    let mut unit_spans: Vec<Span> = Vec::with_capacity(sc.tokens.len());
    let mut unit_breaks: Vec<ClauseBreak> = Vec::with_capacity(sc.breaks.len());
    let mut cur_start = sent_span.start;
    for (k, &tok) in sc.tokens.iter().enumerate() {
        let end = if k + 1 == sc.tokens.len() { sent_span.end } else { tok.end.max(cur_start) };
        if k + 1 < sc.tokens.len() {
            let brk = sc.breaks[k];
            let next_start = sc.tokens[k + 1].start;
            if brk == ClauseBreak::Held || end > next_start || end <= cur_start {
                continue;
            }
            unit_spans.push(Span::new(cur_start, end));
            unit_breaks.push(brk);
            cur_start = next_start;
        } else if end > cur_start {
            unit_spans.push(Span::new(cur_start, end));
        }
    }
    let u_len = unit_spans.len();
    if u_len <= 1 {
        return Ok(false);
    }

    let (sw_starts, sw_ends) = cx.subwords.spans.columns();
    let tok_lengths = Lengths {
        start: unit_spans.iter().map(|s| sw_ends.partition_point(|&e| e <= s.start) as u32).collect(),
        end: unit_spans.iter().map(|s| sw_starts.partition_point(|&x| x < s.end) as u32).collect(),
    };

    let mut clause_ranges: Vec<Range<usize>> = Vec::new();
    let mut clause_costs: Vec<i32> = Vec::new();
    let mut c_start = 0usize;
    for (u_idx, &brk) in unit_breaks.iter().enumerate() {
        if let ClauseBreak::Clause(cost) = brk {
            clause_ranges.push(c_start..u_idx + 1);
            clause_costs.push(cost);
            c_start = u_idx + 1;
        }
    }
    clause_ranges.push(c_start..u_len);
    clause_costs.push(0);

    let clause_segs = segment(
        &clause_costs,
        |i, j| tok_lengths.run(clause_ranges[i].start, clause_ranges[j - 1].end),
        budget,
        no_term,
    );

    let mut final_pieces: Vec<(Range<usize>, u64, i128)> = Vec::new();
    for seg in clause_segs {
        let u_range = clause_ranges[seg.start].start..clause_ranges[seg.end - 1].end;
        if seg.tokens > u64::from(budget.max) && u_range.len() > 1 {
            let mut sub_costs: Vec<i32> = unit_breaks[u_range.start..u_range.end - 1]
                .iter()
                .map(|b| b.fallback_cost().unwrap_or(i32::MAX / 4))
                .collect();
            sub_costs.push(0);
            for t_seg in
                segment(&sub_costs, |i, j| tok_lengths.run(u_range.start + i, u_range.start + j), budget, no_term)
            {
                final_pieces.push((
                    (u_range.start + t_seg.start)..(u_range.start + t_seg.end),
                    t_seg.tokens,
                    t_seg.cost,
                ));
            }
        } else {
            final_pieces.push((u_range, seg.tokens, seg.cost));
        }
    }

    if final_pieces.len() <= 1 {
        return Ok(false);
    }

    for (idx, (u_range, tok_count, seg_cost)) in final_pieces.into_iter().enumerate() {
        let piece_span = Span::new(unit_spans[u_range.start].start, unit_spans[u_range.end - 1].end);
        let tokens = u32::try_from(tok_count).map_err(|_| Error::Annotator {
            annotator: id,
            detail: format!("a chunk of {tok_count} tokens, more than a u32 holds"),
        })?;
        let cut = if idx == 0 { opening } else { Cut::Clause };
        let row = Row { span: piece_span, level: 0, parent, cut, cost: saturate(seg_cost), tokens };
        row.push(chunks, sent_ix..sent_ix + 1, tok_count > u64::from(budget.max));
    }
    Ok(true)
}
