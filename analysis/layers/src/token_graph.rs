//! Position-length lexical token lattice (`turbo:token-graph`, issue #2340).
//!
//! While [`Tokens`] holds a flat 1-best token sequence and
//! [`Alternatives<Tokens>`](turbo_analysis::Alternatives) enforces strict
//! non-overlapping byte partitions for morphological lattices, [`TokenGraph`]
//! represents a directed acyclic lexical graph over integer positions
//! `0..=num_positions` where every arc carries `(position, position_length,
//! term, view_span, kind)`.
//!
//! This allows indexers and query analyzers to emit:
//! - dehyphenated or punctuation-joined compounds (`"database"`, `pos_len = 2`)
//!   alongside their pre-join sub-parts (`"data"`, `"base"`, `pos_len = 1`),
//! - Hunspell and MeCab compound decompositions alongside whole compounds,
//! - intra-document abbreviation expansions (`"FDIC"` <->
//!   `"Federal Deposit Insurance Corporation"`) where expanded multi-word arcs
//!   over a short-form mention retain the mention's exact view span, and
//! - multi-token phrase synonyms from `turbo-matcher` and `turbo-glossary`,
//!
//! while preserving exact `Document::to_source` byte/UTF-16/code-point
//! projection and `slop = 0` phrase adjacency without shifting downstream
//! ordinals.

use turbo_analysis::{
    Alternatives, Anchor, Document, Error, Fnv1a, Ix, IxRange, Layer, Overlap, ProjectableLayer, Span, Spans,
    TextSpans, ViewId,
};

use crate::{Strs, Tokens};

/// Origin classification of an arc in a [`TokenGraph`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum TokenArcKind {
    /// Base 1-best token (or its primary folded/stemmed form).
    #[default]
    Primary = 0,
    /// Sub-part or joined compound from dehyphenation or connector joining.
    DehyphenatedPart = 1,
    /// Decomposed part of a morphological compound (e.g. Hunspell).
    CompoundPart = 2,
    /// Whole-word compound arc spanning decomposed parts.
    CompoundWhole = 3,
    /// Alternative path from a dictionary word lattice (e.g. MeCab).
    LatticeAlternative = 4,
    /// Short-form abbreviation spanning a multi-token long-form definition.
    AbbreviationShort = 5,
    /// Expanded long-form token sequence over an abbreviation mention.
    AbbreviationLong = 6,
    /// Single-token or multi-token phrase synonym expansion.
    Synonym = 7,
}

impl TokenArcKind {
    #[inline]
    pub const fn to_byte(self) -> u8 {
        self as u8
    }

    #[inline]
    pub const fn from_byte(b: u8) -> Option<TokenArcKind> {
        match b {
            0 => Some(TokenArcKind::Primary),
            1 => Some(TokenArcKind::DehyphenatedPart),
            2 => Some(TokenArcKind::CompoundPart),
            3 => Some(TokenArcKind::CompoundWhole),
            4 => Some(TokenArcKind::LatticeAlternative),
            5 => Some(TokenArcKind::AbbreviationShort),
            6 => Some(TokenArcKind::AbbreviationLong),
            7 => Some(TokenArcKind::Synonym),
            _ => None,
        }
    }
}

/// Borrowed view of one lexical arc in a [`TokenGraph`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TokenArc<'a> {
    pub span: Span,
    pub term: &'a str,
    pub position: u32,
    pub position_length: u32,
    pub kind: TokenArcKind,
}

/// Directed acyclic position-length token lattice (`turbo:token-graph`).
#[derive(Clone, Debug)]
pub struct TokenGraph {
    pub spans: Spans<TokenGraph>,
    pub terms: Strs,
    pub position: Vec<u32>,
    pub position_length: Vec<u32>,
    pub num_positions: u32,
    pub kind: Vec<TokenArcKind>,
}

impl TokenGraph {
    /// Creates an empty [`TokenGraph`] anchored to `view`.
    pub fn new(view: ViewId) -> TokenGraph {
        TokenGraph {
            spans: Spans::new(view),
            terms: Strs::default(),
            position: Vec::new(),
            position_length: Vec::new(),
            num_positions: 0,
            kind: Vec::new(),
        }
    }

    /// Appends one arc in topological order, updating `num_positions` to at
    /// least `position + position_length`.
    pub fn push(
        &mut self,
        span: Span,
        term: &str,
        position: u32,
        position_length: u32,
        kind: TokenArcKind,
    ) -> Ix<Self> {
        let ix = self.spans.push(span);
        self.terms.push(term);
        self.position.push(position);
        self.position_length.push(position_length);
        self.kind.push(kind);
        let end = position.saturating_add(position_length);
        if end > self.num_positions {
            self.num_positions = end;
        }
        ix
    }

    /// Reads arc `ix`.
    pub fn get(&self, ix: Ix<Self>) -> TokenArc<'_> {
        let i = ix.get();
        TokenArc {
            span: self.spans.get(ix),
            term: self.terms.get(i),
            position: self.position[i],
            position_length: self.position_length[i],
            kind: self.kind[i],
        }
    }

    /// Iterates over all `(Ix<TokenGraph>, TokenArc)` pairs in topological order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Ix<Self>, TokenArc<'_>)> + '_ {
        (0..self.spans.len() as u32).map(|i| {
            let ix = Ix::new(i);
            (ix, self.get(ix))
        })
    }

    /// Returns the contiguous index range of arcs leaving `pos`.
    pub fn at_position(&self, pos: u32) -> IxRange<Self> {
        let a = self.position.partition_point(|&p| p < pos);
        let b = a + self.position[a..].partition_point(|&p| p == pos);
        IxRange::new(a as u32, b as u32)
    }

    /// Builds a [`TokenGraph`] directly from an [`Alternatives<Tokens>`]
    /// lattice over `view_text`, using each alternative's explicit `form` when
    /// present or its slice of `view_text`.
    pub fn from_alternatives(view_text: &str, alt: &Alternatives<Tokens>) -> Result<TokenGraph, String> {
        alt.check()?;
        if alt.spans.is_empty() {
            return Ok(TokenGraph::new(alt.spans.view()));
        }
        let mut rows: Vec<(u32, u32, Span, TokenArcKind, &str)> = Vec::with_capacity(alt.len());
        for i in 0..alt.len() as u32 {
            let a = alt.get(Ix::new(i));
            let term = match a.form {
                Some(f) => f,
                None => view_text
                    .get(a.span.range())
                    .ok_or_else(|| format!("alternative {i} span {:?} is outside view text", a.span))?,
            };
            let kind = if a.mode.contains(turbo_analysis::SplitMode::Best) {
                TokenArcKind::Primary
            } else {
                TokenArcKind::LatticeAlternative
            };
            rows.push((a.position, a.position_length, a.span, kind, term));
        }
        rows.sort_by(|a, b| (a.0, a.1, a.2.start, a.2.end, a.3, a.4).cmp(&(b.0, b.1, b.2.start, b.2.end, b.3, b.4)));
        let mut out = TokenGraph::new(alt.spans.view());
        out.num_positions = alt.num_positions;
        for (pos, len, span, kind, term) in rows {
            if !out.spans.is_empty() {
                let prev = out.get(Ix::new((out.len() - 1) as u32));
                if prev.position == pos && prev.position_length == len && prev.span == span && prev.term == term {
                    continue;
                }
            }
            out.push(span, term, pos, len, kind);
        }
        out.check()?;
        Ok(out)
    }

    /// Finds all exact (`slop = 0`) contiguous paths in the DAG matching
    /// `phrase_terms`, where consecutive arcs satisfy
    /// `arc[k].position + arc[k].position_length == arc[k + 1].position`.
    pub fn matches_phrase(&self, phrase_terms: &[&str]) -> Vec<Vec<Ix<Self>>> {
        let Some((&first, rest)) = phrase_terms.split_first() else {
            return Vec::new();
        };
        let mut matches = Vec::new();
        let mut cur_path = Vec::with_capacity(phrase_terms.len());
        for (ix, arc) in self.iter() {
            if arc.term == first {
                cur_path.clear();
                cur_path.push(ix);
                self.extend_phrase(arc.position + arc.position_length, rest, &mut cur_path, &mut matches);
            }
        }
        matches
    }

    fn extend_phrase(
        &self,
        at_pos: u32,
        remaining: &[&str],
        cur_path: &mut Vec<Ix<Self>>,
        out: &mut Vec<Vec<Ix<Self>>>,
    ) {
        let Some((&want, rest)) = remaining.split_first() else {
            out.push(cur_path.clone());
            return;
        };
        if at_pos >= self.num_positions {
            return;
        }
        for ix in self.at_position(at_pos).iter() {
            let arc = self.get(ix);
            if arc.term == want {
                cur_path.push(ix);
                self.extend_phrase(arc.position + arc.position_length, rest, cur_path, out);
                cur_path.pop();
            }
        }
    }

    fn payload_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        let (starts, ends) = self.spans.columns();
        h.u16(self.spans.view().0);
        h.u32(self.num_positions);
        h.usize(starts.len());
        for i in 0..starts.len() {
            h.u32(starts[i]);
            h.u32(ends[i]);
            h.u32(self.position[i]);
            h.u32(self.position_length[i]);
            h.u8(self.kind[i].to_byte());
            h.str(self.terms.get(i));
        }
        h.finish()
    }
}

impl Layer for TokenGraph {
    const ID: &'static str = turbo_analysis::ids::TOKEN_GRAPH;
    const OVERLAP: Overlap = Overlap::Any;

    fn anchor() -> Anchor {
        Anchor::Text
    }

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn content_hash(&self) -> Option<u64> {
        Some(self.payload_hash())
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.terms.len() != n || self.position.len() != n || self.position_length.len() != n || self.kind.len() != n
        {
            return Err("spans, terms, position, position_length and kind columns must have equal length".into());
        }
        if n == 0 {
            if self.num_positions != 0 {
                return Err("zero positions expected when token graph is empty".into());
            }
            return Ok(());
        }
        if self.num_positions == 0 {
            return Err("positive num_positions expected when token graph has arcs".into());
        }
        if self.num_positions as usize > n {
            return Err(format!("{} positions cannot be covered by {n} arcs", self.num_positions));
        }
        let (starts, ends) = self.spans.columns();
        for i in 0..n {
            if starts[i] > ends[i] {
                return Err(format!("arc {i} span {}..{} is inverted", starts[i], ends[i]));
            }
            if self.position_length[i] == 0 {
                return Err(format!("arc {i} has position_length 0 (cycles/self-loops forbidden)"));
            }
            let end_pos = self.position[i]
                .checked_add(self.position_length[i])
                .ok_or_else(|| format!("arc {i} position + position_length overflows u32"))?;
            if end_pos > self.num_positions {
                return Err(format!("arc {i} ends at position {end_pos} past num_positions {}", self.num_positions));
            }
            if i > 0 {
                let prev_key = (self.position[i - 1], self.position_length[i - 1], starts[i - 1], ends[i - 1]);
                let cur_key = (self.position[i], self.position_length[i], starts[i], ends[i]);
                if cur_key < prev_key {
                    return Err(format!(
                        "arc {i} {cur_key:?} is out of topological order after arc {} {prev_key:?}",
                        i - 1
                    ));
                }
            }
        }
        let p_len = self.num_positions as usize;
        let mut from_start = vec![false; p_len + 1];
        let mut has_outgoing = vec![false; p_len];
        from_start[0] = true;
        for i in 0..n {
            let u = self.position[i] as usize;
            let v = u + self.position_length[i] as usize;
            has_outgoing[u] = true;
            if from_start[u] {
                from_start[v] = true;
            }
        }
        if !from_start[p_len] {
            return Err(format!("no path from position 0 to num_positions {}", self.num_positions));
        }
        let mut to_end = vec![false; p_len + 1];
        to_end[p_len] = true;
        for i in (0..n).rev() {
            let u = self.position[i] as usize;
            let v = u + self.position_length[i] as usize;
            if to_end[v] {
                to_end[u] = true;
            }
        }
        for (p, &out) in has_outgoing.iter().enumerate() {
            if !out {
                return Err(format!("position {p} has no outgoing arc"));
            }
        }
        for i in 0..n {
            let u = self.position[i] as usize;
            let v = u + self.position_length[i] as usize;
            if !from_start[u] || !to_end[v] {
                return Err(format!("arc {i} ({u}..{v}) does not lie on any complete path 0..{p_len}"));
            }
        }
        self.spans.check(Self::OVERLAP)
    }
}

impl ProjectableLayer for TokenGraph {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let spans = doc.project_spans_monotonic(&self.spans, to)?;
        let out = TokenGraph {
            spans,
            terms: self.terms.clone(),
            position: self.position.clone(),
            position_length: self.position_length.clone(),
            num_positions: self.num_positions,
            kind: self.kind.clone(),
        };
        out.check().map_err(|detail| Error::Invalid { layer: turbo_analysis::Key::<Self>::DEFAULT.id(), detail })?;
        Ok(out)
    }
}
