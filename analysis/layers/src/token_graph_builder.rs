//! Declarative builder for [`TokenGraph`] across multiple producers
//! (dehyphenation, compound decomposition, abbreviations, and multi-token
//! phrase synonyms) without shifting downstream phrase ordinals.

use std::ops::Range;

use turbo_analysis::{Span, ViewId};

use crate::token_graph::{TokenArcKind, TokenGraph};
use crate::{Stems, Tokens};

struct PartSpec {
    span: Span,
    term: String,
    kind: TokenArcKind,
}

struct SplitSpec {
    parts: Vec<PartSpec>,
    whole_kind: TokenArcKind,
}

struct SingleExpansion {
    terms: Vec<String>,
    kind: TokenArcKind,
}

struct SpanSequence {
    token_range: Range<usize>,
    span: Span,
    terms: Vec<String>,
    kind: TokenArcKind,
}

/// Accumulates sub-token splits, single-token multi-word expansions, and
/// multi-token spanning synonyms over a base [`Tokens`] sequence and emits a
/// topologically sorted, cycle-free [`TokenGraph`].
pub struct TokenGraphBuilder {
    view: ViewId,
    base_spans: Vec<Span>,
    base_terms: Vec<String>,
    splits: Vec<Option<SplitSpec>>,
    extra_unit_arcs: Vec<Vec<(Span, String, TokenArcKind)>>,
    single_expansions: Vec<Vec<SingleExpansion>>,
    span_sequences: Vec<SpanSequence>,
}

impl TokenGraphBuilder {
    /// Initializes a builder from `tokens` over `view_text`, using `stems` for
    /// primary token terms when provided or slicing `view_text` otherwise.
    pub fn from_tokens(view_text: &str, tokens: &Tokens, stems: Option<&Stems>) -> Result<Self, String> {
        let n = tokens.spans.len();
        if let Some(s) = stems
            && s.stem.len() != n
        {
            return Err(format!("stems length {} does not match tokens length {n}", s.stem.len()));
        }
        let (starts, ends) = tokens.spans.columns();
        let mut base_spans = Vec::with_capacity(n);
        let mut base_terms = Vec::with_capacity(n);
        for i in 0..n {
            let s = Span::new(starts[i], ends[i]);
            let slice = view_text
                .get(s.range())
                .ok_or_else(|| format!("token {i} span {s:?} is outside view text or cuts a UTF-8 character"))?;
            let term = match stems {
                Some(st) => st.stem.get(i),
                None => slice,
            };
            base_spans.push(s);
            base_terms.push(term.to_owned());
        }
        Ok(Self::from_base_tokens(tokens.spans.view(), base_spans, base_terms))
    }

    /// Initializes a builder from pre-extracted `(span, term)` base tokens.
    pub fn from_base_tokens(view: ViewId, base_spans: Vec<Span>, base_terms: Vec<String>) -> Self {
        assert_eq!(base_spans.len(), base_terms.len(), "base_spans and base_terms must match");
        let n = base_spans.len();
        TokenGraphBuilder {
            view,
            base_spans,
            base_terms,
            splits: (0..n).map(|_| None).collect(),
            extra_unit_arcs: (0..n).map(|_| Vec::new()).collect(),
            single_expansions: (0..n).map(|_| Vec::new()).collect(),
            span_sequences: Vec::new(),
        }
    }

    pub fn token_count(&self) -> usize {
        self.base_spans.len()
    }

    pub fn base_span(&self, token_idx: usize) -> Span {
        self.base_spans[token_idx]
    }

    pub fn base_term(&self, token_idx: usize) -> &str {
        &self.base_terms[token_idx]
    }

    /// Adds an extra single-token synonym/surface arc on `token_idx` spanning
    /// that token's full width in the final lattice.
    pub fn add_token_term(&mut self, token_idx: usize, span: Span, term: impl Into<String>, kind: TokenArcKind) {
        if token_idx < self.base_spans.len() {
            let term = term.into();
            if !term.is_empty() {
                self.extra_unit_arcs[token_idx].push((span, term, kind));
            }
        }
    }

    /// Splits `token_idx` into `parts` (`parts.len() >= 2`), emitting each part
    /// at unit sub-positions alongside the whole token spanning the full split.
    pub fn split_token(
        &mut self,
        token_idx: usize,
        parts: impl IntoIterator<Item = (Span, String, TokenArcKind)>,
        whole_kind: TokenArcKind,
    ) {
        if token_idx >= self.base_spans.len() {
            return;
        }
        let collected: Vec<PartSpec> = parts
            .into_iter()
            .filter(|(_, t, _)| !t.is_empty())
            .map(|(span, term, kind)| PartSpec { span, term, kind })
            .collect();
        if collected.len() < 2 {
            return;
        }
        if self.splits[token_idx].is_none() {
            self.splits[token_idx] = Some(SplitSpec { parts: collected, whole_kind });
        }
    }

    /// Expands a single token `token_idx` into a multi-word sequence (e.g.
    /// `"FDIC"` -> `["Federal", "Deposit", "Insurance", "Corporation"]`),
    /// preserving `base_span(token_idx)` on every expanded arc.
    pub fn expand_token_sequence<S: Into<String>>(
        &mut self,
        token_idx: usize,
        terms: impl IntoIterator<Item = S>,
        kind: TokenArcKind,
    ) {
        if token_idx >= self.base_spans.len() {
            return;
        }
        let words: Vec<String> = terms.into_iter().map(Into::into).filter(|w| !w.is_empty()).collect();
        if words.is_empty() {
            return;
        }
        if words.len() == 1 {
            let span = self.base_spans[token_idx];
            self.add_token_term(token_idx, span, words.into_iter().next().unwrap(), kind);
        } else {
            self.single_expansions[token_idx].push(SingleExpansion { terms: words, kind });
        }
    }

    /// Emits a single spanning arc (`pos_len` covering `token_range`) with
    /// `span` and `term` (e.g. `"data-base"` -> `"database"` or
    /// `"Federal Deposit Insurance Corporation"` -> `"FDIC"`).
    pub fn add_span_arc(&mut self, token_range: Range<usize>, span: Span, term: impl Into<String>, kind: TokenArcKind) {
        let term = term.into();
        if !term.is_empty() {
            self.add_span_sequence(token_range, span, [term], kind);
        }
    }

    /// Emits a sequence of `terms` across `token_range` (`start < end`).
    pub fn add_span_sequence<S: Into<String>>(
        &mut self,
        token_range: Range<usize>,
        span: Span,
        terms: impl IntoIterator<Item = S>,
        kind: TokenArcKind,
    ) {
        if token_range.is_empty() || token_range.end > self.base_spans.len() {
            return;
        }
        let words: Vec<String> = terms.into_iter().map(Into::into).filter(|w| !w.is_empty()).collect();
        if words.is_empty() {
            return;
        }
        if token_range.len() == 1 && span == self.base_spans[token_range.start] {
            self.expand_token_sequence(token_range.start, words, kind);
            return;
        }
        self.span_sequences.push(SpanSequence { token_range, span, terms: words, kind });
    }

    /// Resolves all position widths, deduplicates identical arcs, sorts in
    /// canonical topological order, and validates the resulting [`TokenGraph`].
    pub fn finish(self) -> Result<TokenGraph, String> {
        let n = self.base_spans.len();
        if n == 0 {
            return Ok(TokenGraph::new(self.view));
        }
        let mut width = vec![1u32; n];
        for (t, w) in width.iter_mut().enumerate() {
            if let Some(split) = &self.splits[t] {
                *w = (*w).max(split.parts.len() as u32);
            }
            for exp in &self.single_expansions[t] {
                *w = (*w).max(exp.terms.len() as u32);
            }
        }
        for seq in &self.span_sequences {
            let cur: u32 = width[seq.token_range.clone()].iter().sum();
            let need = seq.terms.len() as u32;
            if need > cur {
                width[seq.token_range.end - 1] += need - cur;
            }
        }

        let mut pos_offset = Vec::with_capacity(n + 1);
        pos_offset.push(0u32);
        for &w in &width {
            let next = pos_offset
                .last()
                .unwrap()
                .checked_add(w)
                .ok_or_else(|| String::from("token lattice position count overflows u32"))?;
            pos_offset.push(next);
        }
        let num_positions = *pos_offset.last().unwrap();

        let mut raw: Vec<(u32, u32, Span, TokenArcKind, String)> = Vec::with_capacity(n * 2);
        for t in 0..n {
            let p = pos_offset[t];
            let w = width[t];
            let base_span = self.base_spans[t];
            let base_term = &self.base_terms[t];
            if let Some(split) = &self.splits[t] {
                raw.push((p, w, base_span, split.whole_kind, base_term.clone()));
                let k = split.parts.len() as u32;
                for (j, part) in split.parts.iter().enumerate() {
                    let j_u32 = j as u32;
                    let len = if j_u32 + 1 == k { w - j_u32 } else { 1 };
                    raw.push((p + j_u32, len, part.span, part.kind, part.term.clone()));
                }
            } else {
                raw.push((p, w, base_span, TokenArcKind::Primary, base_term.clone()));
            }
            for (span, term, kind) in &self.extra_unit_arcs[t] {
                raw.push((p, w, *span, *kind, term.clone()));
            }
            for exp in &self.single_expansions[t] {
                let m = exp.terms.len() as u32;
                for (j, word) in exp.terms.iter().enumerate() {
                    let j_u32 = j as u32;
                    let len = if j_u32 + 1 == m { w - j_u32 } else { 1 };
                    raw.push((p + j_u32, len, base_span, exp.kind, word.clone()));
                }
            }
        }
        for seq in &self.span_sequences {
            let p_start = pos_offset[seq.token_range.start];
            let p_end = pos_offset[seq.token_range.end];
            let w_span = p_end - p_start;
            let m = seq.terms.len() as u32;
            for (j, word) in seq.terms.iter().enumerate() {
                let j_u32 = j as u32;
                let len = if j_u32 + 1 == m { w_span - j_u32 } else { 1 };
                raw.push((p_start + j_u32, len, seq.span, seq.kind, word.clone()));
            }
        }

        raw.sort_by(|a, b| {
            (a.0, a.1, a.2.start, a.2.end, a.3, a.4.as_str()).cmp(&(b.0, b.1, b.2.start, b.2.end, b.3, b.4.as_str()))
        });
        let mut graph = TokenGraph::new(self.view);
        graph.num_positions = num_positions;
        for (pos, len, span, kind, term) in raw {
            if !graph.spans.is_empty() {
                let last_ix = turbo_analysis::Ix::new((graph.spans.len() - 1) as u32);
                let prev = graph.get(last_ix);
                if prev.position == pos && prev.position_length == len && prev.span == span && prev.term == term {
                    continue;
                }
            }
            graph.push(span, &term, pos, len, kind);
        }
        turbo_analysis::Layer::check(&graph)?;
        Ok(graph)
    }
}
