//! Token-lattice emission for dehyphenation and inline connector joins
//! (issue #2340).
//!
//! Connects `turbo-normalize` edits and inline hyphenated compounds to
//! [`TokenGraphBuilder`]:
//! - Pure deletion edits recorded inside a token's byte span in a derived
//!   [`View`]'s [`Alignment`] (such as [`crate::Dehyphenate`] deleting `"-\n"`
//!   across a line break, even after composed `Chain` stages like NFC and case
//!   folding) split the joined token into its pre-join sub-spans
//!   (`pos_len = 1`) alongside the joined compound (`pos_len = k`).
//!   Projecting those sub-spans via `Document::to_source` maps each part back
//!   to its exact half in `ViewId::SOURCE`.
//! - Inline hyphenated compounds (`"data-base"`, `"state-of-the-art"`) emit
//!   both split parts and the hyphen-free joined compound whether the
//!   tokenizer split at the hyphen or kept the hyphenated token whole.

use turbo_analysis::{Alignment, Annotator, Error, Refs, Span, View};
use turbo_analysis_layers::{Stems, TokenArcKind, TokenGraph, TokenGraphBuilder, Tokens};
use turbo_unicode::java::is_letter_or_digit;

#[inline]
fn is_join_connector(c: char) -> bool {
    matches!(c, '-' | '\u{00AD}' | '\u{2010}' | '\u{2011}')
}

fn has_word_char(s: &str) -> bool {
    s.chars().any(is_letter_or_digit)
}

/// Returns the non-empty sub-spans of `token_span` cut at pure deletion edits
/// (`edit.child.is_empty() && !edit.parent.is_empty()`) strictly inside
/// `token_span` in `alignment`.
pub fn alignment_join_parts(view_text: &str, alignment: &Alignment, token_span: Span) -> Option<Vec<Span>> {
    if token_span.is_empty() {
        return None;
    }
    let edits = alignment.edits();
    let lo = edits.partition_point(|e| e.child.end <= token_span.start);
    let mut cuts = Vec::new();
    for e in &edits[lo..] {
        if e.child.start >= token_span.end {
            break;
        }
        if e.child.is_empty()
            && !e.parent.is_empty()
            && e.child.start > token_span.start
            && e.child.start < token_span.end
            && view_text.is_char_boundary(e.child.start as usize)
        {
            cuts.push(e.child.start);
        }
    }
    cuts.dedup();
    if cuts.is_empty() {
        return None;
    }
    let mut parts = Vec::with_capacity(cuts.len() + 1);
    let mut cur = token_span.start;
    for cut in cuts {
        let sub = Span::new(cur, cut);
        if !has_word_char(view_text.get(sub.range())?) {
            return None;
        }
        parts.push(sub);
        cur = cut;
    }
    let tail = Span::new(cur, token_span.end);
    if !has_word_char(view_text.get(tail.range())?) {
        return None;
    }
    parts.push(tail);
    (parts.len() >= 2).then_some(parts)
}

fn inline_split_parts(view_text: &str, token_span: Span) -> Option<(Vec<(Span, String)>, String)> {
    let slice = view_text.get(token_span.range())?;
    if !slice.chars().any(is_join_connector) {
        return None;
    }
    let mut parts = Vec::new();
    let mut joined = String::with_capacity(slice.len());
    let mut part_start = 0usize;
    for (idx, c) in slice.char_indices() {
        if is_join_connector(c) {
            let sub = &slice[part_start..idx];
            if !has_word_char(sub) {
                return None;
            }
            parts
                .push((Span::new(token_span.start + part_start as u32, token_span.start + idx as u32), sub.to_owned()));
            joined.push_str(sub);
            part_start = idx + c.len_utf8();
        }
    }
    let tail = &slice[part_start..];
    if !has_word_char(tail) {
        return None;
    }
    parts.push((Span::new(token_span.start + part_start as u32, token_span.end), tail.to_owned()));
    joined.push_str(tail);
    (parts.len() >= 2).then_some((parts, joined))
}

/// Populates `builder` with dehyphenation and inline-hyphen lattice arcs using
/// `alignment` (typically `view.alignment()` or
/// `doc.view_alignment(ViewId::SOURCE, view.id())`).
pub fn emit_normalize_lattice(
    view_text: &str,
    alignment: &Alignment,
    tokens: &Tokens,
    builder: &mut TokenGraphBuilder,
) {
    let n = tokens.spans.len().min(builder.token_count());
    let (starts, ends) = tokens.spans.columns();
    for i in 0..n {
        let span = Span::new(starts[i], ends[i]);
        if let Some(sub_spans) = alignment_join_parts(view_text, alignment, span) {
            let parts = sub_spans.into_iter().filter_map(|s| {
                let sub = view_text.get(s.range())?;
                Some((s, sub.to_owned(), TokenArcKind::DehyphenatedPart))
            });
            builder.split_token(i, parts, TokenArcKind::CompoundWhole);
        } else if let Some((sub_parts, joined)) = inline_split_parts(view_text, span) {
            let parts = sub_parts.into_iter().map(|(s, term)| (s, term, TokenArcKind::DehyphenatedPart));
            builder.split_token(i, parts, TokenArcKind::Primary);
            builder.add_token_term(i, span, joined, TokenArcKind::CompoundWhole);
        }
    }

    // Detect runs of adjacent word tokens separated exclusively by a single
    // inline hyphen/connector character (e.g. WordTokenizer over "data-base").
    let mut i = 0usize;
    while i + 1 < n {
        let mut run_end = i + 1;
        while run_end < n {
            let left_end = ends[run_end - 1] as usize;
            let right_start = starts[run_end] as usize;
            let Some(gap) = view_text.get(left_end..right_start) else {
                break;
            };
            let mut chars = gap.chars();
            let is_single_connector = matches!((chars.next(), chars.next()), (Some(c), None) if is_join_connector(c));
            let left_ok = view_text.get(starts[run_end - 1] as usize..left_end).is_some_and(has_word_char);
            let right_ok = view_text.get(right_start..ends[run_end] as usize).is_some_and(has_word_char);
            if is_single_connector && left_ok && right_ok {
                run_end += 1;
            } else {
                break;
            }
        }
        if run_end > i + 1 {
            let mut joined = String::new();
            for &idx in &(i..run_end).collect::<Vec<_>>() {
                joined.push_str(builder.base_term(idx));
            }
            let full_span = Span::new(starts[i], ends[run_end - 1]);
            builder.add_span_arc(i..run_end, full_span, joined, TokenArcKind::CompoundWhole);
            i = run_end;
        } else {
            i += 1;
        }
    }
}

/// Pipeline step emitting a [`TokenGraph`] enriched with dehyphenation and
/// inline-connector compound/part arcs over a [`View`] and its [`Tokens`].
#[derive(Copy, Clone, Debug, Default)]
pub struct NormalizeLatticeAnnotator;

impl Annotator for NormalizeLatticeAnnotator {
    const ID: &'static str = "turbo-normalize:token-lattice";
    type Reads = (View, Tokens, Option<Stems>);
    type Writes = (TokenGraph,);

    fn annotate(&self, (view, tokens, stems): Refs<'_, Self>) -> Result<(TokenGraph,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "tokens belong to another view".into() });
        }
        let mut builder = TokenGraphBuilder::from_tokens(view.text(), tokens, stems)
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        emit_normalize_lattice(view.text(), view.alignment(), tokens, &mut builder);
        let graph = builder.finish().map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok((graph,))
    }
}
