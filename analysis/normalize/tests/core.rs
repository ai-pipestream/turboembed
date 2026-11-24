//! Tests for the normalizer core: `normalize`, `apply`, `compose`, `Chain`
//! and `NormalizeStep`. The normalizers here are small and need no Unicode
//! data, so a failure points at the composition code and not at a table.

#[path = "core/chain.rs"]
mod chain;
#[path = "core/compose.rs"]
mod compose;
#[path = "core/documents.rs"]
mod documents;

use std::borrow::Cow;

use turbo_analysis::text::Side;
use turbo_analysis::{Alignment, Analyzer, Bound, Document, Key, Span, View, ViewBuilder};
use turbo_normalize::{Chain, DynNormalizer, NormalizeStep, Normalizer, apply, compose, hash_config, normalize};

/// Expands U+FB01 (3 bytes) to "fi" (2 bytes).
struct Ligature;

impl Normalizer for Ligature {
    const ID: &'static str = "test:ligature";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        for (i, c) in text.char_indices() {
            if c == '\u{FB01}' {
                out.keep(i);
                out.replace(i + c.len_utf8(), "fi");
            }
        }
    }
}

/// Folds a run of two or more spaces to one space.
struct Spaces;

impl Normalizer for Spaces {
    const ID: &'static str = "test:spaces";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] != b' ' {
                i += 1;
                continue;
            }
            let start = i;
            while i < bytes.len() && bytes[i] == b' ' {
                i += 1;
            }
            if i - start > 1 {
                out.keep(start);
                out.replace(i, " ");
            }
        }
    }
}

/// Deletes U+00AD (2 bytes).
struct SoftHyphen;

impl Normalizer for SoftHyphen {
    const ID: &'static str = "test:soft-hyphen";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        for (i, c) in text.char_indices() {
            if c == '\u{AD}' {
                out.keep(i);
                out.replace(i + c.len_utf8(), "");
            }
        }
    }
}

/// Inserts a marker after every period. The marker is an option, so two
/// configurations differ in `config_hash`.
struct AfterPeriod {
    marker: &'static str,
}

impl Default for AfterPeriod {
    fn default() -> AfterPeriod {
        AfterPeriod { marker: "|" }
    }
}

impl Normalizer for AfterPeriod {
    const ID: &'static str = "test:after-period";
    const VERSION: &'static str = "3";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        for (i, c) in text.char_indices() {
            if c == '.' {
                out.keep(i + 1);
                out.replace(i + 1, self.marker);
            }
        }
    }

    fn config_hash(&self) -> u64 {
        hash_config(&self.marker)
    }
}

/// Uppercases ASCII letters in place, so the edit has the same length on
/// both sides.
struct Upper;

impl Normalizer for Upper {
    const ID: &'static str = "test:upper";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        for (i, b) in text.bytes().enumerate() {
            if b.is_ascii_lowercase() {
                out.keep(i);
                out.replace(i + 1, &text[i..i + 1].to_ascii_uppercase());
            }
        }
    }
}

/// Replaces every "ab" with "x", so the text shrinks.
struct AbToX;

impl Normalizer for AbToX {
    const ID: &'static str = "test:ab-to-x";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        for (i, _) in text.match_indices("ab") {
            out.keep(i);
            out.replace(i + 2, "x");
        }
    }
}

/// Changes nothing.
struct Identity;

impl Normalizer for Identity {
    const ID: &'static str = "test:identity";

    fn normalize(&self, _text: &str, _out: &mut ViewBuilder<'_>) {}
}

/// Deletes the whole text, whatever it is.
struct DeleteAll;

impl Normalizer for DeleteAll {
    const ID: &'static str = "test:delete-all";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        out.replace(text.len(), "");
    }
}

/// Replaces the whole text with one fixed string.
struct ReplaceAll;

impl Normalizer for ReplaceAll {
    const ID: &'static str = "test:replace-all";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        out.replace(text.len(), "<all>");
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Kind {
    Ligature,
    Spaces,
    SoftHyphen,
    AfterPeriod,
    Upper,
    AbToX,
    Identity,
    DeleteAll,
    ReplaceAll,
}

const KINDS: [Kind; 7] =
    [Kind::Ligature, Kind::Spaces, Kind::SoftHyphen, Kind::AfterPeriod, Kind::Upper, Kind::AbToX, Kind::Identity];

fn boxed(kind: Kind) -> Box<dyn DynNormalizer> {
    match kind {
        Kind::Ligature => Box::new(Ligature),
        Kind::Spaces => Box::new(Spaces),
        Kind::SoftHyphen => Box::new(SoftHyphen),
        Kind::AfterPeriod => Box::new(AfterPeriod::default()),
        Kind::Upper => Box::new(Upper),
        Kind::AbToX => Box::new(AbToX),
        Kind::Identity => Box::new(Identity),
        Kind::DeleteAll => Box::new(DeleteAll),
        Kind::ReplaceAll => Box::new(ReplaceAll),
    }
}

fn chain_of(kinds: &[Kind]) -> Chain {
    kinds.iter().fold(Chain::new(), |chain, &k| chain.then_dyn(boxed(k)))
}

/// Inputs that mix the triggers of every toy normalizer: triggers next to
/// each other, at the start and at the end, a space run holding a soft
/// hyphen, an insertion point at the edge of a replaced stretch, and
/// multi-byte characters (2, 3 and 4 bytes) beside every kind of edit.
const INPUTS: &[&str] = &[
    "",
    "plain",
    "abc",
    "ab",
    "abab",
    "a b",
    "a  b",
    "   ",
    " ",
    ".",
    "..",
    "a.b",
    "ab.ab",
    "\u{FB01}",
    "\u{FB01}\u{FB01}",
    "\u{FB01}nal",
    "of\u{FB01}ce  report",
    "\u{AD}",
    "\u{AD}\u{AD}",
    "co\u{AD}op",
    "a  \u{AD}  b",
    "a \u{AD} b",
    " \u{AD} ",
    "  \u{AD}",
    "\u{AD}  ",
    "a.\u{AD}.b",
    "a\u{AD}b",
    "a\u{AD}b.",
    "a.  b",
    "a  .b",
    "ab  ab",
    "a\u{AD}b ab",
    "ab\u{AD}ab",
    "a\u{AD}\u{AD}b",
    ".ab.",
    "x.ab.y",
    "\u{FB01}.\u{FB01}",
    "\u{FB01}  \u{FB01}",
    "\u{FB01}\u{AD}\u{FB01}",
    "\u{E9}ab\u{20AC}",
    "\u{E9}.\u{20AC}",
    "\u{E9}  \u{20AC}",
    "\u{E9}\u{AD}\u{20AC}",
    "\u{1F600}ab\u{1F600}",
    "\u{1F600}.\u{1F600}",
    "\u{1F600}  \u{1F600}\u{AD}",
    "\u{1F600}\u{FB01}\u{1F600}",
    "\u{20AC}\u{AD}ab.\u{FB01} x  y",
    "The \u{FB01}nal  re\u{AD}port ab.  Done.",
    "abc abc.  cab\u{AD}ab",
    "..  ..",
    "a. b. c.",
    "  a  ",
];

/// Every span `[i, j)` of `text` whose ends are on character boundaries.
fn spans_of(text: &str) -> Vec<Span> {
    let bounds: Vec<u32> =
        text.char_indices().map(|(i, _)| i as u32).chain(std::iter::once(text.len() as u32)).collect();
    let mut spans = Vec::new();
    for (n, &i) in bounds.iter().enumerate() {
        for &j in &bounds[n..] {
            spans.push(Span::new(i, j));
        }
    }
    spans
}

/// Runs the steps one after another, keeping every intermediate text and
/// alignment. `texts[0]` is the input and `texts[k + 1]` is the output of
/// step `k`; `alignments[k]` maps `texts[k + 1]` to `texts[k]`.
struct Sequence {
    texts: Vec<String>,
    alignments: Vec<Alignment>,
}

impl Sequence {
    fn run(kinds: &[Kind], text: &str) -> Sequence {
        let mut texts = vec![text.to_string()];
        let mut alignments = Vec::new();
        for &k in kinds {
            let step = boxed(k);
            let current = texts.last().unwrap().clone();
            let mut b = ViewBuilder::new(&current);
            step.normalize_dyn(&current, &mut b);
            let (next, alignment) = b.finish();
            texts.push(next);
            alignments.push(alignment);
        }
        Sequence { texts, alignments }
    }

    fn last(&self) -> &str {
        self.texts.last().unwrap()
    }

    /// A span of the last text, mapped back to the input through every
    /// alignment in turn.
    fn to_input(&self, s: Span) -> Span {
        walk_back(self.alignments.iter(), s).0
    }

    /// A span of the input, mapped forward to the last text.
    fn to_output(&self, s: Span) -> Span {
        walk_forward(self.alignments.iter(), s).0
    }
}

/// Maps `s` from the last text to the first through `alignments` (given
/// first stage first). An empty span in the middle of the walk is mapped
/// to the span between the two ends of what the alignment has at that
/// offset, because `Alignment::span_to_parent` cannot map it. The flag is
/// false when that happened, which is also when `Document::to_source`
/// cannot be called on the same span.
fn walk_back<'a>(alignments: impl DoubleEndedIterator<Item = &'a Alignment>, s: Span) -> (Span, bool) {
    alignments.rev().fold((s, true), |(s, plain), a| (edit_span_to_parent(a, s), plain && !s.is_empty()))
}

/// Maps `s` from the first text to the last, like `walk_back`.
fn walk_forward<'a>(alignments: impl Iterator<Item = &'a Alignment>, s: Span) -> (Span, bool) {
    alignments.fold((s, true), |(s, plain), a| {
        let mapped = if s.is_empty() {
            let (x, y) = (a.to_child(s.start, Side::Start), a.to_child(s.start, Side::End));
            Span::new(x.min(y), x.max(y))
        } else {
            a.span_to_child(s)
        };
        (mapped, plain && !s.is_empty())
    })
}

/// Every non-empty span of `text` on character boundaries. Empty spans are
/// left out of the exhaustive checks: `Alignment` cannot map an empty span
/// that sits at an insertion or a deletion (see the ignored tests below).
fn non_empty_spans_of(text: &str) -> Vec<Span> {
    spans_of(text).into_iter().filter(|s| !s.is_empty()).collect()
}

/// True when `outer` covers every byte `inner` covers.
fn covers(outer: Span, inner: Span) -> bool {
    outer.start <= inner.start && inner.end <= outer.end
}

/// True when `s` is at least one byte away from every span in `edits`.
fn clear_of(edits: impl Iterator<Item = Span>, s: Span) -> bool {
    edits.into_iter().all(|e| s.end < e.start || e.end < s.start)
}

/// What a composed alignment must satisfy against the same steps run in
/// sequence. `Chain` merges stretches that overlap into one edit, so a
/// span inside an edit can map to a wider source span than it does step
/// by step. Therefore:
///
/// - the composed span covers the one found step by step, in both
///   directions, so nothing a step-by-step mapping covers is lost;
/// - the two are equal for every span at least one byte away from every
///   edit, where the composed alignment is plain offset shifting.
fn assert_mapping_agrees(
    case: &str,
    text: &str,
    out: &str,
    alignment: &Alignment,
    to_input: impl Fn(Span) -> Span,
    to_output: impl Fn(Span) -> Span,
) {
    for s in non_empty_spans_of(out) {
        let (composed, stepwise) = (alignment.span_to_parent(s), to_input(s));
        assert!(covers(composed, stepwise), "span_to_parent {s:?} of {case}: {composed:?} does not cover {stepwise:?}");
        if clear_of(alignment.edits().iter().map(|e| e.child), s) {
            assert_eq!(composed, stepwise, "span_to_parent {s:?} of {case}");
        }
    }
    for s in non_empty_spans_of(text) {
        let (composed, stepwise) = (alignment.span_to_child(s), to_output(s));
        assert!(covers(composed, stepwise), "span_to_child {s:?} of {case}: {composed:?} does not cover {stepwise:?}");
        if clear_of(alignment.edits().iter().map(|e| e.parent), s) {
            assert_eq!(composed, stepwise, "span_to_child {s:?} of {case}");
        }
    }
}

/// Maps the span of an edit to the parent. `Alignment::span_to_parent`
/// cannot map an empty span (an insertion) that sits on another edit, so
/// that case takes the span between the two ends of what the alignment
/// has at that offset.
fn edit_span_to_parent(a: &Alignment, s: Span) -> Span {
    if !s.is_empty() {
        return a.span_to_parent(s);
    }
    let (x, y) = (a.to_parent(s.start, Side::Start), a.to_parent(s.start, Side::End));
    Span::new(x.min(y), x.max(y))
}

/// Every edit any step made, mapped back to the input, lies inside one
/// edit of the composed alignment: no change a step made is left out.
fn assert_edits_are_kept(case: &str, seq: &Sequence, alignment: &Alignment) {
    for (k, stage) in seq.alignments.iter().enumerate() {
        for e in stage.edits() {
            let back = seq.alignments[..k].iter().rev().fold(e.parent, |s, a| edit_span_to_parent(a, s));
            assert!(
                alignment.edits().iter().any(|c| covers(c.parent, back)),
                "edit {e:?} of step {k} (input span {back:?}) is not inside an edit of the chain for {case}"
            );
        }
    }
}

/// The edits are in text order on both sides, do not overlap,
/// the text between them is the same on both sides, and every offset is
/// on a character boundary.
fn assert_well_formed(case: &str, text: &str, out: &str, alignment: &Alignment) {
    let (mut p, mut c) = (0usize, 0usize);
    for (n, e) in alignment.edits().iter().enumerate() {
        let (ps, pe, cs, ce) =
            (e.parent.start as usize, e.parent.end as usize, e.child.start as usize, e.child.end as usize);
        assert!(ps >= p && cs >= c, "edit {n} out of order for {case}");
        assert_eq!(text[p..ps], out[c..cs], "unedited text before edit {n} for {case}");
        assert!(text.is_char_boundary(ps) && text.is_char_boundary(pe), "edit {n} cuts the input for {case}");
        assert!(out.is_char_boundary(cs) && out.is_char_boundary(ce), "edit {n} cuts the output for {case}");
        (p, c) = (pe, ce);
    }
    assert_eq!(text[p..], out[c..], "unedited text after the last edit for {case}");
}

/// Checks the chain of `kinds` on `text` against the steps run in
/// sequence.
fn check_chain(kinds: &[Kind], text: &str) {
    let seq = Sequence::run(kinds, text);
    let (out, alignment) = normalize(&chain_of(kinds), text);
    let case = format!("{kinds:?} on {text:?}");
    assert_eq!(out, seq.last(), "text of {case}");
    assert_well_formed(&case, text, &out, &alignment);
    assert_edits_are_kept(&case, &seq, &alignment);
    assert_mapping_agrees(&case, text, &out, &alignment, |s| seq.to_input(s), |s| seq.to_output(s));
}
