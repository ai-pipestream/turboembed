//! Tests for the normalizer core: `normalize`, `apply`, `compose`, `Chain`
//! and `NormalizeStep`. The normalizers here are small and need no Unicode
//! data, so a failure points at the composition code and not at a table.

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

// 1. normalize and apply

#[test]
fn normalize_returns_the_text_and_its_alignment() {
    let src = "of\u{FB01}ce  \u{AD}x";
    let (out, a) = normalize(&Ligature, src);
    assert_eq!(out, "office  \u{AD}x");
    assert_eq!(a.edits().len(), 1);
    let e = a.edits()[0];
    assert_eq!(&src[e.parent.range()], "\u{FB01}");
    assert_eq!(&out[e.child.range()], "fi");
}

#[test]
fn apply_borrows_when_nothing_changes() {
    for text in ["", "plain text", "a b c", "\u{E9}\u{20AC}\u{1F600}"] {
        for kind in KINDS {
            let borrowed = matches!(apply_dyn(kind, text), Cow::Borrowed(_));
            // Upper is the only toy with a trigger in these texts, and only
            // where they hold an ASCII lowercase letter.
            let has_lower = text.bytes().any(|b| b.is_ascii_lowercase());
            assert_eq!(borrowed, !(kind == Kind::Upper && has_lower), "{kind:?} on {text:?}");
        }
    }
    match apply(&Identity, "unchanged") {
        Cow::Borrowed(s) => assert_eq!(s, "unchanged"),
        Cow::Owned(_) => panic!("identity must borrow"),
    }
    match apply(&Ligature, "no ligature here \u{E9}") {
        Cow::Borrowed(s) => assert_eq!(s, "no ligature here \u{E9}"),
        Cow::Owned(_) => panic!("no edit must borrow"),
    }
    match apply(&Spaces, "single spaces only") {
        Cow::Borrowed(_) => {}
        Cow::Owned(_) => panic!("single spaces must borrow"),
    }
}

#[test]
fn apply_owns_when_something_changes_and_matches_normalize() {
    for &text in INPUTS {
        for kind in KINDS {
            let n = boxed(kind);
            let mut b = ViewBuilder::new(text);
            n.normalize_dyn(text, &mut b);
            let (expected, alignment) = b.finish();
            let applied = apply_dyn(kind, text);
            assert_eq!(applied, expected, "{kind:?} on {text:?}");
            assert_eq!(matches!(applied, Cow::Owned(_)), !alignment.edits().is_empty(), "{kind:?} on {text:?}");
        }
    }
}

fn apply_dyn(kind: Kind, text: &str) -> Cow<'_, str> {
    match kind {
        Kind::Ligature => apply(&Ligature, text),
        Kind::Spaces => apply(&Spaces, text),
        Kind::SoftHyphen => apply(&SoftHyphen, text),
        Kind::AfterPeriod => apply(&AfterPeriod::default(), text),
        Kind::Upper => apply(&Upper, text),
        Kind::AbToX => apply(&AbToX, text),
        Kind::Identity => apply(&Identity, text),
        Kind::DeleteAll => apply(&DeleteAll, text),
        Kind::ReplaceAll => apply(&ReplaceAll, text),
    }
}

// 2. compose, through Chain

#[test]
fn every_single_step_chain_equals_the_step() {
    for &text in INPUTS {
        for kind in KINDS {
            check_chain(&[kind], text);
        }
    }
}

#[test]
fn every_pair_of_steps_composes() {
    for &text in INPUTS {
        for a in KINDS {
            for b in KINDS {
                check_chain(&[a, b], text);
            }
        }
    }
}

#[test]
fn every_triple_of_steps_composes() {
    for &text in INPUTS {
        for a in KINDS {
            for b in KINDS {
                for c in KINDS {
                    check_chain(&[a, b, c], text);
                }
            }
        }
    }
}

#[test]
fn compose_function_joins_two_alignments() {
    // "a  \u{AD}  b": the first stage collapses both space runs, the
    // second deletes the soft hyphen between them.
    let src = "a  \u{AD}  b";
    let (mid, first) = normalize(&Spaces, src);
    assert_eq!(mid, "a \u{AD} b");
    let (child, second) = normalize(&SoftHyphen, &mid);
    assert_eq!(child, "a  b");
    let mut out = ViewBuilder::new(src);
    compose(src, &first, &child, &second, &mut out);
    let (text, alignment) = out.finish();
    assert_eq!(text, "a  b");
    assert_eq!(text, normalize(&chain_of(&[Kind::Spaces, Kind::SoftHyphen]), src).0);
    // The three edits touch but do not overlap, so each stays its own
    // edit: the first space run became one space, the soft hyphen went
    // away, the second space run became one space.
    let edits: Vec<(Span, Span)> = alignment.edits().iter().map(|e| (e.parent, e.child)).collect();
    assert_eq!(
        edits,
        [(Span::new(1, 3), Span::new(1, 2)), (Span::new(3, 5), Span::new(2, 2)), (Span::new(5, 7), Span::new(2, 3)),]
    );
    for s in non_empty_spans_of(&text) {
        assert!(covers(alignment.span_to_parent(s), first.span_to_parent(second.span_to_parent(s))), "{s:?}");
    }
    for s in non_empty_spans_of(src) {
        assert!(covers(alignment.span_to_child(s), second.span_to_child(first.span_to_child(s))), "{s:?}");
    }
}

#[test]
fn compose_with_an_identity_stage_keeps_the_other_alignment() {
    let src = "o\u{FB01}ce ab";
    let (mid, first) = normalize(&Ligature, src);
    let (child, second) = normalize(&Identity, &mid);
    let mut out = ViewBuilder::new(src);
    compose(src, &first, &child, &second, &mut out);
    let (text, alignment) = out.finish();
    assert_eq!(text, mid);
    assert_eq!(alignment, first);

    let (mid, first) = normalize(&Identity, src);
    let (child, second) = normalize(&Ligature, &mid);
    let mut out = ViewBuilder::new(src);
    compose(src, &first, &child, &second, &mut out);
    let (text, alignment) = out.finish();
    assert_eq!(text, child);
    assert_eq!(alignment, second);
}

#[test]
fn an_insertion_beside_a_replaced_stretch_merges_into_one_edit() {
    // The period gets a marker after it, and the next stage replaces "ab"
    // right behind the marker: the two edits touch in the source.
    let src = "x.ab";
    let kinds = [Kind::AfterPeriod, Kind::AbToX];
    let (out, a) = normalize(&chain_of(&kinds), src);
    assert_eq!(out, "x.|x");
    assert_eq!(a.edits().len(), 1);
    assert_eq!(&src[a.edits()[0].parent.range()], "ab");
    check_chain(&kinds, src);
}

#[test]
fn an_insertion_on_the_edge_of_an_edit_is_kept() {
    // Each case: the steps, the input, the expected text and the expected
    // edits as ((parent start, end), (child start, end)).
    type Case = (&'static [Kind], &'static str, &'static str, &'static [((u32, u32), (u32, u32))]);
    let cases: [Case; 6] = [
        // The marker goes after the period and "ab" is replaced right behind
        // it: the marker is at the start of the replaced stretch, so both
        // are one edit.
        (&[Kind::AfterPeriod, Kind::AbToX], "x.ab", "x.|x", &[((2, 4), (2, 4))]),
        // The marker at the very end of the text.
        (&[Kind::AfterPeriod, Kind::Identity], "a.", "a.|", &[((2, 2), (2, 3))]),
        // The soft hyphen is deleted, then the marker goes in where the
        // period ends: the insertion touches the deletion.
        (&[Kind::SoftHyphen, Kind::AfterPeriod], ".\u{AD}", ".|", &[((1, 3), (1, 2))]),
        // A replaced stretch, then a marker after a period that follows it.
        (&[Kind::AbToX, Kind::AfterPeriod], "ab.", "x.|", &[((0, 2), (0, 1)), ((3, 3), (2, 3))]),
        // The marker touches the ligature that is expanded after it.
        (&[Kind::AfterPeriod, Kind::Ligature], ".\u{FB01}", ".|fi", &[((1, 4), (1, 4))]),
        // A deletion at the start of the text and a marker further on.
        (&[Kind::SoftHyphen, Kind::AfterPeriod], "\u{AD}.", ".|", &[((0, 2), (0, 0)), ((3, 3), (1, 2))]),
    ];
    for (kinds, src, expected, edits) in cases {
        let (out, a) = normalize(&chain_of(kinds), src);
        assert_eq!(out, expected, "{kinds:?} on {src:?}");
        let found: Vec<((u32, u32), (u32, u32))> =
            a.edits().iter().map(|e| ((e.parent.start, e.parent.end), (e.child.start, e.child.end))).collect();
        assert_eq!(found, edits, "{kinds:?} on {src:?}");
        check_chain(kinds, src);
    }
}

#[test]
fn chains_with_a_step_that_replaces_everything_compose() {
    let kinds_list: [&[Kind]; 6] = [
        &[Kind::DeleteAll],
        &[Kind::ReplaceAll],
        &[Kind::Ligature, Kind::DeleteAll],
        &[Kind::DeleteAll, Kind::AfterPeriod],
        &[Kind::ReplaceAll, Kind::Upper, Kind::AbToX],
        &[Kind::Spaces, Kind::ReplaceAll, Kind::SoftHyphen],
    ];
    for &text in INPUTS {
        for kinds in kinds_list {
            check_chain(kinds, text);
        }
    }
}

// Through a Document

/// Adds one `NormalizeStep` for `kind`, reading the view `from` and
/// writing the view `to` ("" is the source).
fn add_step(
    builder: turbo_analysis::analyzer::Builder,
    kind: Kind,
    from: &'static str,
    to: &'static str,
) -> turbo_analysis::analyzer::Builder {
    macro_rules! step {
        ($n:expr) => {
            builder.step(Bound::new(NormalizeStep::new($n)).read(0, from).write(0, to))
        };
    }
    match kind {
        Kind::Ligature => step!(Ligature),
        Kind::Spaces => step!(Spaces),
        Kind::SoftHyphen => step!(SoftHyphen),
        Kind::AfterPeriod => step!(AfterPeriod::default()),
        Kind::Upper => step!(Upper),
        Kind::AbToX => step!(AbToX),
        Kind::Identity => step!(Identity),
        Kind::DeleteAll => step!(DeleteAll),
        Kind::ReplaceAll => step!(ReplaceAll),
    }
}

const VIEW_NAMES: [&str; 4] = ["v1", "v2", "v3", "v4"];

/// Runs `kinds` as separate views, one step each, and returns the document
/// and the name of the last view.
fn run_separately(kinds: &[Kind], text: &str) -> (Document, &'static str) {
    let mut builder = Analyzer::builder();
    let mut from = "";
    for (k, &kind) in kinds.iter().enumerate() {
        builder = add_step(builder, kind, from, VIEW_NAMES[k]);
        from = VIEW_NAMES[k];
    }
    let mut docs = vec![Document::new(text)];
    let status = builder.build().unwrap().run(&mut docs);
    assert!(status[0].is_ok(), "{status:?}");
    (docs.pop().unwrap(), from)
}

/// Runs `kinds` as one `Chain` step writing the view "v1".
fn run_chained(kinds: &[Kind], text: &str) -> Document {
    let analyzer =
        Analyzer::builder().step(Bound::new(NormalizeStep::new(chain_of(kinds))).write(0, "v1")).build().unwrap();
    let mut docs = vec![Document::new(text)];
    let status = analyzer.run(&mut docs);
    assert!(status[0].is_ok(), "{status:?}");
    docs.pop().unwrap()
}

fn check_document(kinds: &[Kind], text: &str) {
    let case = format!("{kinds:?} on {text:?}");
    let (separate, last) = run_separately(kinds, text);
    let chained = run_chained(kinds, text);
    let sep_view = separate.get_key(&Key::<View>::named(last)).unwrap();
    let chain_view = chained.get_key(&Key::<View>::named("v1")).unwrap();
    assert_eq!(sep_view.text(), chain_view.text(), "text of {case}");
    let stages: Vec<&Alignment> = VIEW_NAMES[..kinds.len()]
        .iter()
        .map(|n| separate.get_key(&Key::<View>::named(n)).unwrap().alignment())
        .collect();
    // The chain's one alignment against the views' alignments in turn.
    assert_mapping_agrees(
        &case,
        text,
        chain_view.text(),
        chain_view.alignment(),
        |s| walk_back(stages.iter().copied(), s).0,
        |s| walk_forward(stages.iter().copied(), s).0,
    );
    // `Document` walks the same alignments; it agrees with the walk above
    // wherever no empty span turns up half way (it cannot map those).
    for s in non_empty_spans_of(sep_view.text()) {
        let (stepwise, plain) = walk_back(stages.iter().copied(), s);
        if plain {
            assert_eq!(separate.to_source(sep_view.id(), s), stepwise, "to_source {s:?} of {case}");
        }
        assert_eq!(chained.to_source(chain_view.id(), s), chain_view.alignment().span_to_parent(s), "{case}");
    }
    for s in non_empty_spans_of(text) {
        let (stepwise, plain) = walk_forward(stages.iter().copied(), s);
        if plain {
            assert_eq!(separate.from_source(sep_view.id(), s), stepwise, "from_source {s:?} of {case}");
        }
        assert_eq!(chained.from_source(chain_view.id(), s), chain_view.alignment().span_to_child(s), "{case}");
    }
}

#[test]
fn a_chain_step_and_separate_views_give_the_same_source_spans() {
    for &text in INPUTS {
        for a in KINDS {
            check_document(&[a], text);
            for b in KINDS {
                check_document(&[a, b], text);
            }
        }
    }
}

#[test]
fn three_separate_views_and_a_triple_chain_give_the_same_source_spans() {
    for &text in INPUTS {
        for a in KINDS {
            for b in KINDS {
                for c in KINDS {
                    check_document(&[a, b, c], text);
                }
            }
        }
    }
}

#[test]
fn the_view_a_chain_writes_has_the_source_as_parent() {
    let doc = run_chained(&[Kind::Ligature, Kind::Spaces], "o\u{FB01}ce  x");
    let v = doc.get_key(&Key::<View>::named("v1")).unwrap();
    assert_eq!(v.text(), "ofice x");
    assert_eq!(v.parent(), Some(doc.get_key(&Key::<View>::DEFAULT).unwrap().id()));
    // One edit list straight to the source, not one per step.
    assert_eq!(v.alignment().edits().len(), 2);
}

// 3. Property loop

/// xorshift64: deterministic, no dependencies.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Characters of 1 to 4 bytes, plus every trigger of the toys.
const ALPHABET: &[&str] =
    &["a", "b", "a", "b", "c", "x", " ", " ", ".", "\u{AD}", "\u{FB01}", "\u{E9}", "\u{20AC}", "\u{1F600}", "A"];

fn random_text(rng: &mut Rng) -> String {
    let n = rng.below(14);
    (0..n).map(|_| ALPHABET[rng.below(ALPHABET.len())]).collect()
}

fn random_kinds(rng: &mut Rng) -> Vec<Kind> {
    let n = 1 + rng.below(4);
    (0..n).map(|_| KINDS[rng.below(KINDS.len())]).collect()
}

#[test]
fn random_chains_on_random_text_match_the_steps_run_in_sequence() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..4000 {
        let text = random_text(&mut rng);
        let kinds = random_kinds(&mut rng);
        check_chain(&kinds, &text);
    }
}

#[test]
fn random_chains_match_separate_views_in_a_document() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for _ in 0..600 {
        let text = random_text(&mut rng);
        let kinds = random_kinds(&mut rng);
        check_document(&kinds, &text);
    }
}

#[test]
fn a_chain_alignment_is_well_formed() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    for _ in 0..2000 {
        let text = random_text(&mut rng);
        let kinds = random_kinds(&mut rng);
        let (out, a) = normalize(&chain_of(&kinds), &text);
        assert_well_formed(&format!("{kinds:?} on {text:?}"), &text, &out, &a);
    }
}

// 4. Chain bookkeeping

#[test]
fn chain_reports_its_steps() {
    let empty = Chain::new();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.ids().count(), 0);

    let chain = Chain::new().then(Ligature).then(Spaces).then_dyn(Box::new(Upper));
    assert!(!chain.is_empty());
    assert_eq!(chain.len(), 3);
    assert_eq!(chain.ids().collect::<Vec<_>>(), ["test:ligature", "test:spaces", "test:upper"]);
}

#[test]
fn an_empty_chain_changes_nothing() {
    let (out, a) = normalize(&Chain::new(), "a  b\u{AD}");
    assert_eq!(out, "a  b\u{AD}");
    assert!(a.edits().is_empty());
    assert!(matches!(apply(&Chain::new(), "x  y"), Cow::Borrowed("x  y")));
}

#[test]
fn chain_config_hash_changes_with_the_steps_and_their_order() {
    let ab = Chain::new().then(Ligature).then(Spaces);
    let ab_again = Chain::new().then(Ligature).then(Spaces);
    let ba = Chain::new().then(Spaces).then(Ligature);
    let abc = Chain::new().then(Ligature).then(Spaces).then(SoftHyphen);
    let a = Chain::new().then(Ligature);
    assert_eq!(ab.config_hash(), ab_again.config_hash());
    assert_ne!(ab.config_hash(), ba.config_hash());
    assert_ne!(ab.config_hash(), abc.config_hash());
    assert_ne!(ab.config_hash(), a.config_hash());
    assert_ne!(a.config_hash(), Chain::new().config_hash());
}

#[test]
fn chain_config_hash_follows_a_step_option() {
    let bar = Chain::new().then(AfterPeriod { marker: "|" });
    let star = Chain::new().then(AfterPeriod { marker: "*" });
    assert_ne!(bar.config_hash(), star.config_hash());
    assert_eq!(bar.config_hash(), Chain::new().then(AfterPeriod::default()).config_hash());
    // Nested chains hash by what they hold.
    let nested = Chain::new().then(Chain::new().then(Ligature));
    assert_ne!(nested.config_hash(), Chain::new().then(Ligature).config_hash());
}

#[test]
fn a_step_version_is_part_of_the_chain_hash() {
    // AfterPeriod has version "3" and the others "0": swapping a step for
    // another with the same hash and a different version still differs.
    let a = Chain::new().then(AfterPeriod::default());
    let b = Chain::new().then(Identity);
    assert_ne!(a.config_hash(), b.config_hash());
}

// 5. Provenance

#[test]
fn the_view_of_a_step_records_the_normalizer_id_version_and_config() {
    let normalizer = AfterPeriod { marker: "*" };
    let expected_config = normalizer.config_hash();
    let analyzer = Analyzer::builder().step(Bound::new(NormalizeStep::new(normalizer)).write(0, "v")).build().unwrap();
    let mut docs = vec![Document::new("one. two.")];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    let doc = &docs[0];
    assert_eq!(doc.get_key(&Key::<View>::named("v")).unwrap().text(), "one.* two.*");
    let p = doc.provenance(Key::<View>::named("v").id()).unwrap();
    assert_eq!(p.annotator, "test:after-period");
    assert_eq!(p.version, "3");
    assert_eq!(p.config, expected_config);
    assert_ne!(p.config, 0);
    assert_eq!(p.reads, vec![Key::<View>::DEFAULT.id()]);
}

#[test]
fn a_normalizer_without_options_records_config_zero() {
    let doc = run_chained(&[], "text");
    let p = doc.provenance(Key::<View>::named("v1").id()).unwrap();
    assert_eq!(p.annotator, "turbo:normalize:chain");
    assert_eq!(p.version, "0");
    assert_eq!(p.config, Chain::new().config_hash());

    let analyzer = Analyzer::builder().step(Bound::new(NormalizeStep::new(Upper)).write(0, "v")).build().unwrap();
    let mut docs = vec![Document::new("text")];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    let p = docs[0].provenance(Key::<View>::named("v").id()).unwrap();
    assert_eq!((p.annotator, p.config), ("test:upper", 0));
}

#[test]
fn a_chain_step_records_the_hash_of_its_steps() {
    let kinds = [Kind::Ligature, Kind::AfterPeriod];
    let doc = run_chained(&kinds, "\u{FB01}.");
    let p = doc.provenance(Key::<View>::named("v1").id()).unwrap();
    assert_eq!(p.config, chain_of(&kinds).config_hash());
    let reversed = [Kind::AfterPeriod, Kind::Ligature];
    assert_ne!(p.config, chain_of(&reversed).config_hash());
}

#[test]
fn normalize_step_exposes_its_normalizer() {
    let step = NormalizeStep::new(AfterPeriod { marker: "#" });
    assert_eq!(step.normalizer().marker, "#");
}

// 6. Edge cases

#[test]
fn empty_text_stays_empty() {
    for kind in KINDS {
        let (out, a) = normalize(&chain_of(&[kind]), "");
        assert_eq!(out, "");
        assert!(a.edits().is_empty(), "{kind:?}");
        check_chain(&[kind, kind, kind], "");
        check_document(&[kind, kind], "");
    }
    // A step that replaces everything still writes its text for no input.
    let (out, a) = normalize(&ReplaceAll, "");
    assert_eq!(out, "<all>");
    assert_eq!(a.edits().len(), 1);
    check_chain(&[Kind::ReplaceAll, Kind::Upper], "");
}

#[test]
fn text_without_triggers_has_no_edits_through_any_chain() {
    let text = "plain \u{E9}\u{20AC}\u{1F600} text";
    // Upper changes the lowercase ASCII, so leave it out.
    let kinds = [Kind::Ligature, Kind::Spaces, Kind::SoftHyphen, Kind::AbToX, Kind::Identity];
    let (out, a) = normalize(&chain_of(&kinds), text);
    assert_eq!(out, text);
    assert!(a.edits().is_empty());
    assert!(matches!(apply(&chain_of(&kinds), text), Cow::Borrowed(_)));
}

#[test]
fn an_edit_that_covers_the_whole_text() {
    for text in ["ab", "\u{FB01}", "\u{AD}", "   ", "\u{1F600}", "a b"] {
        let (out, a) = normalize(&ReplaceAll, text);
        assert_eq!(out, "<all>");
        assert_eq!(a.edits().len(), 1);
        assert_eq!(a.edits()[0].parent, Span::new(0, text.len() as u32));
        assert_eq!(a.edits()[0].child, Span::new(0, 5));
    }
    // Two stages that each cover everything merge into one edit.
    let chain = Chain::new().then(DeleteAll).then(ReplaceAll);
    let (out, a) = normalize(&chain, "some \u{E9} text");
    assert_eq!(out, "<all>");
    assert_eq!(a.edits().len(), 1);
    assert_eq!(a.span_to_parent(Span::new(1, 2)), Span::new(0, "some \u{E9} text".len() as u32));
}

#[test]
fn a_normalizer_that_deletes_everything() {
    let text = "co\u{AD}op \u{FB01}\u{1F600}";
    let chain = Chain::new().then(Ligature).then(DeleteAll).then(Upper);
    let (out, a) = normalize(&chain, text);
    assert_eq!(out, "");
    assert_eq!(a.edits().len(), 1);
    assert_eq!(a.edits()[0].parent, Span::new(0, text.len() as u32));
    assert_eq!(a.edits()[0].child, Span::new(0, 0));
    assert_eq!(a.span_to_child(Span::new(2, 4)), Span::new(0, 0));
    assert!(matches!(apply(&chain, text), Cow::Owned(s) if s.is_empty()));
    check_chain(&[Kind::Ligature, Kind::DeleteAll, Kind::Upper], text);
    check_document(&[Kind::Ligature, Kind::DeleteAll, Kind::Upper], text);
}

#[test]
fn multi_byte_characters_beside_every_edit_boundary() {
    // Each trigger gets a 2, 3 and 4 byte character on both sides, and is
    // also placed at the very start and end of the text.
    let neighbours = ["\u{E9}", "\u{20AC}", "\u{1F600}", ""];
    let triggers = ["\u{FB01}", "  ", "\u{AD}", ".", "ab", "q", "  \u{AD}  "];
    for left in neighbours {
        for right in neighbours {
            for trigger in triggers {
                let text = format!("{left}{trigger}{right}");
                for a in KINDS {
                    check_chain(&[a], &text);
                    for b in KINDS {
                        check_chain(&[a, b], &text);
                    }
                }
                check_chain(&[Kind::Ligature, Kind::Spaces, Kind::SoftHyphen], &text);
                check_chain(&[Kind::SoftHyphen, Kind::Spaces, Kind::AbToX], &text);
                check_chain(&[Kind::AbToX, Kind::AfterPeriod, Kind::Upper], &text);
                check_document(&[Kind::SoftHyphen, Kind::Spaces], &text);
            }
        }
    }
}

#[test]
fn a_chain_can_be_reused_on_many_texts() {
    let chain = chain_of(&[Kind::Ligature, Kind::Spaces, Kind::SoftHyphen]);
    for &text in INPUTS {
        let first = normalize(&chain, text);
        let second = normalize(&chain, text);
        assert_eq!(first.0, second.0);
        assert_eq!(first.1, second.1);
    }
}

// Known problem in turbo-analysis, outside this crate

#[test]
fn alignment_maps_an_empty_source_span_at_an_insertion() {
    let (_, a) = normalize(&AfterPeriod::default(), "a.b");
    // The empty span at byte 2 sits where "|" was inserted. Its start maps
    // to after the insertion and its end to before it, which is a span
    // that ends before it starts.
    let mapped = a.span_to_child(Span::new(2, 2));
    assert!(mapped.start <= mapped.end);
}

#[test]
fn alignment_maps_an_empty_view_span_at_a_deletion() {
    let (_, a) = normalize(&SoftHyphen, "a\u{AD}b");
    // The empty span at byte 1 of "ab" sits where the soft hyphen was.
    let mapped = a.span_to_parent(Span::new(1, 1));
    assert!(mapped.start <= mapped.end);
}
