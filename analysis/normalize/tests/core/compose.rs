//! `normalize`, `apply` and `compose`, and chains of one, two and three
//! steps.

use super::*;

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
