use super::*;
use crate::{PiiDetectorRef, PiiEvidence, PiiMention};
use turbo_analysis::{Provenance, ViewBuilder};

const TEXT: &str = "mail ann@x.io or 555-1234 now";

/// A mention by the test detector, which [`doc_with`] registers at 0.
fn mention(start: u32, end: u32, kind: PiiKind) -> PiiMention<'static> {
    PiiMention {
        span: Span::new(start, end),
        kind,
        detector: PiiDetectorRef::new(0),
        evidence: &[PiiEvidence::Shape],
        score: None,
        normalized: "v",
    }
}

/// The source with its pii layer and a derived view replacing 5..12 with
/// `[E]` and 16..24 with `[P]`, the view's id returned.
fn doc() -> (Document, ViewId) {
    doc_with(&[(5, 12, PiiKind::Email), (16, 24, PiiKind::Phone), (20, 24, PiiKind::Phone)], true)
}

/// The source with a pii layer of `mentions` and the derived view, which
/// replaces nothing unless `replace`.
fn doc_with(mentions: &[(u32, u32, PiiKind)], replace: bool) -> (Document, ViewId) {
    let mut doc = Document::new(TEXT);
    let mut pii = Pii::new(ViewId::SOURCE);
    pii.detector("test", "1").unwrap();
    for &(start, end, kind) in mentions {
        pii.push(mention(start, end, kind));
    }
    doc.add(&Key::DEFAULT, pii, Provenance::by("test")).unwrap();
    let src = doc.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(TEXT);
    if replace {
        b.keep(5);
        b.replace(12, "[E]");
        b.keep(16);
        b.replace(24, "[P]");
    }
    let (text, alignment) = b.finish();
    doc.add(&Key::<View>::named("redacted"), View::derived(&src, text, alignment), Provenance::by("test")).unwrap();
    let id = doc.get_key(&Key::<View>::named("redacted")).unwrap().id();
    (doc, id)
}

fn edits(view: ViewId, second_members: &[Ix<Pii>]) -> PiiEdits {
    let mut e = PiiEdits::new(view, 7);
    e.push(PiiEdit {
        span: Span::new(5, 8),
        source: Span::new(5, 12),
        members: &[Ix::new(0)],
        rule: PiiRule::Default,
        strategy: PiiStrategyKind::Static,
        rule_id: "",
    });
    e.push(PiiEdit {
        span: Span::new(12, 15),
        source: Span::new(16, 24),
        members: second_members,
        rule: PiiRule::Kind(PiiKind::Phone),
        strategy: PiiStrategyKind::Static,
        rule_id: "",
    });
    e
}

fn add(doc: &mut Document, e: PiiEdits) -> Result<(), turbo_analysis::Error> {
    let mut p = Provenance::by("test");
    p.reads.push(Key::<Pii>::DEFAULT.id());
    doc.add(&Key::DEFAULT, e, p)
}

#[test]
fn strategy_ids_are_stable_and_round_trip() {
    for (i, k) in PiiStrategyKind::ALL.iter().enumerate() {
        assert_eq!(usize::from(k.id()), i + 1);
        assert_eq!(PiiStrategyKind::from_id(k.id()), Some(*k));
    }
    assert_eq!(PiiStrategyKind::from_id(0), None);
    assert_eq!(PiiStrategyKind::from_id(12), Some(PiiStrategyKind::ZipSafeHarbor));
    assert_eq!(PiiStrategyKind::ZipSafeHarbor.name(), "zip-safe-harbor");
    assert_eq!(PiiStrategyKind::from_id(13), Some(PiiStrategyKind::AgeSafeHarbor));
    assert_eq!(PiiStrategyKind::AgeSafeHarbor.name(), "age-safe-harbor");
    assert_eq!(PiiStrategyKind::from_id(14), Some(PiiStrategyKind::DateYear));
    assert_eq!(PiiStrategyKind::from_id(15), Some(PiiStrategyKind::DateShift));
    assert_eq!(PiiStrategyKind::from_id(16), Some(PiiStrategyKind::DateRelative));
    assert_eq!(PiiStrategyKind::DateRelative.name(), "date-relative");
    assert_eq!(PiiStrategyKind::from_id(17), Some(PiiStrategyKind::DateSafeHarbor));
    assert_eq!(PiiStrategyKind::DateSafeHarbor.name(), "date-safe-harbor");
    assert_eq!(PiiStrategyKind::from_id(18), None);
    assert_eq!(PiiRule::Default.code(), 0);
    assert_eq!(PiiRule::Kind(PiiKind::Phone).code(), PiiKind::Phone.code());
}

#[test]
fn edits_matching_the_alignment_and_the_mentions_are_taken() {
    let (mut doc, view) = doc();
    let e = edits(view, &[Ix::new(1), Ix::new(2)]);
    assert_eq!(e.members_of(Ix::new(1)), &[Ix::new(1), Ix::new(2)]);
    add(&mut doc, e).unwrap();
}

#[test]
fn edits_missing_a_covered_mention_are_refused() {
    let (mut doc, view) = doc();
    let err = add(&mut doc, edits(view, &[Ix::new(1)])).unwrap_err();
    assert!(format!("{err:?}").contains("exactly the mentions"), "{err:?}");
}

#[test]
fn edits_disagreeing_with_the_alignment_are_refused() {
    let (mut doc, view) = doc();
    let mut e = edits(view, &[Ix::new(1), Ix::new(2)]);
    e.source[1] = Span::new(17, 24);
    let err = add(&mut doc, e).unwrap_err();
    assert!(format!("{err:?}").contains("alignment"), "{err:?}");
}

#[test]
fn edits_over_the_source_are_refused() {
    let (mut doc, _) = doc();
    let err = add(&mut doc, edits(ViewId::SOURCE, &[Ix::new(1), Ix::new(2)])).unwrap_err();
    assert!(format!("{err:?}").contains("derived view"), "{err:?}");
}

#[test]
fn column_rules_are_checked() {
    let mut e = edits(ViewId(1), &[Ix::new(2), Ix::new(1)]);
    assert!(e.check().unwrap_err().contains("out of order"));
    e = edits(ViewId(1), &[]);
    assert!(e.check().unwrap_err().contains("covers no mention"));
    e = edits(ViewId(1), &[Ix::new(1)]);
    e.source[1] = Span::new(6, 24);
    assert!(e.check().unwrap_err().contains("overlaps"));
    e.source[1] = Span::new(16, 16);
    assert!(e.check().unwrap_err().contains("empty source"));
}

#[test]
fn a_pending_view_is_bound_and_a_real_one_kept() {
    let mut e = edits(ViewId::PENDING, &[Ix::new(1)]);
    e.bind_pending_view(ViewId(3));
    assert_eq!(e.spans.view(), ViewId(3));
    assert_eq!(e.spans.get(Ix::new(1)), Span::new(12, 15));
    e.bind_pending_view(ViewId(4));
    assert_eq!(e.spans.view(), ViewId(3));
}

fn refused(doc: &mut Document, e: PiiEdits) -> String {
    format!("{:?}", add(doc, e).unwrap_err())
}

#[test]
fn edits_dropping_an_alignment_edit_are_refused() {
    let (mut first_doc, view) = doc();
    let mut e = PiiEdits::new(view, 7);
    let first = edits(view, &[Ix::new(1), Ix::new(2)]);
    e.push(first.get(Ix::new(0)));
    assert!(refused(&mut first_doc, e).contains("every edit of the view's alignment"));
    let (mut again, view) = doc();
    assert!(refused(&mut again, PiiEdits::new(view, 7)).contains("every edit of the view's alignment"));
}

#[test]
fn an_edit_on_an_unchanged_stretch_is_refused() {
    let (mut doc, view) = doc();
    let mut e = edits(view, &[Ix::new(1), Ix::new(2)]);
    // 0..4 maps to itself both ways, but the alignment made no edit there.
    e.source[0] = Span::new(0, 4);
    e.spans = {
        let mut s = Spans::new(view);
        s.push(Span::new(0, 4));
        s.push(Span::new(12, 15));
        s
    };
    assert!(refused(&mut doc, e).contains("disagrees with the view's alignment"));
}

#[test]
fn a_mention_under_no_edit_is_refused() {
    let (mut doc, view) = doc_with(&[(0, 4, PiiKind::Email), (5, 12, PiiKind::Email), (16, 24, PiiKind::Phone)], true);
    let mut e = PiiEdits::new(view, 7);
    let base = edits(view, &[Ix::new(2)]);
    e.push(PiiEdit { members: &[Ix::new(1)], ..base.get(Ix::new(0)) });
    e.push(base.get(Ix::new(1)));
    assert!(refused(&mut doc, e).contains("pii mention 0 is under no edit"));
}

#[test]
fn an_empty_layer_is_checked_against_its_view_and_pii() {
    let (mut doc, view) = doc_with(&[], false);
    add(&mut doc, PiiEdits::new(view, 0)).unwrap();
    let (mut doc, view) = doc_with(&[(5, 12, PiiKind::Email)], false);
    assert!(refused(&mut doc, PiiEdits::new(view, 0)).contains("pii mention 0 is under no edit"));
    // No pii layer at all: refused before the edits are looked at.
    let mut bare = Document::new(TEXT);
    assert!(add(&mut bare, PiiEdits::new(ViewId::SOURCE, 0)).is_err());
}

/// The edits of [`edits`] as made by a named policy: a rule id on the
/// second edit and the policy's digest.
fn policy_edits(view: ViewId) -> PiiEdits {
    let base = edits(view, &[Ix::new(1), Ix::new(2)]);
    let mut e = PiiEdits::new(view, 7);
    e.push(base.get(Ix::new(0)));
    e.push(PiiEdit { rule_id: "phones", ..base.get(Ix::new(1)) });
    e.policy_digest = Some(PolicyDigest::from_bytes([9; 32]));
    e
}

#[test]
fn rule_ids_and_the_policy_digest_round_trip_and_enter_the_hash() {
    use turbo_analysis::export::{DocumentGraph, export, import};
    let (mut doc, view) = doc();
    let named = policy_edits(view);
    assert_eq!((named.get(Ix::new(0)).rule_id, named.get(Ix::new(1)).rule_id), ("", "phones"));
    let plain = edits(view, &[Ix::new(1), Ix::new(2)]);
    assert_ne!(named.content_hash(), plain.content_hash());
    let mut other = policy_edits(view);
    other.policy_digest = Some(PolicyDigest::from_bytes([8; 32]));
    assert_ne!(named.content_hash(), other.content_hash());
    add(&mut doc, named).unwrap();
    let codecs = crate::export::codecs();
    let bytes = export(&doc, &codecs).unwrap().to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    let e = back.get::<PiiEdits>().unwrap();
    assert_eq!(e.rule_id.iter().collect::<Vec<_>>(), ["", "phones"]);
    assert_eq!(e.policy_digest, Some(PolicyDigest::from_bytes([9; 32])));
    assert_eq!(export(&back, &codecs).unwrap().to_bytes().unwrap(), bytes);
}

#[test]
fn edits_naming_no_policy_round_trip_without_one() {
    use turbo_analysis::export::{DocumentGraph, export, import};
    let (mut doc, view) = doc();
    add(&mut doc, edits(view, &[Ix::new(1), Ix::new(2)])).unwrap();
    let codecs = crate::export::codecs();
    let bytes = export(&doc, &codecs).unwrap().to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    let e = back.get::<PiiEdits>().unwrap();
    assert_eq!((e.policy_digest, e.rule_id.iter().all(str::is_empty)), (None, true));
}

#[test]
fn a_rule_id_column_of_the_wrong_length_is_refused() {
    let (_, view) = doc();
    let mut e = policy_edits(view);
    e.rule_id.push("extra");
    assert_eq!(e.check(), Err("every column as long as the spans".into()));
}
