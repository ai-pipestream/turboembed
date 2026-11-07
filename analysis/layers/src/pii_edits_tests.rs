use super::*;
use crate::{PiiConfidence, PiiMention};
use turbo_analysis::{Provenance, ViewBuilder};

const TEXT: &str = "mail ann@x.io or 555-1234 now";

fn mention(start: u32, end: u32, kind: PiiKind) -> PiiMention<'static> {
    PiiMention { span: Span::new(start, end), kind, confidence: PiiConfidence::Format, normalized: "v" }
}

/// The source with its pii layer and a derived view replacing 5..12 with
/// `[E]` and 16..24 with `[P]`, the view's id returned.
fn doc() -> (Document, ViewId) {
    let mut doc = Document::new(TEXT);
    let mut pii = Pii::new(ViewId::SOURCE);
    pii.push(mention(5, 12, PiiKind::Email));
    pii.push(mention(16, 24, PiiKind::Phone));
    pii.push(mention(20, 24, PiiKind::Phone));
    doc.add(&Key::DEFAULT, pii, Provenance::by("test")).unwrap();
    let src = doc.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(TEXT);
    b.keep(5);
    b.replace(12, "[E]");
    b.keep(16);
    b.replace(24, "[P]");
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
    });
    e.push(PiiEdit {
        span: Span::new(12, 15),
        source: Span::new(16, 24),
        members: second_members,
        rule: PiiRule::Kind(PiiKind::Phone),
        strategy: PiiStrategyKind::Static,
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
    assert_eq!(PiiStrategyKind::from_id(10), None);
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
